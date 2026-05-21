import argparse
import ast
import logging
import os
import random
import re
import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from logsage.auto_resume_policy.attribution_classes import *
from logsage.auto_resume_policy.error_attribution import (
    CONTEXT_SIZE,
    get_proposed_solution_cat,

)

from logsage.auto_resume_policy.error_attribution import get_attribution, get_auto_resume, get_proposed_solution_policies
from logsage.auto_resume_policy.util_postprocessing import get_auto_resume_postprocessing
from logsage.auto_resume_policy.prompts import template_post_error_check
from logsage.auto_resume_policy.error_extraction import return_application_errors, \
    return_application_errors_rt, finished_validation
from logsage.auto_resume_policy.utils import chunk_indices

from nvidia_resiliency_ext.attribution.base import (
    AttributionState,
    NVRxAttribution,
    effective_run_or_init_config,
    normalize_attribution_args,
)
from nvidia_resiliency_ext.attribution.logging_utils import bounded_log_value
from nvidia_resiliency_ext.attribution.orchestration.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_P,
    resolved_llm_runtime_kwargs,
)
from nvidia_resiliency_ext.attribution.orchestration.llm_output import logsage_recommendation
from nvidia_resiliency_ext.attribution.orchestration.types import (
    RECOMMENDATION_CONTINUE,
    RECOMMENDATION_RESTART,
    RECOMMENDATION_STOP,
    RECOMMENDATION_UNKNOWN,
    LogSageAnalysisResult,
    RawAnalysisResultItem,
)

logger = logging.getLogger(__name__)

LogSageCycleFields = tuple[str, str, str, str, str]

FINISHED_STATUS_LLM_FAILURE = "LLM_FAILURE"
FINISHED_STATUS_SLURM_CANCELLED = "SLURM_CANCELLED"
FINISHED_STATUS_SLURM_CANCELLED_JOB_REQUEUE = "SLURM_CANCELLED_JOB_REQUEUE"
FINISHED_STATUS_APPLICATION_DONE = "APPLICATION_DONE"
# pattern-based (not exact match)
FINISHED_STATUS_SLURM_CANCELLED_TIME_LIMIT = "SLURM_CANCELLED_TIME_LIMIT"
FINISHED_STATUS_SLURM_CANCELLED_PREEMPTION_REGEX = re.compile(r"slurmstepd.*DUE TO PREEMPTION")

RESTART_IMMEDIATE = "RESTART IMMEDIATE"
STOP_NO_RESTART = "STOP - DONT RESTART IMMEDIATE"

ATTR_LLM_FAILURE = "LLM FAILURE"
ATTR_SLURM_STEP_CANCELLED = "SLURM STEP CANCELLED"
ATTR_SLURM_STEP_CANCELLED_JOB_REQUEUE = "SLURM STEP CANCELLED JOB REQUEUE"
ATTR_APPLICATION_DONE = "APPLICATION DONE"
ATTR_ERRORS_NOT_FOUND = "ERRORS NOT FOUND"
ATTR_NO_LOGS = "NO LOGS"
ATTR_SLURM_CANCELLED_DUE_TO_PREEMPTION = "SLURM CANCELLED DUE TO PREEMPTION"
LOGSAGE_LLM_ENDPOINT_FAILED = "LLM ENDPOINT FAILED"


MARKER_NEW_RUN_DIR_ADDED = "[sbatch_script]: New run dir added:"


def _previous_path(path: str) -> str | None:
    """Return ``path`` with its last digit-run decremented by one.

    Returns None if the path has no digits or the last number is 0.
    """
    match = re.search(r"(\d+)(?!.*\d)", path)
    if not match:
        return None
    num = int(match.group(1))
    if num <= 0:
        return None
    start, end = match.span(1)
    return path[:start] + str(num - 1) + path[end:]


def _cycle_counter_key(path: str) -> str:
    """Strip a trailing ``_<digits>`` from the filename stem so per-cycle
    paths (``nvrx_0.log``, ``nvrx_1.log``, ...) share one
    ``cycle_counter_dict`` entry.
    """
    stem, ext = os.path.splitext(path)
    stem = re.sub(r"_\d+$", "", stem)
    return stem + ext


def _action_from_logsage_head(head: str) -> str:
    normalized = head.strip().upper()
    if STOP_NO_RESTART in normalized or normalized.startswith("STOP"):
        return RECOMMENDATION_STOP
    if RESTART_IMMEDIATE in normalized:
        return RECOMMENDATION_RESTART
    if ATTR_ERRORS_NOT_FOUND in normalized:
        return RECOMMENDATION_CONTINUE
    if ATTR_NO_LOGS in normalized:
        return RECOMMENDATION_UNKNOWN
    if ATTR_LLM_FAILURE in normalized or LOGSAGE_LLM_ENDPOINT_FAILED in normalized:
        return RECOMMENDATION_UNKNOWN
    return RECOMMENDATION_UNKNOWN


def _parse_issue_list(inner: str) -> list[str]:
    inner = inner.strip()
    if not inner:
        return []
    literal_parsed = True
    try:
        parsed = ast.literal_eval(f"[{inner}]")
    except (SyntaxError, ValueError, RecursionError):
        literal_parsed = False
        parsed = [inner]
    if not isinstance(parsed, list):
        parsed = [parsed]

    issues: list[str] = []
    for item in parsed:
        if item is None:
            continue
        if isinstance(item, str):
            parts = [item] if literal_parsed or "," not in item else item.split(",")
        else:
            parts = [str(item)]
        for part in parts:
            issue = part.strip().strip("'\"")
            if issue:
                issues.append(issue)
    return issues


def _extract_bracketed_issue_list(attribution_text: str, label: str) -> Optional[str]:
    marker = f"{label}:"
    marker_index = attribution_text.find(marker)
    if marker_index < 0:
        return None
    start = attribution_text.find("[", marker_index + len(marker))
    if start < 0:
        return None

    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(attribution_text[start:], start=start):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0:
                return attribution_text[start + 1 : index]
    return None


def _extract_issue_lists(attribution_text: str) -> Tuple[list[str], list[str]]:
    primary: list[str] = []
    secondary: list[str] = []
    primary_inner = _extract_bracketed_issue_list(attribution_text, "Primary issues")
    secondary_inner = _extract_bracketed_issue_list(attribution_text, "Secondary issues")
    if primary_inner is not None:
        primary = _parse_issue_list(primary_inner)
    if secondary_inner is not None:
        secondary = _parse_issue_list(secondary_inner)
    return primary, secondary


def _result_item_from_logsage_fields(
    fields: LogSageCycleFields,
    *,
    raw_text: str,
    action: str,
) -> RawAnalysisResultItem:
    auto_resume = str(fields[0]) if len(fields) > 0 else ""
    auto_resume_explanation = str(fields[1]) if len(fields) > 1 else ""
    attribution_field = str(fields[2]) if len(fields) > 2 else ""
    checkpoint_saved = str(fields[4]) if len(fields) > 4 else "false"

    attribution_text = ""
    if "Attribution:" in attribution_field:
        attribution_text = (
            attribution_field.split("Attribution:", 1)[1]
            .strip()
            .replace('"\\', "")
            .replace('\\"', "")
        )

    checkpoint_saved_flag = 0
    if checkpoint_saved.strip().lower() != "false":
        checkpoint_saved_flag = 1
    primary_issues, secondary_issues = _extract_issue_lists(attribution_text)

    return RawAnalysisResultItem(
        raw_text=raw_text,
        auto_resume=auto_resume,
        auto_resume_explanation=auto_resume_explanation,
        attribution_text=attribution_text,
        checkpoint_saved_flag=checkpoint_saved_flag,
        action=action,
        primary_issues=primary_issues,
        secondary_issues=secondary_issues,
    )


def attribution_from_finished_status(
    app_data,
    application_errors_list_unique,
) -> ErrorAttribution:
    """Build ErrorAttribution when no application errors were found,
    based solely on app_data.finished status.
    """
    finished = app_data.finished

    if finished == FinishedStatus.LLM_FAILURE:
        logger.info("LLM failure")
        return ErrorAttribution(
            application_errors_full=[],
            application_errors_unique=application_errors_list_unique,
            auto_resume=AutoResumeAction.LLM_FAILURE,
            auto_resume_verbose=AutoResumeAction.LLM_FAILURE,
            attribution=Attribution.LLM_FAILURE,
            infra_category="",
            temp_category="",
            single_multiple="",
            cor_category="",
        )

    if finished == FinishedStatus.SLURM_CANCELLED:
        logger.info("Slurm cancelled")
        return ErrorAttribution(
            application_errors_full=[],
            application_errors_unique=application_errors_list_unique,
            auto_resume=AutoResumeAction.RESTART_IMMEDIATE,
            auto_resume_verbose="",
            attribution=Attribution.SLURM_STEP_CANCELLED,
            infra_category="",
            temp_category="",
            single_multiple="",
            cor_category="",
        )

    if finished == FinishedStatus.SLURM_CANCELLED_JOB_REQUEUE:
        logger.info("Slurm cancelled due to job requeue")
        return ErrorAttribution(
            application_errors_full=[],
            application_errors_unique=application_errors_list_unique,
            auto_resume=AutoResumeAction.RESTART_IMMEDIATE,
            auto_resume_verbose="",
            attribution=Attribution.SLURM_STEP_CANCELLED_JOB_REQUEUE,
            infra_category="",
            temp_category="",
            single_multiple="",
            cor_category="",
        )

    if FinishedStatus.SLURM_CANCELLED_TIME_LIMIT in finished:
        logger.info("Slurm cancelled due to time limit")
        return ErrorAttribution(
            application_errors_full=[],
            application_errors_unique=application_errors_list_unique,
            auto_resume=AutoResumeAction.STOP_NO_RESTART,
            auto_resume_verbose="",
            attribution=finished.replace("_", " "),
            infra_category="",
            temp_category="",
            single_multiple="",
            cor_category="",
        )

    if finished == FinishedStatus.APPLICATION_DONE:
        logger.info(Attribution.APPLICATION_DONE)
        return ErrorAttribution(
            application_errors_full=[],
            application_errors_unique=application_errors_list_unique,
            auto_resume=AutoResumeAction.STOP_NO_RESTART,
            auto_resume_verbose="",
            attribution=Attribution.APPLICATION_DONE,
            infra_category="",
            temp_category="",
            single_multiple="",
            cor_category="",
        )

    return ErrorAttribution(
        application_errors_full=[],
        application_errors_unique=application_errors_list_unique,
        auto_resume=AutoResumeAction.ERRORS_NOT_FOUND,
        auto_resume_verbose=AutoResumeAction.ERRORS_NOT_FOUND,
        attribution=Attribution.ERRORS_NOT_FOUND,
        infra_category="",
        temp_category="",
        single_multiple="",
        cor_category="",
    )

def lines_after(lines, needle):
    for i, line in enumerate(lines):
        if needle in line:
            return lines[i + 1 :]
    return lines


def chunk_logs_strict(lines):
    """Chunks logs strictly between:
    - START: The LAST occurrence of Cycle N
    - END: The LAST occurrence of Cycle N+1

    Lines after the highest Cycle number are ignored.
    If no 'Cycle' markers are found, returns all lines as Cycle 0.
    """
    # Regex to match the profiling line
    cycle_pattern = re.compile(r"profiling\.py:.*Cycle:\s*(\d+)")

    # Step 1: Find the LAST index for every cycle number
    last_cycle_indices = {}
    start_cycle_indices = {}
    for index, line in enumerate(lines):
        match = cycle_pattern.search(line)
        if match:
            cycle_num = int(match.group(1))
            if cycle_num not in start_cycle_indices:
                start_cycle_indices[cycle_num] = index
            last_cycle_indices[cycle_num] = index

    # Sort cycles (0, 1, 2...)
    sorted_cycles = sorted(last_cycle_indices.keys())

    final_chunks = {}

    # --- NEW LOGIC START ---
    # If no cycles were found, return all lines as Cycle 0
    if not sorted_cycles:
        final_chunks[0] = lines
        return final_chunks
    # --- NEW LOGIC END ---

    # Step 2: Create chunks ONLY when we have both a Start (N) and an End (N+1)
    # We iterate up to len() - 1 because the last cycle in the list
    # serves only as the end boundary for the previous one.
    for i in range(len(sorted_cycles)):
        curr_cycle = sorted_cycles[i]
        start_index = start_cycle_indices[curr_cycle]
        if i == len(sorted_cycles) - 1:
            end_index = None
        else:
            next_cycle = sorted_cycles[i + 1]  # This is N+1
            end_index = start_cycle_indices[next_cycle]

        # Extract lines between LAST Cycle N and LAST Cycle N+1
        raw_chunk = lines[start_index:end_index]

        # Step 3: Remove marker lines
        clean_chunk = [line for line in raw_chunk if not cycle_pattern.search(line)]

        clean_chunk = lines_after(clean_chunk, "FT: initialized")

        final_chunks[curr_cycle] = clean_chunk

    return final_chunks


def _log_analysis_retry_config() -> tuple[int, float, float, float]:
    retries = int(os.getenv("NVRX_LOG_ANALYSIS_LLM_RETRIES", "3"))
    initial_backoff = float(os.getenv("NVRX_LOG_ANALYSIS_LLM_INITIAL_BACKOFF_SEC", "1.0"))
    max_backoff = float(os.getenv("NVRX_LOG_ANALYSIS_LLM_MAX_BACKOFF_SEC", "8.0"))
    jitter = float(os.getenv("NVRX_LOG_ANALYSIS_LLM_JITTER_SEC", "0.25"))
    return retries, initial_backoff, max_backoff, jitter


def _finished_status_name(status: Any) -> str:
    return getattr(status, "name", status)


def _sleep_with_backoff(
    attempt: int, retries: int, backoff: float, max_backoff: float, jitter: float
) -> float:
    sleep_for = min(backoff, max_backoff) + random.uniform(0.0, jitter)
    logger.info(
        "Retrying log-analysis LLM in %.2fs after attempt %d/%d",
        sleep_for,
        attempt,
        retries,
    )
    time.sleep(sleep_for)
    return min(backoff * 2, max_backoff)


def _retry_return_application_errors(
    llm: ChatOpenAI, lines: list[str], cache_dict: LRUCache
) -> ApplicationData:
    retries, initial_backoff, max_backoff, jitter = _log_analysis_retry_config()
    backoff = initial_backoff
    last_status = None

    for attempt in range(1, retries + 1):
        app_data = return_application_errors(llm, lines, cache_dict)
        status_name = _finished_status_name(app_data.finished)
        if status_name != FINISHED_STATUS_LLM_FAILURE:
            return app_data

        last_status = status_name
        if attempt == retries:
            logger.error(
                "Log-analysis extraction failed after %d attempts; last status: %s",
                retries,
                last_status,
            )
            return app_data

        backoff = _sleep_with_backoff(attempt, retries, backoff, max_backoff, jitter)

    return app_data


def _retry_return_application_errors_rt(
    llm: ChatOpenAI, lines: list[str], cache_dict: LRUCache, temporal_cache: dict[str, str]
) -> ApplicationData:
    retries, initial_backoff, max_backoff, jitter = _log_analysis_retry_config()
    backoff = initial_backoff
    last_status = None

    for attempt in range(1, retries + 1):
        app_data = return_application_errors_rt(llm, lines, cache_dict, temporal_cache)
        status_name = _finished_status_name(app_data.finished)
        if status_name != FINISHED_STATUS_LLM_FAILURE:
            return app_data

        last_status = status_name
        if attempt == retries:
            logger.error(
                "Log-analysis extraction failed after %d attempts; last status: %s",
                retries,
                last_status,
            )
            return app_data

        backoff = _sleep_with_backoff(attempt, retries, backoff, max_backoff, jitter)

    return app_data


def _with_exponential_backoff(llm_call, checkpoint_saved: bool) -> tuple[str, str, str, str, str]:
    retries, initial_backoff, max_backoff, jitter = _log_analysis_retry_config()
    backoff = initial_backoff
    last_error = "no attempts made (retries=0)"
    fallback = (
        ATTR_LLM_FAILURE,
        ATTR_LLM_FAILURE,
        ATTR_LLM_FAILURE,
        ATTR_LLM_FAILURE,
        str(checkpoint_saved),
    )

    for attempt in range(1, retries + 1):
        try:
            result = llm_call()
            if result and not any(field == LOGSAGE_LLM_ENDPOINT_FAILED for field in result[:4]):
                return result
            last_error = LOGSAGE_LLM_ENDPOINT_FAILED
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Log-analysis LLM attempt %d/%d failed: %s", attempt, retries, exc)

        if attempt == retries:
            logger.error(
                "Log-analysis LLM failed after %d attempts; last error: %s",
                retries,
                last_error,
            )
            return fallback

        backoff = _sleep_with_backoff(attempt, retries, backoff, max_backoff, jitter)

    logger.error(
        "Log-analysis LLM failed after %d attempts; last error: %s",
        retries,
        last_error,
    )
    return fallback


class NVRxLogAnalyzer(NVRxAttribution):
    def __init__(self, args: Union[argparse.Namespace, Mapping[str, Any]]):
        from nvidia_resiliency_ext.attribution.api_keys import (
            llm_api_key_missing_message,
            load_llm_api_key,
        )

        self._init_config = normalize_attribution_args(args)
        self.api_key = load_llm_api_key()
        if not self.api_key:
            raise ValueError(llm_api_key_missing_message())
        logger.debug("API key loaded (length=%d)", len(self.api_key))
        llm_kwargs = resolved_llm_runtime_kwargs(self._init_config)
        logger.debug("Using model: %s", llm_kwargs["model"])
        self.lru_cache = LRUCache(100_000)
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            **llm_kwargs,
        )
        self.temporal_cache_dict = {}
        self.cycle_counter_dict = {}
        self.job_inline_data_dict = {}
        self.attribution_dict = {}
        self.exclude_nvrx_logs = bool(self._init_config.get("exclude_nvrx_logs", False))
        self.is_per_cycle = bool(self._init_config.get("is_per_cycle", False))
        self.is_streaming_logs = bool(self._init_config.get("is_streaming_logs", False))
        self.repeated_amount = int(self._init_config.get("repeated_amount", 3)),
        self.stop_accumulating_count = int(self._init_config.get("stop_accumulating_count", 3)),
        self.logs_minutes_before_job_end = int(self._init_config.get("logs_minutes_before_job_end", 20)),
        self.chunks_per_time = int(self._init_config.get("chunks_per_time", 5)),
        if self.is_streaming_logs:
            super().__init__(
                preprocess_input=self._analyze_logs_rt_dispatch,
                attribution=self.llm_analyze_rt,
                output_handler=self.print_output,
            )
        else:
            super().__init__(
                preprocess_input=self.analyze_logs,
                attribution=self.llm_analyze,
                output_handler=self.print_output,
            )

    @property
    def init_config(self) -> Dict[str, Any]:
        return dict(self._init_config)


    async def _analyze_logs_rt_dispatch(self) -> ErrorAttribution | None:
        cfg = effective_run_or_init_config(self._init_config)
        if cfg.get("job_stage") == "end":
            return await self.analyze_logs_rt_end()
        return await self.analyze_logs_rt_start()


    async def llm_analyze_rt(self, rt_result: ErrorAttribution | None) -> list[LogSageCycleFields]:
        if rt_result is None:
            return []
        return [(
            str(rt_result.auto_resume),
            str(rt_result.auto_resume_verbose),
            f"Attribution: Primary issues: [{rt_result.attribution}], Secondary issues: []",
            "",
            str(getattr(rt_result, "checkpoint_saved", False)),
        )]


    async def analyze_logs_rt_start(self) -> list[ApplicationData]:
        """
        Analyzes the logs and returns the application errors.

        Args:
            input_data: The input data to analyze.

        Returns:
            application_errors_list_full_purified: The application errors list full purified.
            application_errors_list_full: The application errors list full.
            application_errors_list_full_purified_with_rank: The application errors list full purified with rank.
            application_errors_list_full_with_rank: The application errors list full with rank.
            error_type: The error type.
            error_type_with_rank: The error type with rank.
            error_type_with_rank_and_rank: The error type with rank and rank.
            error_type_with_rank_and_rank_and_rank: The error type with rank and rank and rank.

        """
        cfg = effective_run_or_init_config(self._init_config)
        path = cfg["log_path"]

        llm = self.llm
        cache_dict = self.lru_cache

        cycle_counter = int(cfg.get("cycle_counter", 0))
        cycle_counter_key = _cycle_counter_key(path)
        if cycle_counter == 0:
            self.cycle_counter_dict[cycle_counter_key] = cycle_counter

        if path not in self.temporal_cache_dict:
            self.temporal_cache_dict[path] = {}
        file_offset = 0
        log_lines: list[str] = []
        empty_logs_stop = self.stop_accumulating_count

        application_log, attribution_raw_chunk, attribution_dict_chunk, hw_category_chunk = None, None, None, None

        while True:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    f.seek(file_offset)
                    new_lines = f.readlines()
                    file_offset = f.tell()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1') as f:
                    f.seek(file_offset)
                    new_lines = f.readlines()
                    file_offset = f.tell()

            if len(new_lines):
                empty_logs_stop = self.stop_accumulating_count
            else:
                empty_logs_stop -= 1

            if empty_logs_stop <= 0:
                break

            log_lines.extend(new_lines)
            attribution_list = []

            chunk_data = _retry_return_application_errors_rt(llm, new_lines, cache_dict, self.temporal_cache_dict[path])
            app_data = chunk_data
            if chunk_data.application_errors_list_full:
                application_log, attribution_raw_chunk, attribution_dict_chunk, hw_category_chunk = get_attribution(
                    llm, app_data, True)
                attribution_list.append(attribution_raw_chunk)

            self.job_inline_data_dict[path].append((file_offset, new_lines, chunk_data, application_log, attribution_raw_chunk, attribution_dict_chunk, hw_category_chunk))

            time.sleep(self.chunks_per_time*60)

        return None


    async def analyze_logs_rt_end(self) -> list[ApplicationData]:
        """
        Analyzes the logs and returns the application errors.

        Args:
            input_data: The input data to analyze.

        Returns:
            application_errors_list_full_purified: The application errors list full purified.
            application_errors_list_full: The application errors list full.
            application_errors_list_full_purified_with_rank: The application errors list full purified with rank.
            application_errors_list_full_with_rank: The application errors list full with rank.
            error_type: The error type.
            error_type_with_rank: The error type with rank.
            error_type_with_rank_and_rank: The error type with rank and rank.
            error_type_with_rank_and_rank_and_rank: The error type with rank and rank and rank.

        """
        cfg = effective_run_or_init_config(self._init_config)
        path = cfg["log_path"]

        s_time = time.time()

        llm = self.llm
        cache_dict = self.lru_cache

        path_previous = _previous_path(path)
        attribution_previous = (
            self.attribution_dict.get(path_previous, '') if path_previous else ''
        )
        cycle_counter = int(cfg.get("cycle_counter", 0))
        cycle_counter_key = _cycle_counter_key(path)
        if cycle_counter == 0:
            self.cycle_counter_dict[cycle_counter_key] = cycle_counter

        if path not in self.temporal_cache_dict:
            self.temporal_cache_dict[path] = {}

        if len(self.job_inline_data_dict[path]):
            file_offset = self.job_inline_data_dict[path][-1][0]
        else:
            file_offset = 0

        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.seek(file_offset)
                new_lines = f.readlines()
                file_offset = f.tell()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                f.seek(file_offset)
                new_lines = f.readlines()
                file_offset = f.tell()

        history = self.job_inline_data_dict.get(path, [])
        num_chunks = int(self.logs_minutes_before_job_end / self.chunks_per_time)
        chunk = []
        for item in history[-num_chunks:]:
            chunk = chunk + item[1]
        chunk = chunk + new_lines

        chunk_data = _retry_return_application_errors_rt(llm, chunk, cache_dict, self.temporal_cache_dict[path])

        last_with_errors = chunk_data
        if last_with_errors.application_errors_list_full:
            indices = [error[2] for error in last_with_errors.application_errors_list_full]
            error_groups = chunk_indices(indices, len(last_with_errors.original_text))

            n_lines = len(last_with_errors.original_text)
            prompt_post_error = ChatPromptTemplate.from_template(template_post_error_check)
            post_error_chain = (
                    {"question": RunnablePassthrough()}
                    | prompt_post_error
                    | llm
                    | StrOutputParser()
            )

            post_error_texts = []
            checked_groups = []
            for group in error_groups:
                if len(group) == 0:
                    continue
                first_idx = int(group[0])
                last_idx = int(group[-1])
                if first_idx < n_lines - last_idx and len(last_with_errors.original_text) > last_idx + 50:
                    post_error_lines = last_with_errors.original_text[last_idx + 1: last_idx + 51]
                    post_error_texts.append("\n".join(post_error_lines)[:CONTEXT_SIZE])
                    checked_groups.append(group)

            post_error_results = post_error_chain.batch(post_error_texts) if post_error_texts else []

            indices_to_remove = set()
            for group, result in zip(checked_groups, post_error_results):
                if result.strip().lower() == "no":
                    indices_to_remove.update(int(idx) for idx in group)

            last_with_errors.application_errors_list_full = [
                error
                for error in last_with_errors.application_errors_list_full
                if error[2] not in indices_to_remove
            ]
        if last_with_errors.application_errors_list_full:
            application_log, attribution_raw_chunk, attribution_dict_chunk, hw_category_chunk = get_attribution(
                llm, last_with_errors, True)
            last_attribution_dict_chunk = attribution_dict_chunk
            last_attribution_raw_chunk = attribution_raw_chunk
            last_application_log_chunk = application_log
            last_hw_category_chunk = hw_category_chunk

        if last_with_errors is None:
            history = self.job_inline_data_dict.get(path, [])
            last_with_errors = (history[-1][2] if len(history) >= 1 else None)

        last_with_errors.checkpoint_saved = any([item[2].checkpoint_saved for item in self.job_inline_data_dict[path]])

        last_with_errors = finished_validation(llm, last_with_errors)

        logger.info("error extraction latency: %s", time.time() - s_time)
        application_errors_full = [error[0] for error in last_with_errors.application_errors_list_full]

        self.temporal_cache_dict.pop(path, None)

        if (
                len(last_with_errors.application_errors_list_full) == 0
                or last_with_errors.finished == FinishedStatus.APPLICATION_DONE
        ):
            attribution_finished = attribution_from_finished_status(last_with_errors, last_with_errors.application_errors_list_unique)
            return attribution_finished

        attribution_output = last_attribution_dict_chunk["attribution"]
        self.attribution_dict[path] = attribution_output
        auto_resume_output, auto_resume_verbose = get_auto_resume(
            llm,
            last_with_errors,
            last_attribution_raw_chunk,
            last_attribution_dict_chunk,
            last_hw_category_chunk,
            last_application_log_chunk,
        )

        is_attribution_current_last = get_auto_resume_postprocessing(
            attribution_output,
            attribution_previous,
            cycle_counter,
            llm,
        )

        if last_with_errors.checkpoint_saved and cycle_counter > 0:
            is_attribution_current_last = False

        if is_attribution_current_last and self.cycle_counter_dict[cycle_counter_key] == self.repeated_amount - 1:
            auto_resume_output = 'STOP - DONT RESTART IMMEDIATE'
            auto_resume_verbose = "Stop job due to repeated issue"

        if is_attribution_current_last:
            if auto_resume_verbose != "Stop job due to repeated issue":
                self.cycle_counter_dict[cycle_counter_key]+=1
        else:
            self.cycle_counter_dict[cycle_counter_key]=1

        logger.info("Policy suggestion and Error attribution started")

        logger.info(auto_resume_output)

        return ErrorAttribution(
            application_errors_full=application_errors_full,
            application_errors_unique=last_with_errors.application_errors_list_unique,
            auto_resume=auto_resume_output,
            auto_resume_verbose=auto_resume_verbose,
            attribution=attribution_output,
        )


    async def analyze_logs(self) -> list[ApplicationData]:
        """
        Analyzes the logs and returns the application errors.

        Args:
            input_data: The input data to analyze.

        Returns:
            application_errors_list_full_purified: The application errors list full purified.
            application_errors_list_full: The application errors list full.
            application_errors_list_full_purified_with_rank: The application errors list full purified with rank.
            application_errors_list_full_with_rank: The application errors list full with rank.
            error_type: The error type.
            error_type_with_rank: The error type with rank.
            error_type_with_rank_and_rank: The error type with rank and rank.
            error_type_with_rank_and_rank_and_rank: The error type with rank and rank and rank.

        """
        cfg = effective_run_or_init_config(self._init_config)
        path = cfg["log_path"]
        is_per_cycle = bool(cfg.get("is_per_cycle", self.is_per_cycle))
        exclude_nvrx = bool(cfg.get("exclude_nvrx_logs", self.exclude_nvrx_logs))
        try:
            with open(path, 'r', encoding='utf-8') as f:
                input_data = f.readlines()
        except UnicodeDecodeError:
            # Fallback for non-UTF-8 or mixed encoding; latin-1 never raises decode errors.
            # Other exceptions (e.g. PermissionError, FileNotFoundError) propagate.
            with open(path, 'r', encoding='latin-1') as f:
                input_data = f.readlines()

        # If is_per_cycle is set, skip filtering and chunking (data is already single-cycle)
        if is_per_cycle:
            logger.info("is_per_cycle=True: skipping nvrx log filtering and cycle chunking")
            chunks = {0: input_data}
        else:
            if exclude_nvrx:
                input_data = [line for line in input_data if "nvidia_resiliency_ext" not in line]
                input_data = [
                    line for line in input_data if "[workload:" not in line or 'Cycle:' in line
                ]
                logger.info(f"Excluded {len(input_data)} lines from the input data")
                with open(os.path.join(os.path.dirname(path), "nvrx_logs_edited.txt"), 'w') as f:
                    f.writelines(input_data)
            chunks = chunk_logs_strict(input_data)  # Splitting the app log to cycles

        # Adding another parser for other application logs marks
        if (
            len(chunks) == 1
            and input_data
            and any(MARKER_NEW_RUN_DIR_ADDED in line for line in input_data)
        ):

            chunks = {}
            current_chunk = []
            cycle = -1  # will become 0 on first marker

            for line in input_data:
                if MARKER_NEW_RUN_DIR_ADDED in line:
                    # start a new chunk
                    cycle += 1
                    current_chunk = []
                    chunks[cycle] = current_chunk

                if cycle >= 0:
                    current_chunk.append(line)

        output_list = [
            _retry_return_application_errors(self.llm, lines, self.lru_cache)
            for cycle, lines in chunks.items()
        ]
        return output_list

    async def llm_analyze(self, output_list: list[ApplicationData]) -> list[LogSageCycleFields]:

        result = []
        logger.info("output_list_size: %s", str(len(output_list)))
        for output in output_list:
            if output.finished == FINISHED_STATUS_APPLICATION_DONE:
                result.append(
                    (
                        STOP_NO_RESTART,
                        "",
                        f"""Attribution: Primary issues: [{ATTR_APPLICATION_DONE}], Secondary issues: []""",
                        "",
                        str(output.checkpoint_saved),
                    )
                )
            elif output.original_text and any(
                FINISHED_STATUS_SLURM_CANCELLED_PREEMPTION_REGEX.search(line)
                for line in output.original_text
            ):
                result.append(
                    (
                        RESTART_IMMEDIATE,
                        "",
                        f"""Attribution: Primary issues: [{ATTR_SLURM_CANCELLED_DUE_TO_PREEMPTION}], Secondary issues: []""",
                        "",
                        str(output.checkpoint_saved),
                    )
                )
            else:
                if len(output.application_errors_list_full):
                    result.append(
                        _with_exponential_backoff(
                            lambda: get_proposed_solution_cat(self.llm, output),
                            checkpoint_saved=output.checkpoint_saved,
                        )
                    )
                else:
                    if output.finished == FINISHED_STATUS_LLM_FAILURE:
                        result.append(
                            (
                                ATTR_LLM_FAILURE,
                                ATTR_LLM_FAILURE,
                                ATTR_LLM_FAILURE,
                                ATTR_LLM_FAILURE,
                                str(output.checkpoint_saved),
                            )
                        )
                    elif output.finished == FINISHED_STATUS_SLURM_CANCELLED:
                        result.append(
                            (
                                RESTART_IMMEDIATE,
                                "",
                                f"""Attribution: Primary issues: [{ATTR_SLURM_STEP_CANCELLED}], Secondary issues: []""",
                                "",
                                str(output.checkpoint_saved),
                            )
                        )
                    elif output.finished == FINISHED_STATUS_SLURM_CANCELLED_JOB_REQUEUE:
                        result.append(
                            (
                                RESTART_IMMEDIATE,
                                "",
                                f"""Attribution: Primary issues: [{ATTR_SLURM_STEP_CANCELLED_JOB_REQUEUE}], Secondary issues: []""",
                                "",
                                str(output.checkpoint_saved),
                            )
                        )
                    elif FINISHED_STATUS_SLURM_CANCELLED_TIME_LIMIT in output.finished:
                        result.append(
                            (
                                STOP_NO_RESTART,
                                "",
                                f"""Attribution: Primary issues: [{output.finished.replace("_", " ")}], Secondary issues: []""",
                                "",
                                str(output.checkpoint_saved),
                            )
                        )
                    elif not output.original_text:
                        result.append(
                            (
                                ATTR_NO_LOGS,
                                ATTR_NO_LOGS,
                                ATTR_NO_LOGS,
                                ATTR_NO_LOGS,
                                str(output.checkpoint_saved),
                            )
                        )
                    else:
                        result.append(
                            (
                                ATTR_ERRORS_NOT_FOUND,
                                ATTR_ERRORS_NOT_FOUND,
                                ATTR_ERRORS_NOT_FOUND,
                                ATTR_ERRORS_NOT_FOUND,
                                str(output.checkpoint_saved),
                            )
                        )
        return result

    async def print_output(
        self, attribution_results: list[LogSageCycleFields]
    ) -> tuple[LogSageAnalysisResult, AttributionState]:
        output_list: list[RawAnalysisResultItem] = []
        overall_state = AttributionState.CONTINUE
        for attribution_result in attribution_results:
            if attribution_result:
                logger.info("attribution_result: %s", bounded_log_value(attribution_result))
                concatenated_result = '\n'.join(str(item) for item in attribution_result)
                head = str(attribution_result[0])
                action = _action_from_logsage_head(head)
                if action == RECOMMENDATION_STOP:
                    overall_state = AttributionState.STOP
                output_list.append(
                    _result_item_from_logsage_fields(
                        attribution_result,
                        raw_text=concatenated_result,
                        action=action,
                    )
                )
        recommendation = logsage_recommendation(output_list, source="log_analyzer")
        return (LogSageAnalysisResult(output_list, recommendation), overall_state)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze collective operations across JSON dump files.'
    )
    parser.add_argument('--log-path', type=str, help='Path to log files')
    parser.add_argument(
        '-m',
        '--model',
        default=DEFAULT_LLM_MODEL,
        help='Model to use for LLM analysis',
    )
    parser.add_argument(
        '-b',
        '--base_url',
        default=DEFAULT_LLM_BASE_URL,
        help='Base URL for the OpenAI-compatible API endpoint',
    )
    parser.add_argument(
        '-t',
        '--temperature',
        type=float,
        default=DEFAULT_LLM_TEMPERATURE,
        help='Temperature for LLM',
    )
    parser.add_argument(
        '-p',
        '--top_p',
        type=float,
        default=DEFAULT_LLM_TOP_P,
        help='Top P for LLM',
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=DEFAULT_LLM_MAX_TOKENS,
        help='Max tokens for LLM',
    )
    parser.add_argument(
        '--exclude_nvrx_logs', action='store_true', help='Exclude nvrx logs from the input data'
    )
    parser.add_argument(
        '--is_per_cycle',
        action='store_true',
        help='Input is already per-cycle data (skip filtering and chunking)',
    )
    parser.add_argument(
        '--emit-stdout',
        action='store_true',
        help='Print final attribution payload to stdout for machine consumers',
    )

    args = parser.parse_args()

    analyzer = NVRxLogAnalyzer(args)
    results = analyzer.run_sync(args)

    if args.emit_stdout:
        for result in results:
            if not result:
                continue
            payload = result[0] if isinstance(result, tuple) else result
            if payload:
                print(payload)


if __name__ == "__main__":
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.getLogger("nvidia_resiliency_ext").setLevel(logging.INFO)
    main()
