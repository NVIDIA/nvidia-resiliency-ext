import argparse
import ast
import logging
import os
import random
import re
import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from langchain_openai import ChatOpenAI
from logsage.auto_resume_policy.attribution_classes import ApplicationData, LRUCache
from logsage.auto_resume_policy.error_attribution import get_proposed_solution_cat
from logsage.auto_resume_policy.error_extraction import return_application_errors

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
        self.exclude_nvrx_logs = bool(self._init_config.get("exclude_nvrx_logs", False))
        self.is_per_cycle = bool(self._init_config.get("is_per_cycle", False))
        super().__init__(
            preprocess_input=self.analyze_logs,
            attribution=self.llm_analyze,
            output_handler=self.print_output,
        )

    @property
    def init_config(self) -> Dict[str, Any]:
        return dict(self._init_config)

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
