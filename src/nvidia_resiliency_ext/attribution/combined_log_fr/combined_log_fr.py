# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import re
from typing import Any, Mapping, Union

from nvidia_resiliency_ext.attribution.api_keys import load_llm_api_key
from nvidia_resiliency_ext.attribution.base import (
    AttributionState,
    NVRxAttribution,
    merged_attribution_config,
    normalize_attribution_args,
)
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
from nvidia_resiliency_ext.attribution.logging_utils import bounded_log_value
from nvidia_resiliency_ext.attribution.orchestration.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_P,
    resolved_llm_runtime_kwargs,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer

from .llm_merge import merge_log_fr_llm, unpack_run_result

logger = logging.getLogger(__name__)

_EXCLUDED_RANKS_LABEL = "list of ranks to be excluded:"


def _excluded_ranks_from_attribution_result(attribution_result: str) -> set[int]:
    """Parse ranks from merge output lines like ``List of ranks to be excluded: 1,2,3``."""
    ranks: set[int] = set()
    for line in str(attribution_result).splitlines():
        if _EXCLUDED_RANKS_LABEL not in line.lower():
            continue
        _, _, rank_text = line.partition(":")
        ranks.update(int(token) for token in re.findall(r"\d+", rank_text))
    return ranks


class CombinedLogFR(NVRxAttribution):
    def __init__(
        self, args: Union[argparse.Namespace, Mapping[str, Any], list[Any], tuple[Any, ...]]
    ):
        ad = normalize_attribution_args(args)
        self.threshold = int(ad.get("threshold", 0))
        super().__init__(
            preprocess_input=self.preprocess_input,
            attribution=self.collective_analysis,
            output_handler=self.print_output,
        )
        self._init_config = ad
        # Resolve once per CombinedLogFR instance so merge_log_fr_llm does not re-read env/files each call.
        self._llm_api_key = load_llm_api_key()

    async def preprocess_input(self) -> dict:
        cfg = merged_attribution_config(self._init_config)
        input_data = cfg["input_data"]
        logger.info("input_data: %s", bounded_log_value(input_data))
        log_result = input_data[0]
        fr_result = input_data[1]
        output: dict[str, Any] = {}
        output['Application Logs'] = log_result
        output['Collective Operations Analysis'] = fr_result
        return output

    async def collective_analysis(self, output: dict) -> str:
        log_result = output['Application Logs']
        fr_result = output['Collective Operations Analysis']

        logger.info("fr_result: %s", bounded_log_value(fr_result))
        cfg = merged_attribution_config(self._init_config)
        return await merge_log_fr_llm(
            log_result,
            fr_result,
            llm_api_key=self._llm_api_key,
            **resolved_llm_runtime_kwargs(cfg),
        )

    async def print_output(self, attribution_result: str) -> tuple[str, AttributionState]:
        excluded_ranks = _excluded_ranks_from_attribution_result(attribution_result)
        logger.info("attribution_result: %s", bounded_log_value(attribution_result))
        if excluded_ranks and len(excluded_ranks) > self.threshold:
            logger.info("excluded_ranks: %s", bounded_log_value(sorted(excluded_ranks)))
            # Standalone merge runner state only; service policy still comes from
            # the explicit recommendation envelope, not FR/merge rank output.
            return attribution_result, AttributionState.STOP
        return attribution_result, AttributionState.CONTINUE


def main():
    parser = argparse.ArgumentParser(
        description='Analyze application logs with PyTorch Flight Recorder dumps.'
    )
    parser.add_argument('--log-path', type=str, required=True, help='Path to application log file')

    parser.add_argument(
        '--fr-path',
        type=str,
        nargs='+',
        required=True,
        help='Path to FR dump directory or files (first path is used for analysis)',
    )
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
        '--pattern', default="_dump_*", help='File pattern to match (default: _dump_*)'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument(
        '-c', '--health-check', action='store_true', help='Show node health check results'
    )
    parser.add_argument(
        '-l', '--llm-analyze', action='store_true', help='Use LLM to analyze the output'
    )
    parser.add_argument(
        '--exclude_nvrx_logs', action='store_true', help='Exclude nvrx logs from the input data'
    )
    parser.add_argument(
        '--threshold', type=int, default=0, help='Threshold for the number of ranks to be excluded'
    )

    args = parser.parse_args()
    args_dict = dict(vars(args))
    # argparse nargs='+' yields a list; CollectiveAnalyzer expects a single fr_path string.
    fr_paths = args_dict.get("fr_path")
    if isinstance(fr_paths, list) and fr_paths:
        args_dict["fr_path"] = fr_paths[0]

    log_analyzer = NVRxLogAnalyzer(args_dict)
    log_result = log_analyzer.run_sync(args_dict)
    fr_kw = {**args_dict, "llm_analyze": False}
    fr_result = CollectiveAnalyzer(fr_kw).run_sync(fr_kw)
    log_actual, _ = unpack_run_result(log_result)
    fr_actual, _ = unpack_run_result(fr_result)
    merge_args = {**args_dict, "input_data": [log_actual, fr_actual]}
    CombinedLogFR(merge_args).run_sync(merge_args)


if __name__ == "__main__":
    main()
