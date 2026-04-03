# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from typing import Any, Mapping, Union

from nvidia_resiliency_ext.attribution.api_keys import load_nvidia_api_key
from nvidia_resiliency_ext.attribution.base import (
    AttributionState,
    NVRxAttribution,
    merged_attribution_config,
    normalize_attribution_args,
)
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer

from .llm_merge import merge_log_fr_llm, unpack_run_result

logger = logging.getLogger(__name__)


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
        self._nvidia_api_key = load_nvidia_api_key()

    async def preprocess_input(self) -> dict:
        cfg = merged_attribution_config(self._init_config)
        input_data = cfg["input_data"]
        logger.info("input_data: %s", cfg["input_data"])
        log_result = input_data[0]
        fr_result = input_data[1]
        output: dict[str, Any] = {}
        output['Application Logs'] = log_result
        output['Collective Operations Analysis'] = fr_result
        return output

    async def collective_analysis(self, output: dict) -> str:
        log_result = output['Application Logs']
        fr_result = output['Collective Operations Analysis']

        logger.info(f"fr_result: {fr_result}")
        cfg = merged_attribution_config(self._init_config)
        return await merge_log_fr_llm(
            log_result,
            fr_result,
            nvidia_api_key=self._nvidia_api_key,
            model=cfg.get("model", "nvidia/qwen/qwen-235b"),
            base_url = cfg.get("base_url", "https://inference-api.nvidia.com/v1"),
            temperature=float(cfg.get("temperature", 0.2)),
            top_p=float(cfg.get("top_p", 0.7)),
            max_tokens=int(cfg.get("max_tokens", 8192)),
        )

    async def print_output(self, attribution_result: str) -> tuple[str, AttributionState]:
        rank_list = []
        logger.info(f"attribution_result: {attribution_result}")
        for line in attribution_result.split('\n'):
            if 'List of ranks to be excluded:' in line:
                rank_list.append(line.split(':')[1].strip())
        if rank_list and len(rank_list) > self.threshold:
            logger.info(f"rank_list: {rank_list}")
            return attribution_result, AttributionState.STOP
        return attribution_result, AttributionState.CONTINUE


def main():
    parser = argparse.ArgumentParser(
        description='Analyze collective operations across JSON dump files.'
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
        default="nvidia/qwen/qwen-235b",
        help='Model to use for LLM analysis',
    )
    parser.add_argument(
        '-b',
        '--base_url',
        default="https://inference-api.nvidia.com/v1",
        help='Base URL for the OpenAI-compatible API endpoint',
    )
    parser.add_argument('-t', '--temperature', type=float, default=0.2, help='Temperature for LLM')
    parser.add_argument('-p', '--top_p', type=float, default=0.7, help='Top P for LLM')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Max tokens for LLM')
    parser.add_argument(
        '--pattern', default="*.json", help='File pattern to match (default: *.json)'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument(
        '-c', '--health-check', action='store_true', help='Show node health check results'
    )
    parser.add_argument(
        '-l', '--llm-analyze', action='store_true', help='Use LLM to analyze the output'
    )
    parser.add_argument(
        '-s', '--scheduling-order-file', default="TP->PP->DP", help='Scheduling order of TP->PP->DP'
    )
    parser.add_argument('--time-spread', action='store_true', help='Show time spread analysis')
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
