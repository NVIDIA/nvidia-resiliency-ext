import argparse
import logging
import os
import re

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from logsage.auto_resume_policy.attribution_classes import ApplicationData, LRUCache
from logsage.auto_resume_policy.error_attribution import get_proposed_solution_cat
from logsage.auto_resume_policy.error_extraction import return_application_errors

from nvidia_resiliency_ext.attribution.base import AttributionState, NVRxAttribution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def chunk_logs_strict(lines):
    """
    Chunks logs strictly between:
    - START: The LAST occurrence of Cycle N
    - END: The LAST occurrence of Cycle N+1

    Lines after the highest Cycle number are ignored.
    If no 'Cycle' markers are found, returns all lines as Cycle 0.
    """
    # Regex to match the profiling line
    cycle_pattern = re.compile(r"profiling\.py:.*Cycle:\s*(\d+)")

    # Step 1: Find the LAST index for every cycle number
    last_cycle_indices = {}
    for index, line in enumerate(lines):
        match = cycle_pattern.search(line)
        if match:
            print(line)
            cycle_num = int(match.group(1))
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
    for i in range(len(sorted_cycles) - 1):
        curr_cycle = sorted_cycles[i]
        next_cycle = sorted_cycles[i + 1]  # This is N+1

        start_index = last_cycle_indices[curr_cycle]
        end_index = last_cycle_indices[next_cycle]

        # Extract lines between LAST Cycle N and LAST Cycle N+1
        raw_chunk = lines[start_index:end_index]

        # Step 3: Remove marker lines
        clean_chunk = [
            line for line in raw_chunk
            if not cycle_pattern.search(line)
        ]

        final_chunks[curr_cycle] = clean_chunk

    return final_chunks


class NVRxLogAnalyzer(NVRxAttribution):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set")
        logger.info(
            f"Using API key: {self.api_key[:10]}..." if self.api_key else "No API key found"
        )
        logger.info(f"Using model: {args.model}")
        self.lru_cache = LRUCache(100_000)
        self.llm = ChatNVIDIA(
            model=args.model,
            api_key=self.api_key,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens,
        )
        self.exclude_nvrx_logs = args.exclude_nvrx_logs
        super().__init__(
            preprocess_input=self.analyze_logs,
            attribution=self.llm_analyze,
            output_handler=self.print_output,
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
        path = self.args.log_path
        with open(path, 'r') as f:
            input_data = f.readlines()

        if self.exclude_nvrx_logs:
            input_data = [line for line in input_data if "nvidia_resiliency_ext" not in line]
            input_data = [line for line in input_data if "[workload:" not in line or 'Cycle:' in line]
            logger.info(f"Excluded {len(input_data)} lines from the input data")
            with open(os.path.join(os.path.dirname(path), "nvrx_logs_edited.txt"), 'w') as f:
                f.writelines(input_data)
        chunks = chunk_logs_strict(input_data)
        output_list = [return_application_errors(self.llm, lines, self.lru_cache) for cycle, lines in chunks.items()]
        return output_list

    async def llm_analyze(self, output_list: list[ApplicationData]) -> list[str]:
        return [
            get_proposed_solution_cat(self.llm, output) if len(output.application_errors_list_full) > 0 else "No error found from application logs"
            for output in output_list
        ]

    async def print_output(self, attribution_results: list[str]) -> list[tuple[str, AttributionState]]:
        output_list = []
        for attribution_result in attribution_results:
            if attribution_result != "No error found from application logs":
                # Concatenate all strings in attribution_result if it's a list/tuple
                logger.info(f"attribution_result: {attribution_result}")
                attr_state = (
                    AttributionState.STOP if 'STOP' in attribution_result[0] else AttributionState.CONTINUE
                )
                if isinstance(attribution_result, (list, tuple)):
                    concatenated_result = '\n'.join(str(item) for item in attribution_result)
                else:
                    concatenated_result = str(attribution_result)
                output_list.append((concatenated_result, attr_state))
            else:
                output_list.append(("No error found from application logs", AttributionState.CONTINUE))
        return output_list


def main():
    parser = argparse.ArgumentParser(
        description='Analyze collective operations across JSON dump files.'
    )
    parser.add_argument('--log-path', type=str, help='Path to log files')
    parser.add_argument(
        '-m',
        '--model',
        default="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
        help='Model to use for LLM analysis',
    )
    parser.add_argument('-t', '--temperature', type=float, default=0.2, help='Temperature for LLM')
    parser.add_argument('-p', '--top_p', type=float, default=0.7, help='Top P for LLM')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Max tokens for LLM')
    parser.add_argument(
        '--exclude_nvrx_logs', action='store_true', help='Exclude nvrx logs from the input data'
    )

    args = parser.parse_args()

    analyzer = NVRxLogAnalyzer(args)
    analyzer.run_sync(args)


if __name__ == "__main__":
    main()
