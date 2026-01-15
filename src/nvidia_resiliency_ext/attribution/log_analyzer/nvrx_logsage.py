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

def lines_after(lines, needle):
    for i, line in enumerate(lines):
        if needle in line:
            return lines[i + 1 :]
    return lines


def chunk_logs_strict(lines):
    """Chunks logs strictly between:
    - START: The LAST occurrence of Cycle N
    - END: The LAST occurrence of Cycle N+1 OR End of File (for the last cycle).

    If no 'Cycle' markers are found, returns all lines as Cycle 0.
    """
    # Regex to match the profiling line
    cycle_pattern = re.compile(r"profiling\.py:.*Cycle:\s*(\d+)")

    # Step 1: Find the LAST index for every cycle number
    last_cycle_indices = {}
    for index, line in enumerate(lines):
        match = cycle_pattern.search(line)
        if match:
            cycle_num = int(match.group(1))
            last_cycle_indices[cycle_num] = index

    # Sort cycles (0, 1, 2...)
    sorted_cycles = sorted(last_cycle_indices.keys())

    final_chunks = {}

    # If no cycles were found, return all lines as Cycle 0
    if not sorted_cycles:
        final_chunks[0] = lines
        return final_chunks

    # Step 2: Iterate through cycles to capture chunks
    # We iterate through ALL sorted cycles now (not len - 1)
    for i in range(len(sorted_cycles)):
        curr_cycle = sorted_cycles[i]
        start_index = last_cycle_indices[curr_cycle]

        # Determine the End Index
        if i < len(sorted_cycles) - 1:
            # If there is a next cycle, stop there
            next_cycle = sorted_cycles[i + 1]
            end_index = last_cycle_indices[next_cycle]
            raw_chunk = lines[start_index:end_index]
        else:
            # --- FIX: Handling the Last Cycle ---
            # If this is the last cycle in the list, go to the end of the lines
            raw_chunk = lines[start_index:]

        # Step 3: Remove marker lines using the existing logic
        clean_chunk = [line for line in raw_chunk if not cycle_pattern.search(line)]

        # Apply the external 'lines_after' filter
        # (Assuming lines_after is defined in your scope)
        clean_chunk = lines_after(clean_chunk, "FT: initialized")

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
        self.exclude_nvrx_logs = getattr(args, 'exclude_nvrx_logs', False)
        self.is_per_cycle = getattr(args, 'is_per_cycle', False)
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
        with open(path, 'r', encoding="latin-1") as f:
            input_data = f.readlines()

        # If is_per_cycle is set, skip filtering and chunking (data is already single-cycle)
        if self.is_per_cycle:
            logger.info("is_per_cycle=True: skipping nvrx log filtering and cycle chunking")
            chunks = {0: input_data}
        else:
            if self.exclude_nvrx_logs:
                input_data = [line for line in input_data if "nvidia_resiliency_ext" not in line]
                input_data = [
                    line for line in input_data if "[workload:" not in line or 'Cycle:' in line
                ]
                logger.info(f"Excluded {len(input_data)} lines from the input data")
                with open(os.path.join(os.path.dirname(path), "nvrx_logs_edited.txt"), 'w') as f:
                    f.writelines(input_data)
            chunks = chunk_logs_strict(input_data)  # Splitting the app log to cycles

        output_list = [
            return_application_errors(self.llm, lines, self.lru_cache)
            for cycle, lines in chunks.items()
        ]
        return output_list

    async def llm_analyze(self, output_list: list[ApplicationData]) -> list[str]:

        result = []
        logger.info("output_list_size: %s", str(len(output_list)))
        for output in output_list:
            if len(output.application_errors_list_full):
                result.append(get_proposed_solution_cat(self.llm, output))
            else:
                if output.finished == "LLM_FAILURE":
                    result.append(("LLM FAILURE","LLM FAILURE","LLM FAILURE","LLM FAILURE",""))
                elif output.finished != "SLURM_CANCELLED":
                    result.append(("ERRORS NOT FOUND","ERRORS NOT FOUND","ERRORS NOT FOUND","ERRORS NOT FOUND",""))
                else:
                    result.append(("RESTART IMMEDIATE","","""Attribution: Primary issues: ["SLURM STEP CANCELLED"], Secondary issues: []""","",""))

        return result

    async def print_output(
        self, attribution_results: list[str]
    ) -> list[tuple[str, AttributionState]]:
        output_list = []
        for attribution_result in attribution_results:
            if attribution_result:
                # Concatenate all strings in attribution_result if it's a list/tuple
                logger.info(f"attribution_result: {attribution_result}")
                attr_state = (
                    AttributionState.STOP
                    if 'STOP' in attribution_result[0]
                    else AttributionState.CONTINUE
                )
                if isinstance(attribution_result, (list, tuple)):
                    concatenated_result = '\n'.join(str(item) for item in attribution_result)
                else:
                    concatenated_result = str(attribution_result)
                output_list.append((concatenated_result, attr_state))
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
    parser.add_argument(
        '--is_per_cycle',
        action='store_true',
        help='Input is already per-cycle data (skip filtering and chunking)',
    )

    args = parser.parse_args()

    analyzer = NVRxLogAnalyzer(args)
    analyzer.run_sync(args)


if __name__ == "__main__":
    main()