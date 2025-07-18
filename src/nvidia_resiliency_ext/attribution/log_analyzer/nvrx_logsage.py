import argparse
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from nvidia_resiliency_ext.attribution.base import NVRxAttribution, AttributionState
from nvidia_resiliency_ext.attribution.log_analyzer.logsage.error_extraction import return_application_errors
from nvidia_resiliency_ext.attribution.log_analyzer.logsage.error_attribution import get_proposed_solution_cat
from nvidia_resiliency_ext.attribution.log_analyzer.logsage.attribution_classes import LRUCache, ApplicationData
from nvidia_resiliency_ext.attribution.log_analyzer.logsage.consts import llm_model

class NVRxLogAnalyzer(NVRxAttribution):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.lru_cache = LRUCache(100_000)
        self.llm = ChatNVIDIA(
            model=args.model,
            api_key=self.api_key,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens
        )
        super().__init__(preprocess_input=self.analyze_logs, attribution=self.llm_analyze, output_handler=self.print_output)

    async def analyze_logs(self, path: str):
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
        with open(path, 'r') as f:
            input_data = f.readlines()

        output = return_application_errors(self.llm, input_data, self.lru_cache)
        return output

    async def llm_analyze(self, output: ApplicationData):
        return get_proposed_solution_cat(self.llm,  output)

    async def print_output(self, attribution_result: str):
        print(attribution_result)
        return attribution_result, AttributionState.CONTINUE


def main():
    parser = argparse.ArgumentParser(
        description='Analyze collective operations across JSON dump files.'
    )
    parser.add_argument(
        'paths', type=str, help='Path to log files'
    )
    parser.add_argument(
        '-m', '--model',
        default="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
        help='Model to use for LLM analysis',
    )
    parser.add_argument(
        '-t', '--temperature', type=float, default=0.2, help='Temperature for LLM'
    )
    parser.add_argument(
        '-p', '--top_p', type=float, default=0.7, help='Top P for LLM'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=8192, help='Max tokens for LLM'
    )

    args = parser.parse_args()

    analyzer = NVRxLogAnalyzer(args)
    analyzer.run_sync(args.paths)

if __name__ == "__main__":
    main()
