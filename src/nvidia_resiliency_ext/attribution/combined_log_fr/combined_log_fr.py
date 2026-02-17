import argparse
import logging
from typing import Any

from nvidia_resiliency_ext.attribution.base import AttributionState, NVRxAttribution
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer

logger = logging.getLogger(__name__)


class CombinedLogFR(NVRxAttribution):
    def __init__(self, args: argparse.Namespace):
        super().__init__(
            preprocess_input=self.preprocess_input,
            attribution=self.collective_analysis,
            output_handler=self.print_output,
        )
        self.llm = None
        self.threshold = args.threshold
        self.args = args

    async def preprocess_input(self) -> dict:
        input_data = self.args.input_data
        logger.info(f"input_data: {self.args.input_data}")
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
        template = """
        You are a helpful assistant that analyzes the application logs and collective operations analysis.
        You are given the application logs and collective operations analysis.
        {log_result} includes attribution results based on application logs.
        Its attribution can be highly false positive. Use it to decide whether to restart the application.
        Even if the log results has some suggestion on ranks to be excluded, you should not use it.

        {fr_result} includes health check results per rank and collective analysis, which is more reliable.
        Use the hanging ranks it provides to isolate the ranks that are hanging.
        Even if the fr result has many ranks to be excluded, you can use them as they are to propose a solution.

        You need to analyze the application logs and collective operations analysis and return the proposed solution.
        Summary of the log result: <application log summary> 
        Summary of the fr result: <collective operations analysis summary> 

        The proposed solution should be in the following format: (one line only, if you have extra information, you can add it in the proposed solution with ranks)
        - List of ranks to be excluded: <identified ranks to be excluded, you can use comma to separate multiple ranks without space>
        - Proposed Solution with Ranks: <proposed solution with ranks>
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        if self.llm is None:
            from nvidia_resiliency_ext.attribution.utils import load_nvidia_api_key

            self.llm = ChatNVIDIA(
                model=self.args.model,
                api_key=load_nvidia_api_key(),
                temperature=0.2,
                top_p=0.7,
                max_tokens=16384,
            )
        prompt = PromptTemplate(template=template, input_variables=["log_result", "fr_result"])
        default_values = {"log_result": log_result, "fr_result": fr_result}
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke(input=default_values)
        return result

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
    parser.add_argument('--log-path', type=str, help='Path to log files')

    parser.add_argument(
        '--fr-path',
        type=str,
        nargs='+',
        required=True,
        help='Path to collective operations (FR) log files',
    )
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
    log_analyzer = NVRxLogAnalyzer(args)
    # Now run() is properly synchronized and returns the actual result
    log_result = log_analyzer.run(args.log_path)
    args.llm_analyze = False
    fr_result = CollectiveAnalyzer(args).run(args.fr_path)
    combined_result = CombinedLogFR(args).run_sync([log_result, fr_result])


if __name__ == "__main__":
    main()
