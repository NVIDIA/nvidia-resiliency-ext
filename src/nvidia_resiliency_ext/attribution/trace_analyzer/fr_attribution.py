import argparse
import glob
import json
import os
import pickle
import sys
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from nvidia_resiliency_ext.attribution.base import NVRxAttribution
from nvidia_resiliency_ext.attribution.utils import capture_stdout



@dataclass
class Collective:
    file_id: str
    collective_seq_id: int
    pg_id: int
    op_id: int
    profiling_name: str
    time_created_ns: int
    time_discovered_started_ns: int
    time_discovered_completed_ns: int
    process_group: List[str]
    input_sizes: List[List[int]]
    output_sizes: List[List[int]]
    input_dtypes: List[str]
    output_dtypes: List[str]


class CollectiveAnalyzer(NVRxAttribution):
    def __init__(self, args: argparse.Namespace):
        self.collectives_by_file: Dict[str, List[Collective]] = {}
        self.collective_groups: Dict[Tuple[str, str], List[Collective]] = defaultdict(list)
        self.pg_configs: Dict[str, Dict[int, int]] = {}
        self.node_health_status: Dict[int, Dict[str, Dict[str, str]]] = {}
        self.args = args
        super().__init__(preprocess_input=self.analyze_collectives, attribution=self.llm_analyze, output_handler=self.print_output, attribution_kwargs={"model": args.model, "scheduling_order": args.scheduling_order, "verbose": args.verbose})

    async def print_output(self, attribution_result: str):
        print(attribution_result)
        hanging_ranks = re.search(r'.*hanging ranks: (.*)', attribution_result)
        if hanging_ranks:
            # Parse the hanging ranks from the analysis output
            hanging_ranks_list = list(map(int, hanging_ranks.group(1).split(',')))
            return hanging_ranks_list
        return None

    async def analyze_collectives(self, input_data: List[str]):
        """
        Analyzes the collective operations across multiple JSON files.

        This method performs the following steps:
        - Processes all input paths to collect collective data
        - Prints the process group configurations
        - Analyzes the collective operations
        - Prints the analysis output
        - Uses LLM to analyze the output if requested
        """
        file_paths = input_data
        print(file_paths)
        processed_files = 0
        # Process all input paths
        for path in file_paths:
            if os.path.isdir(path):
                json_files = glob.glob(os.path.join(path, self.args.pattern))
            else:
                json_files = glob.glob(path)

            json_files.sort()

            for filepath in json_files:
                if self.args.verbose:
                    print(f"Processing {filepath}...")
                if self.process_file(filepath):
                    processed_files += 1
            self.print_pg_configs(verbose=self.args.verbose)

        if processed_files == 0:
            print(f"No files at {file_paths} were processed successfully.")
            sys.exit(1)

        print(f"\nSuccessfully processed {processed_files} files.")

        with capture_stdout() as output:
            self.analyze_matches(show_time_spread=self.args.time_spread, verbose=self.args.verbose)
            if self.args.time_spread:
                self.analyze_timing_patterns(verbose=self.args.verbose)
            if self.args.health_check:
                self.print_node_health_status(verbose=self.args.verbose)
        # Print the analysis output
        analysis_output = output.getvalue()
        print(analysis_output)
        attribution_kwargs = {
            "model": self.args.model,
            "scheduling_order": self.args.scheduling_order,
            "verbose": self.args.verbose,
        }
        return analysis_output, attribution_kwargs

    def load_json_file(self, filename: str) -> Dict:
        if filename.lower().endswith('.json'):
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                raise ValueError(f"Error loading JSON file: {filename}")
        else:
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    # Convert pickle data to JSON-compatible format
                    return json.loads(json.dumps(data))
            except (pickle.PickleError, FileNotFoundError, json.JSONDecodeError):
                raise ValueError(f"Error loading pickle file: {filename}")

    def extract_collectives(self, data: Dict, file_id: str) -> List[Collective]:
        collectives = []
        for entry in data['entries']:
            if 'collective_seq_id' in entry:
                collective = Collective(
                    file_id=file_id,
                    collective_seq_id=entry['collective_seq_id'],
                    pg_id=entry['pg_id'],
                    op_id=entry['op_id'],
                    profiling_name=entry['profiling_name'],
                    time_created_ns=entry['time_created_ns'],
                    time_discovered_started_ns=entry.get('time_discovered_started_ns', entry['time_created_ns']),
                    time_discovered_completed_ns=entry.get('time_discovered_completed_ns', entry['time_created_ns']),
                    process_group=entry['process_group'],
                    input_sizes=entry['input_sizes'],
                    output_sizes=entry['output_sizes'],
                    input_dtypes=entry['input_dtypes'],
                    output_dtypes=entry['output_dtypes'],
                )
                collectives.append(collective)
        return collectives

    def process_file(self, filepath: str):
        try:
            file_id = Path(filepath).stem
            # Extract rank ID from the filename assuming it's the last part after an underscore
            rank_id = file_id.split('_')[-1]
            data = self.load_json_file(filepath)

            # Extract pg_config from the JSON data
            if 'pg_config' in data:
                for group, mapping in data['pg_config'].items():
                    def is_int(s):
                        try:
                            int(s)
                            return True
                        except ValueError:
                            return False

                    ranks = [i for i in mapping['ranks'].strip('[]').split(',') if is_int(i)]
                    if 'ranks' in mapping and len(ranks) > 0:
                        mapping['ranks'] = set(map(int,ranks))
                    else:
                        mapping['ranks'] = set()
                    if group not in self.pg_configs:
                        self.pg_configs[group] = mapping
                    else:
                        self.pg_configs[group]['ranks'] = self.pg_configs[group]['ranks'] | mapping['ranks']
            collectives = self.extract_collectives(data, rank_id)
            self.collectives_by_file[rank_id] = collectives

            if 'health_check_results' in data:
                self.node_health_status[rank_id] = data['health_check_results']

            # Group collectives by process_group[0] and process_group[1] only (remove seq_id and op_id)
            for collective in collectives:
                # Use process_group[0] and process_group[1] as they contain the group identifiers
                key = (collective.process_group[0], collective.process_group[1])
                self.collective_groups[key].append(collective)
            return True
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            return False

    def analyze_matches(self, show_time_spread: bool = False, verbose: bool = False):
        """
        Analyze matching collectives across files, grouped by process group type and ordered by sub group.
        Dynamically identifies group types from the data.

        Args:
            show_time_spread (bool): Whether to include time spread analysis in the output
        """
        print("\n=== Collective Operations Analysis ===\n")

        if verbose:
            print("Files processed:")
            for rank_id in sorted(self.collectives_by_file.keys()):
                count = len(self.collectives_by_file[rank_id])
                print(f"  {rank_id}: {count} collectives")
        print()

        # Extract unique sub-group types from the data
        group_types = set()
        for key in self.collective_groups.keys():
            if (
                len(self.collective_groups[key]) > 1
            ):  # Only consider groups with multiple collectives
                _, sub_group = key
                if sub_group:  # Ensure sub_group is not empty
                    group_types.add(sub_group)

        # Convert to sorted list
        group_types = sorted(group_types)

        # If no group types were found, use default ones
        if not group_types:
            group_types = ["TENSOR_MODEL", "PIPELINE_MODEL", "DATA_PARALLEL"]
            print("No sub-group types found in data. Using default group types.")
        else:
            print(f"Found group types: {', '.join(group_types)}")

        # Categorize collective groups by type
        categorized_groups = {group_type: [] for group_type in group_types}
        other_groups = []  # For groups that don't match any of the specified types

        for key, collectives in self.collective_groups.items():
            if len(collectives) <= 1:
                continue

            process_group, sub_group = key

            # Check if this sub_group matches any of our group types
            if sub_group in group_types:
                categorized_groups[sub_group].append((key, collectives))
            else:
                other_groups.append((key, collectives))

        # Process each category in the order of group_types
        for group_type in group_types:
            if categorized_groups[group_type]:
                print(f"\n=== {group_type} Collectives ===\n")

                # Headers for this section
                headers = [
                    ("Process Group", 15),
                    ("PG Desc", 30),
                    ("Op Type", 20),
                    ("Size", 15),
                    ("Dtype", 10),
                    ("Total NRanks", 20),
                    ("Identified Ranks", 40),
                    ("Missing Ranks", 40),
                ]

                if show_time_spread:
                    headers.append(("Time Spread (ns)", 20))

                header_line = " ".join(f"{name:>{width}}" for name, width in headers)
                print(header_line)
                print("-" * len(header_line))

                # Sort groups within this category by process_group
                sorted_groups = sorted(categorized_groups[group_type], key=lambda x: x[0][0])

                for key, collectives in sorted_groups:
                    process_group, sub_group = key

                    # Count occurrences of each rank ID
                    rank_counts = Counter(c.file_id for c in collectives)
                    total_unique_ranks = len(rank_counts)

                    # Get a list of unique ranks
                    unique_ranks = sorted(map(int, rank_counts.keys()))

                    # Find the most common operation type
                    op_counts = Counter(c.profiling_name for c in collectives)
                    op_type = op_counts.most_common(1)[0][0] if op_counts else "Unknown"

                    global_ranks = list(self.pg_configs[process_group]['ranks'])
                    missing_ranks = set(global_ranks) - set(unique_ranks)
                    process_group_str = process_group

                    # Get size and dtype from the first collective
                    size_str = (
                        'x'.join(str(x) for x in collectives[0].input_sizes[0])
                        if collectives[0].input_sizes
                        else "N/A"
                    )
                    dtype = collectives[0].input_dtypes[0] if collectives[0].input_dtypes else "N/A"

                    row_data = [
                        (process_group_str, 15, ''),
                        (sub_group, 30, ''),
                        (op_type, 20, ''),
                        (size_str, 15, ''),
                        (dtype, 10, ''),
                        (total_unique_ranks, 10, 'd'),
                        (','.join(map(str, unique_ranks)), 40, ''),
                        (','.join(map(str, sorted(missing_ranks))), 40, ''),
                    ]

                    if show_time_spread:
                        times = [c.time_created_ns for c in collectives]
                        time_spread = max(times) - min(times)
                        row_data.append((time_spread, 20, 'd'))

                    row = " ".join(f"{val:>{width}{fmt}}" for val, width, fmt in row_data)
                    print(row)

                    # Print detailed rank count distribution
                    if verbose:
                        print(f"  Rank count distribution for {process_group_str}:")
                        for rank, count in sorted(rank_counts.items()):
                            print(f"    Rank {rank}: {count} occurrences")

                            # Print operation type distribution with paired send/recv analysis
                    print(f"  Operation type distribution:")

                    # Calculate totals for each base operation type
                    op_totals = {}
                    for op, count in op_counts.items():
                        base_op = op.split()[0]
                        if base_op not in op_totals:
                            op_totals[base_op] = 0
                        op_totals[base_op] += count

                    # Print summary of operation types
                    for base_op, total in sorted(op_totals.items(), key=lambda x: (-x[1], x[0])):
                        print(f"    {base_op}: {total} total")

                    # Pair corresponding send/recv operations
                    send_ops = {}
                    recv_ops = {}
                    other_ops = {}

                    for op, count in op_counts.items():
                        if "nccl:send" in op:
                            parts = op.split()
                            if len(parts) > 1:
                                direction = parts[1]  # e.g., "0->1"
                                src, dst = direction.split("->")
                                send_ops[(src, dst)] = count
                        elif "nccl:recv" in op:
                            parts = op.split()
                            if len(parts) > 1:
                                direction = parts[1]  # e.g., "0<-1"
                                dst, src = direction.split("<-")
                                recv_ops[(dst, src)] = count
                        else:
                            other_ops[op] = count

                    # Print paired send/recv operations
                    print(f"    Send/Receive pairs (src->dst):")

                    # Combine all unique src-dst pairs
                    all_pairs = set(send_ops.keys()) | set(
                        (dst, src) for dst, src in recv_ops.keys()
                    )

                    # Print each pair with send and recv counts
                    for src, dst in sorted(all_pairs):
                        send_count = send_ops.get((src, dst), 0)
                        recv_count = recv_ops.get((dst, src), 0)

                        # Highlight imbalances
                        if send_count != recv_count:
                            imbalance = f" [IMBALANCE: {send_count-recv_count:+d}]"
                        else:
                            imbalance = ""

                        print(
                            f"      {global_ranks[int(src)]}->{global_ranks[int(dst)]}: {send_count} sends, {recv_count} recvs{imbalance}"
                        )

                    # Print other operations
                    if other_ops:
                        print(f"    Other operations:")
                        for op, count in sorted(other_ops.items(), key=lambda x: (-x[1], x[0])):
                            print(f"      {op}: {count}")

                    print()  # Add an empty line for better readability

    def analyze_timing_patterns(self, verbose: bool = False):
        print("\n=== Timing Analysis ===\n")

        for key, collectives in sorted(self.collective_groups.items()):
            if len(collectives) > 1:
                process_group, sub_group = key
                print(f"\nCollective (Group {process_group}):")

                # Sort by start time
                sorted_collectives = sorted(collectives, key=lambda x: x.time_discovered_started_ns)
                base_time = sorted_collectives[0].time_discovered_started_ns

                # Print header
                print(f"{'Rank':<10} {'Start Time':<15} {'Elapsed Time':<15} {'Operation'}")
                print("-" * 60)

                for c in sorted_collectives:
                    start_time = c.time_discovered_started_ns - base_time
                    elapsed_time = c.time_discovered_completed_ns - c.time_discovered_started_ns
                    print(f"{c.file_id:10} +{start_time:12d} ns {elapsed_time:12d} ns ({c.profiling_name})")

    def print_node_health_status(self, verbose: bool = False):
        print("\n=== Node Health Status ===\n")
        for rank_id, status in self.node_health_status.items():
            # TODO: Add host name to the output
            status_parts = [f"Rank {rank_id}"]
            for device, result in status.items():
                status_parts.append(f"{device}: {result['status']}")
                if verbose:
                    status_parts.append(f"({result['output'].strip()})")
            print(", ".join(status_parts))

    def print_pg_configs(self, verbose: bool = False):
        """Print process group configurations in a more readable format."""
        print("\n=== Process Group Configurations ===\n")

        # Table header
        print(f"{'Group ID':<10} {'Description':<35} {'Ranks':<50}")
        print("-" * 95)
        # Sort by group ID numerically
        for group_id in sorted(self.pg_configs.keys(), key=lambda x: int(x)):
            group = self.pg_configs[group_id]
            ranks = str(group['ranks'])
            print(f"{group_id:<10} {group['desc']:<35} {ranks:<50}")

    async def llm_analyze(self, analysis_output: str, **kwargs):
        """
        Analyze the collective operations using a Large Language Model (LLM).

        This function sends the analysis output to an LLM service (NVIDIA AI Endpoints)
        to identify potential hanging ranks or process groups in distributed training.

        Args:
            analysis_output (str): The output from the collective operations analysis
            verbose (bool): Whether to include more detailed analysis in the prompt
            model (str): The LLM model to use for analysis (e.g., "meta/llama-3.3-70b-instruct")
            scheduling_order (str): The scheduling order of parallelism strategies (default: "TP->PP->DP")

        Returns:
            None: Results are printed to standard output

        Note:
            Requires the NVIDIA_API_KEY environment variable to be set
        """
        model = kwargs["model"]
        scheduling_order = kwargs["scheduling_order"]
        verbose = kwargs["verbose"]

        # Example of using LangChain with the analysis output
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            # Define a template for analyzing collective operation issues
            basic_template = """
            You are an expert in distributed training systems.
            Below is an analysis of collective operations from a distributed training run:
            {analysis_output}

            Please, find the potential hanging rank or process group without any verbosity with the consideration
            on the process configurations, the scheduling order of {scheduling_order}, and the missing ranks.
            """

            heuristic = """
            The imbalance in pipeline parallelism is not a problem if all corresponding process groups have the similar imbalance.
            There can be multiple hanging ranks on multiple process groups.
            """

            simple_answer = """
            We don't need any verbosity, please return the hanging ranks or process groups in the form of f"hanging ranks: <identified ranks>".
            """
            if verbose:
                template = f"{basic_template}\n{heuristic}"
            else:
                template = f"{basic_template}\n{heuristic}\n{simple_answer}"

            prompt = PromptTemplate(
                template=template, input_variables=["analysis_output", "scheduling_order"]
            )

            # Check for API key in environment variables
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                print("NVIDIA_API_KEY environment variable not set. Cannot use AI analysis.")
                return

            default_values = {"analysis_output": analysis_output, "scheduling_order": scheduling_order}

            llm = ChatNVIDIA(
                model=model,
                api_key=api_key,
                temperature=0.2,
                top_p=0.7,
                max_tokens=16384,
            )
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke(input=default_values)
            return result
        except ImportError:
            print("LangChain is not installed. Please install it with:")
            print("pip install langchain langchain-nvidia-ai-endpoints")
        except Exception as e:
            print(f"\nError using LangChain: {e}")
            print("Make sure you have set the NVIDIA_API_KEY environment variable.")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze collective operations across JSON dump files.'
    )
    parser.add_argument(
        'paths', nargs='+', help='Path to JSON files or directories containing JSON files'
    )
    parser.add_argument(
        '-t', '--time-spread', action='store_true', help='Show time spread analysis'
    )
    parser.add_argument(
        '-p', '--pattern', default="*.json", help='File pattern to match (default: *.json)'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument(
        '-c', '--health-check', action='store_true', help='Show node health check results'
    )
    parser.add_argument(
        '-l', '--llm-analyze', action='store_true', help='Use LLM to analyze the output'
    )
    parser.add_argument(
        '-s', '--scheduling-order', default="TP->PP->DP", help='Scheduling order of TP->PP->DP'
    )
    parser.add_argument(
        '-m',
        '--model',
        default="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
        help='Model to use for LLM analysis',
    )
    args = parser.parse_args()

    analyzer = CollectiveAnalyzer(args)
    analyzer.run_sync(args.paths)

if __name__ == "__main__":
    main()
