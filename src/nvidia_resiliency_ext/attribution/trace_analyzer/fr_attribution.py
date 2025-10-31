import argparse
import glob
import json
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple 

from nvidia_resiliency_ext.attribution.base import AttributionState, NVRxAttribution
from nvidia_resiliency_ext.attribution.utils import capture_logs

log_level = logging.DEBUG if os.getenv('FR_DEBUG') else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger()


# Helper to print to stderr instead of stdout (for MCP compatibility)
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@dataclass
class Collective:
    """
    A class that represents a collective operation.
    Each field corresponds to fields in the FR dump file
    """

    record_id: int
    file_id: str
    collective_seq_id: int
    p2p_seq_id: int
    pg_id: int
    op_id: int
    profiling_name: str
    state: str
    time_created_ns: int
    time_discovered_started_ns: int
    time_discovered_completed_ns: int
    process_group: List[str]
    input_sizes: List[List[int]]
    output_sizes: List[List[int]]
    input_dtypes: List[str]
    output_dtypes: List[str]


class CollectiveAnalyzer(NVRxAttribution):
    """
    This attribution module analyzes the PyTorch Flight Recorder (FR) traces dumped at timeout or exceptions.
    This does the following:

    1. Analyzes the PyTorch Flight Recorder (FR) traces dumped at timeout or exceptions.
    This analysis matches the collectives across ranks per process group and shows,
    which ranks are identified or missing.
    2. If application framework provides the descriptions of process groups,
    and provides global ordering of those process groups in the trace,
    we use that ordering to find the root cause of the interruption (hang or exception)
    - e.g.) Reduction should happen later than model parallel collectives.
            Default_pg should happen after all the collectives.
    3. Returns the missing ranks of the wavefront process group of the chained process groups in the step 2.

    This rank list means the ranks that need to be isolated to fix the interruption.
    The actual root cause of the interruption will be fulfilled by combining other attribution modules
    on the ranks in the list.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the CollectiveAnalyzer class.

        Args:
            args: argparse.Namespace object containing the command line arguments

        """
        # the data structures to store the collective operations
        self.collectives_by_file: Dict[str, List[Collective]] = {}
        # the data structure to store the process group status per rank
        self.pg_status: Dict[str, Dict[str, Dict[str, int]]] = {}
        # the data structure to store the collective operations grouped by process group
        self.collective_groups: Dict[Tuple[str, str], List[Collective]] = defaultdict(list)
        # the data structure to store the process group configurations
        self.pg_configs: Dict[str, Dict[int, int]] = {}
        # the data structure to store the node health status per rank
        self.node_health_status: Dict[int, Dict[str, Dict[str, str]]] = {}
        self.args = args
        eprint(f"args: {args}")
        self.llm = None
        self.type_to_order = None
        # initialize the NVRxAttribution class to run the attribution pipeline
        super().__init__(
            preprocess_input=self.preprocess_FR_dumps,
            attribution=self.collective_analysis,
            output_handler=self.print_output,
            attribution_kwargs={
                "model": args.model,
                "verbose": args.verbose,
            },
        )

    """
    Routines registered for the attribution pipeline
    """

    # output handler to print the attribution results
    async def print_output(self, attribution_result: str):
        hanging_ranks_list = []
        if self.llm and self.args.llm_analyze:
            logger.info(attribution_result)
            hanging_ranks = re.search(r'.*hanging ranks: \{([^}]*)\}', attribution_result)
            if hanging_ranks is not None:
                # Parse the hanging ranks from the analysis output
                hanging_ranks_str = hanging_ranks.group(1).strip()
                hanging_ranks_list = list(map(int, hanging_ranks_str.split(',')))
        else:
            for idx, line in enumerate(attribution_result.split('\n')):
                line_list = line.split('|')
                if len(line_list) >= 5:
                    logger.info(line)
                    if idx >= 1:
                        hanging_ranks_list.append(line_list[5])
        return f"hanging ranks: {hanging_ranks_list}", AttributionState.CONTINUE

    # preprocess input to analyze the collective operations
    async def preprocess_FR_dumps(self) -> str:
        """
        Analyzes the collective operations across multiple JSON files.

        This method performs the following steps:
        - Processes all input paths to collect collective data
        - Prints the process group configurations
        - Analyzes the collective operations
        - Prints the analysis output
        - Uses LLM to analyze the output if requested
        """
        logger.info(f"FR args: {self.args}")
        file_paths = [self.args.fr_path]
        logger.info(f"file_paths: {file_paths}, pattern: {self.args.pattern}")
        processed_files = 0
        # Process all input paths
        # read files from file_paths and prepare data structure for collective analysis
        for path in file_paths:
            logger.info(f"path: {path}")
            json_files = (
                glob.glob(os.path.join(path, self.args.pattern))
                if os.path.isdir(path)
                else glob.glob(path)
            )
            logger.info(f"json_files: {json_files}")
            json_files.sort()
            for filepath in json_files:
                if self.args.verbose:
                    logger.info(f"Processing {filepath}...")
                if self.process_file(filepath):
                    processed_files += 1
            self.collective_groups = self.group_collectives_by_windows()

            def collectives_to_order():
                """
                Collectives to order.
                """
                collectives_to_order = {}
                idx = 0
                for key, collectives in self.collective_groups.items():
                    collectives_to_order[key] = idx
                    idx += 1
                return collectives_to_order

            self.collectives_to_order = collectives_to_order()
            logger.info(f"collective_groups: {self.collective_groups.keys()}")
            logger.info(f"collectives_to_order: {self.collectives_to_order}")
            if self.args.verbose:
                self.print_pg_configs(verbose=self.args.verbose)

        if processed_files == 0:
            raise ValueError(f"No files at {file_paths} were processed successfully.")

        logger.info(f"\nSuccessfully processed {processed_files} files.")
        missing_pg = None
        completed_pg = None
        # analyze collectives to find process groups with missing and completed ranks
        completed_pg, missing_pg = self.analyze_matches(verbose=self.args.verbose)
        grouped_missing_pgs = {}
        grouped_completed_pgs = {}

        # if the dump file contains health check results, parse the health check results
        # and print them in a format
        if self.args.health_check:
            self.print_node_health_status(verbose=self.args.verbose)

        # group the process groups with missing and completed ranks
        # by finding longest paths in the graph
        grouped_missing_pgs = self.group_pgs(missing_pg)
        if len(grouped_missing_pgs) == 0:
            grouped_completed_pgs = self.group_pgs(completed_pg)

        # gather the head node of each group with missing and completed ranks
        # the head node is the first node in the group
        # the missing ranks in the head node of the missing process groups
        # are considered to cause the other nodes in the group to hang
        def gather_head_nodes(grouped_pgs):
            head_nodes = set()
            for group_id, pg_indices in grouped_pgs.items():
                head_nodes.add(pg_indices[0])
            return head_nodes

        head_nodes_missing = None
        head_nodes_completed = None
        # Gather the head node of each group
        if len(grouped_missing_pgs) > 0:
            head_nodes_missing = gather_head_nodes(grouped_missing_pgs)
            logger.debug(f"head_nodes of missing_pg: {head_nodes_missing}")
        else:
            head_nodes_completed = gather_head_nodes(grouped_completed_pgs)
            logger.debug(f"head_nodes of completed_pg: {head_nodes_completed}")
        # Print the analysis output
        with capture_logs() as output:

            def print_ranks_in_pgs(head_nodes, pg_dict, missing_or_completed="Missing"):
                logger.info(
                    f"{'PGID':<6} | {'Process Group Desc':<25} | {'Op Type':<10} | {'Size':<8} \
                        | {'Dtype':<8} | {missing_or_completed} Ranks"
                )
                for pg_idx in head_nodes:
                    entry = list(pg_dict[pg_idx][0])
                    entry.remove(entry[-2])
                    if missing_or_completed == "Missing":
                        ranks_to_print = entry[6]
                    else:
                        ranks_to_print = entry[5]
                    logger.info(
                        f"{entry[0]:<6} | {entry[1]:<25} | {entry[2]:<10} | {entry[3]:<8} \
                            | {entry[4]:<8} | {ranks_to_print}"
                    )

            if head_nodes_missing:
                logger.debug(f"head_nodes_missing: {head_nodes_missing}")
                print_ranks_in_pgs(head_nodes_missing, missing_pg, "Missing")
            # TODO: using this completed pg needs to be updated with new algorithm for isolation
            if head_nodes_completed:
                print_ranks_in_pgs(head_nodes_completed, completed_pg, "Completed")
        analysis_output = output.getvalue()
        return analysis_output

    async def collective_analysis(self, analysis_output: str) -> str:
        """
        Analyze the collective operations using a Large Language Model (LLM).

        This function sends the analysis output to an LLM service (NVIDIA AI Endpoints) optionally
        to identify potential hanging ranks or process groups in distributed training.

        Args:
            analysis_output (str): The output from the collective operations analysis
            verbose (bool): Whether to include more detailed analysis in the prompt
            model (str): The LLM model to use for analysis (e.g., "meta/llama-3.3-70b-instruct")

        Returns:
            None: Results are printed to standard output

        Note:
            Requires the NVIDIA_API_KEY environment variable to be set
        """
        result = analysis_output
        if self.args.llm_analyze:
            logger.info(f"Using LLM to analyze the output: {analysis_output}")
            model = self.args.model
            verbose = self.args.verbose
            try:
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import PromptTemplate
                from langchain_nvidia_ai_endpoints import ChatNVIDIA

                # Define a template for analyzing collective operation issues
                basic_template = """
                You are an expert in distributed training systems.
                Below is an analysis of collective operations from a distributed training run:
                {analysis_output}

                Please, find the process groups with missing ranks without any verbosity with the consideration
                Use the missing ranks per process group to propose a solution.
                """

                simple_answer = """
                We don't need any verbosity, please return the list of hanging ranks only 
                """

                output_format = """
                The final output format should be: f'hanging ranks: <set of missing ranks>'.
                """
                if verbose:
                    template = f"{basic_template}\n{output_format}"
                else:
                    template = f"{basic_template}\n{output_format}\n{simple_answer}"

                prompt = PromptTemplate(template=template, input_variables=["analysis_output"])

                # Check for API key in environment variables
                api_key = os.getenv("NVIDIA_API_KEY")
                if not api_key:
                    eprint("NVIDIA_API_KEY environment variable not set. Cannot use AI analysis.")
                    return

                default_values = {
                    "analysis_output": analysis_output,
                }
                if self.llm is None:
                    self.llm = ChatNVIDIA(
                        model=model,
                        api_key=api_key,
                        temperature=0.2,
                        top_p=0.7,
                        max_tokens=32768,
                    )
                chain = prompt | self.llm | StrOutputParser()
                result = await chain.ainvoke(input=default_values)
                return result
            except ImportError:
                eprint("LangChain is not installed. Please install it with:")
                eprint("pip install langchain langchain-nvidia-ai-endpoints")
            except Exception as e:
                eprint(f"\nError using LangChain: {e}")
                eprint("Make sure you have set the NVIDIA_API_KEY environment variable.")
        return result

    """
    Helper functions to define steps for the registered attribution steps above
    """

    def group_collectives_by_windows(self):
        """
        Group the collectives by windows.
        """
        # Track the current index for each rank's collective list
        rank_indices = {rank_id: 0 for rank_id in self.collectives_by_file.keys()}

        # Track how many times each process group has been fully processed (window/phase counter)
        pg_window_counter = defaultdict(int)

        # Track which ranks have participated in the current window for each PG
        # Key: (pg, window_idx), Value: set of rank_ids that have processed this PG in this window
        pg_window_participants = defaultdict(set)

        # Track which PGs were active (had at least one rank working on them) in the last iteration
        # This helps us detect when we've completely left a PG and come back to it
        pgs_with_active_ranks_last_iter = set()

        # Result structure: maps (process_group, sub_group, window_index) to list of collectives
        matched_groups = defaultdict(list)

        # Keep processing until all ranks have processed all their collectives
        while any(
            rank_indices[rank_id] < len(self.collectives_by_file[rank_id])
            for rank_id in rank_indices.keys()
        ):

            # Get the current process group type for each rank that hasn't finished
            current_pg_types = {}
            for rank_id, idx in rank_indices.items():
                if idx < len(self.collectives_by_file[rank_id]):
                    collective = self.collectives_by_file[rank_id][idx]
                    # Use (process_group[0], process_group[1]) as the key
                    pg_key = (collective.process_group[0], collective.process_group[1])
                    current_pg_types[rank_id] = pg_key

            if not current_pg_types:
                break

            # Find the most common process group type among active ranks
            # This represents the "wavefront" - the PG type most ranks are working on
            pg_counter = Counter(current_pg_types.values())
            current_pg, _ = pg_counter.most_common(1)[0]

            # Determine current window for this PG
            window_idx = pg_window_counter[current_pg]
            pg_window_key = (current_pg, window_idx)

            # Check if we should create a new window for this PG
            # This happens when a significantly different set of ranks arrives
            ranks_with_current_pg = set(
                rid for rid, pg in current_pg_types.items() if pg == current_pg
            )
            already_participated = pg_window_participants[pg_window_key] & ranks_with_current_pg
            previous_participants = pg_window_participants[pg_window_key]
            
            has_previous_participants = len(previous_participants) > 0
            has_significant_new_ranks = len(ranks_with_current_pg - previous_participants) >= 2
            
            # Create new window if:
            # 1. Some ranks have already participated (same ranks coming back), OR
            # 2. We have previous participants and mostly/completely new ranks (different batch)
            should_create_new_window = False
            
            if current_pg not in pgs_with_active_ranks_last_iter:
                # PG was inactive - check if we need a new window
                if already_participated or (has_previous_participants and has_significant_new_ranks):
                    should_create_new_window = True
            
            if should_create_new_window:
                # We're starting a new window/phase
                pg_window_counter[current_pg] += 1
                window_idx = pg_window_counter[current_pg]
                pg_window_key = (current_pg, window_idx)

            # Create key with window index to separate different phases
            key_with_window = (current_pg[0], current_pg[1], window_idx)

            # Collect all consecutive collectives for this PG from ranks that are at this PG
            # Keep processing until no more ranks have this PG as their next collective
            has_matches = True
            while has_matches:
                has_matches = False
                for rank_id in list(rank_indices.keys()):
                    idx = rank_indices[rank_id]
                    if idx < len(self.collectives_by_file[rank_id]):
                        collective = self.collectives_by_file[rank_id][idx]
                        pg_key = (collective.process_group[0], collective.process_group[1])

                        # If this rank's next collective matches the current PG, process it
                        if pg_key == current_pg:
                            matched_groups[key_with_window].append(collective)
                            rank_indices[rank_id] += 1
                            has_matches = True
                            # Track that this rank participated in this window
                            pg_window_participants[pg_window_key].add(rank_id)

            # Update the set of PGs that have active ranks for the next iteration
            # After processing, check which PGs still have ranks waiting on them
            pgs_with_active_ranks = set()
            for rank_id, idx in rank_indices.items():
                if idx < len(self.collectives_by_file[rank_id]):
                    collective = self.collectives_by_file[rank_id][idx]
                    pg_key = (collective.process_group[0], collective.process_group[1])
                    pgs_with_active_ranks.add(pg_key)

            pgs_with_active_ranks_last_iter = pgs_with_active_ranks

            # Note: Ranks that have moved to a different PG type "pop up" and wait
            # They'll be processed in the next iteration when we select their PG as the wavefront

        return matched_groups

    def analyze_matches(self, verbose: bool = False):
        """
        Analyze matching collectives across files, grouped by process group type and ordered by sub group.
        Dynamically identifies group types from the data.

        Args:
            verbose (bool): Whether to include more detailed analysis in the output
        """
        logger.info("\n=== Collective Operations Analysis ===\n")

        if verbose:
            logger.info("Files processed:")
            for rank_id in sorted(self.collectives_by_file.keys()):
                count = len(self.collectives_by_file[rank_id])
                logger.info(f"  {rank_id}: {count} collectives")
        logger.info("")

        def match_collectives():
            # Extract unique sub-group types from the data
            group_types = set()
            for key in self.collective_groups.keys():
                if (
                    len(self.collective_groups[key]) > 1
                ):  # Only consider groups with multiple collectives
                    process_group, sub_group, window_idx = key
                    if sub_group:  # Ensure sub_group is not empty
                        group_types.add(sub_group)

            # Convert to sorted list
            group_types = sorted(group_types)

            # If no group types were found, use default ones
            if not group_types:
                group_types = ["TENSOR_MODEL", "PIPELINE_MODEL", "DATA_PARALLEL"]
                logger.info("No sub-group types found in data. Using default group types.")
            else:
                logger.info(f"Found group types: {', '.join(group_types)}")

            # Categorize collective groups by type
            categorized_groups = {group_type: [] for group_type in group_types}
            other_groups = []  # For groups that don't match any of the specified types

            for key, collectives in self.collective_groups.items():
                if len(collectives) <= 1:
                    continue

                process_group, sub_group, window_idx = key

                # Check if this sub_group matches any of our group types
                if sub_group in group_types:
                    categorized_groups[sub_group].append((key, collectives))
                else:
                    other_groups.append((key, collectives))

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

            header_line = " ".join(f"{name:>{width}}" for name, width in headers)
            logger.info(header_line)
            logger.info("-" * len(header_line))

            # Process each category in the order of group_types
            completed_pg = defaultdict(list)
            missing_pg = defaultdict(list)

            def matching_collectives_per_process_group(collective_group):
                logger.debug(f"collective_group: {collective_group}")
                key, collectives = collective_group
                process_group, sub_group, window_idx = key
                # Count occurrences of each rank ID
                group_by_seq_id = defaultdict(list)
                max_completed_collective_seq_id = -1
                max_enqueued_collective_seq_id = -1
                local_pg_map = dict()
                rank_id = None
                for c in collectives:
                    rank_id = c.file_id
                    logger.debug(
                        f"rank_id: {rank_id}, c.pg_id: {c.pg_id}, c.file_id: {c.file_id}, c.collective_seq_id: {c.collective_seq_id}, process_group: {process_group},"
                        f"c.state: {c.state}"
                    )
                    pg_status = self.pg_status[rank_id][str(c.pg_id)]
                    local_pg_map[rank_id] = c.pg_id
                    if pg_status['last_completed_collective'] >= max_completed_collective_seq_id:
                        max_completed_collective_seq_id = pg_status['last_completed_collective']
                    if pg_status['last_enqueued_collective'] >= max_enqueued_collective_seq_id:
                        max_enqueued_collective_seq_id = pg_status['last_enqueued_collective']

                logger.debug(f"max_completed_collective_seq_id: {max_completed_collective_seq_id}")
                logger.debug(f"max_enqueued_collective_seq_id: {max_enqueued_collective_seq_id}")
                local_pg_id = local_pg_map[rank_id]
                # Ranks holding entries earlier than max_completed_collective_seq_id -> ranks failing to complete expected collectives
                rank_counts = defaultdict(list)
                for c in collectives:
                    if c.state != 'scheduled':
                        continue
                    rank_counts['appeared'].append(c.file_id)
#                    if get_correct_seq_id(c) <= max_completed_collective_seq_id:
#                        rank_counts['mismatched'].append(c.file_id)
                appeared_rank_counts = Counter(rank_counts['appeared'])
                # Ranks with less number of enqueued collectives than max_enqueued_collective_seq_id -> host not making expected progress
                for rank_id in self.pg_configs[process_group]['ranks']:
                    rank_id = str(rank_id)
                    if (
                        rank_id not in self.pg_status
                        or str(local_pg_id) not in self.pg_status[rank_id]
                    ):
                        continue
                    if (
                        self.pg_status[rank_id][str(local_pg_id)]['last_enqueued_collective']
                        < max_enqueued_collective_seq_id
                    ):
                        rank_counts['mismatched'].append(rank_id)
                mismatched_rank_counts = Counter(rank_counts['mismatched'])
                logger.debug(f"mismatched_rank_counts: {mismatched_rank_counts}")
                total_unique_ranks = len(appeared_rank_counts)

                # Get a list of unique ranks that appeared in the trace
                unique_ranks = sorted(map(int, appeared_rank_counts.keys()))

                # Find the most common operation type
                op_names = [
                    c.profiling_name
                    for c in collectives
                    if get_correct_seq_id(c) > max_completed_collective_seq_id
                ]
                op_counts = Counter(op_names)
                logger.debug(f"process_group: {process_group}, op_counts: {op_counts}")
                op_type = op_counts.most_common(1)[0][0] if op_counts else "Unknown"

                def pair_send_recv_operations():
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
                    # Combine all unique src-dst pairs
                    all_pairs = set(send_ops.keys()) | set(
                        (dst, src) for dst, src in recv_ops.keys()
                    )
                    return all_pairs, send_ops, recv_ops, other_ops

                all_pairs, send_ops, recv_ops, other_ops = pair_send_recv_operations()
                # expected ranks for this process group
                global_ranks = list(sorted(self.pg_configs[process_group]['ranks']))
                missing_ranks = set()
                if "nccl:send" in op_type or "nccl:recv" in op_type:
                    for src, dst in sorted(all_pairs):
                        send_count = send_ops.get((src, dst), 0)
                        recv_count = recv_ops.get((dst, src), 0)
                        logger.debug(
                            f"src: {src}, dst: {dst}, send_count: {send_count}, recv_count: {recv_count}"
                        )
                        # Only mark as missing if the rank is truly not present in the trace
                        # Check if recv > send AND the sender is not in unique_ranks
                        if recv_count > send_count:
                            missing_global_rank = global_ranks[int(src)]
                            # Only add to missing if this rank is truly not present
                            if missing_global_rank not in unique_ranks:
                                missing_ranks = missing_ranks | set([missing_global_rank])
                        # Similarly for send > recv
                        if send_count > recv_count:
                            missing_global_rank = global_ranks[int(dst)]
                            if missing_global_rank not in unique_ranks:
                                missing_ranks = missing_ranks | set([missing_global_rank])
                else:
                    missing_ranks = set(global_ranks) - set(unique_ranks)
                    missing_ranks = missing_ranks | set(map(int, mismatched_rank_counts.keys()))

                correct_unique_ranks = set(unique_ranks) - missing_ranks
                logger.debug(f"missing_ranks: {missing_ranks}")

                # Get size and dtype from the first collective
                size_str = (
                    'x'.join(str(x) for x in collectives[0].input_sizes[0])
                    if collectives[0].input_sizes
                    else "N/A"
                )
                dtype = collectives[0].input_dtypes[0] if collectives[0].input_dtypes else "N/A"

                row_data = [
                    (process_group, 15, ''),
                    (','.join(map(str, key[1:])), 30, ''),
                    (op_type, 20, ''),
                    (size_str, 15, ''),
                    (dtype, 10, ''),
                    (total_unique_ranks, 10, 'd'),
                    (','.join(map(str, correct_unique_ranks)), 40, ''),
                    (','.join(map(str, sorted(missing_ranks))), 40, ''),
                ]

                row = " ".join(f"{val:>{width}{fmt}}" for val, width, fmt in row_data)
                parsed_row = tuple(row_elem[0] for row_elem in row_data)
                if len(missing_ranks) <= 0:
                    completed_pg[(int)(parsed_row[0])].append(parsed_row)
                    # print(row)
                    return
                else:
                    missing_pg[(int)(parsed_row[0])].append(parsed_row)
                    logger.info(row)

                # Print detailed rank count distribution
                if verbose:
                    logger.info(f"  Rank count distribution for {process_group}:")
                    for rank, count in sorted(appeared_rank_counts.items()):
                        logger.info(f"    Rank {rank}: {count} occurrences")

                # Print operation type distribution with paired send/recv analysis
                logger.info("  Operation type distribution:")
                # Print paired send/recv operations
                logger.info("    Send/Receive pairs (src->dst):")

                # Print each pair with send and recv counts
                for src, dst in all_pairs:
                    send_count = send_ops.get((src, dst), 0)
                    recv_count = recv_ops.get((dst, src), 0)

                    # Highlight imbalances
                    if send_count != recv_count:
                        imbalance = f" [IMBALANCE: {send_count-recv_count:+d}]"
                    else:
                        imbalance = ""

                    logger.info(
                        f"      {global_ranks[int(src)]}->{global_ranks[int(dst)]}: {send_count} sends, {recv_count} recvs{imbalance}"
                    )

                # Print other operations
                if other_ops:
                    logger.info("    Other operations:")
                    for op, count in sorted(other_ops.items(), key=lambda x: (-x[1], x[0])):
                        logger.info(f"      {op}: {count}")

            def get_correct_seq_id(collective):
                if (
                    "nccl:send" in collective.profiling_name
                    or "nccl:recv" in collective.profiling_name
                ):
                    return collective.p2p_seq_id
                else:
                    return collective.collective_seq_id

            for key, collective_group in self.collective_groups.items():
                logger.debug(f"key: {key}, collective_group: {collective_group}")
                matching_collectives_per_process_group((key, collective_group))
            
            # Cross-window matching: if the same PG has missing ranks in different windows,
            # try to match them across windows
            pg_all_windows = defaultdict(list)  # pg_id -> list of (window_idx, identified_ranks, missing_ranks)
            
            for pg_id, entries in missing_pg.items():
                for entry in entries:
                    # entry format: (pg_id, pg_desc, op_type, size, dtype, total_nranks, identified_ranks, missing_ranks)
                    pg_desc = entry[1]  # e.g., "default_pg,0" or "default_pg,1"
                    identified_ranks_str = entry[6]
                    missing_ranks_str = entry[7]
                    
                    identified_ranks = set(map(int, identified_ranks_str.split(','))) if identified_ranks_str else set()
                    missing_ranks = set(map(int, missing_ranks_str.split(','))) if missing_ranks_str else set()
                    
                    window_idx = int(pg_desc.split(',')[-1]) if ',' in pg_desc else 0
                    pg_all_windows[pg_id].append((window_idx, identified_ranks, missing_ranks, entry))
            
            # For each PG with multiple windows, try to match missing ranks across windows
            merged_missing_pg = defaultdict(list)
            for pg_id, windows_data in pg_all_windows.items():
                if len(windows_data) <= 1:
                    # Single window, keep as-is
                    for _, _, _, entry in windows_data:
                        merged_missing_pg[pg_id].append(entry)
                    continue
                
                # Multiple windows for this PG - try to match across windows
                all_identified = set()
                all_missing = set()
                representative_entry = windows_data[0][3]  # Use first window's entry as template
                
                for window_idx, identified, missing, entry in windows_data:
                    all_identified.update(identified)
                    all_missing.update(missing)
                
                # Ranks that are identified in at least one window should not be considered missing
                truly_missing = all_missing - all_identified
                
                if truly_missing:
                    # Create merged entry with truly missing ranks
                    merged_entry = list(representative_entry)
                    merged_entry[6] = ','.join(map(str, sorted(all_identified)))
                    merged_entry[7] = ','.join(map(str, sorted(truly_missing)))
                    merged_entry = tuple(merged_entry)
                    merged_missing_pg[pg_id].append(merged_entry)
                else:
                    # No truly missing ranks after cross-window matching
                    # Don't add to merged_missing_pg (it's complete now)
                    pass
            
            return completed_pg, merged_missing_pg

        completed_pg, missing_pg = match_collectives()
        return completed_pg, missing_pg

    def group_pgs(self, pgs: Dict[str, List[str]]) -> Dict[int, List[int]]:
        """
        Groups process groups by finding longest paths in the graph when their ranks overlap.
        Each process group proceeds to neighbors with equal or higher index of process group type.
        pgs are connected if they share any ranks.

        If there are multiple overlapped paths, the longest path is selected.

        Args:
            pgs: Dictionary where keys are group types and values are lists of process group data

        Returns:
            Dictionary with grouped process groups, where each group contains PGs in the longest path
        """

        grouped_pgs = defaultdict(set)
        pg_rank_mapping = {}
        # Build adjacency graph - PGs are connected if they share any ranks
        graph = defaultdict(set)
        for group_type, pg_list in pgs.items():
            if not pg_list:
                continue

            # Extract rank information from each process group
            # Each pg_data is a list of tuples (row_data), and we need to find the ranks
            pg_data_list = []  # Keep track of original pg_data objects

            for pg_data in pg_list:
                if len(pg_data) > 6:
                    logger.debug(f"pg_data: {pg_data}")
                    ranks_str = pg_data[6].split(',') + pg_data[7].split(
                        ','
                    )  # identified, missing ranks
                    ranks_str = [int(rank) for rank in ranks_str if rank != '']
                    if ranks_str:
                        ranks = set(ranks_str)
                        logger.debug(f"ranks: {ranks}")
                        pg_rank_mapping[(int)(pg_data[0])] = ranks_str  # Use index as key
                        pg_data_list.append(pg_data)

            if not pg_rank_mapping:
                continue
            logger.debug(f"pg_rank_mapping: {pg_rank_mapping}")

        pg_indices = list(map(int, pg_rank_mapping.keys()))
        for i, pg1_idx in enumerate(pg_indices):
            graph[pg1_idx].add(pg1_idx)
            for j, pg2_idx in enumerate(pg_indices):
                if i != j:
                    pg2_ranks = pg_rank_mapping[pg2_idx]
                    # Check if PGs share any ranks
                    if set(pg_rank_mapping[pg1_idx]) & set(
                        pg_rank_mapping[pg2_idx]
                    ):  # Set intersection
                        graph[pg1_idx].add(pg2_idx)
        logger.debug(f"graph: {graph}")
        # Find longest paths in the graph
        visited = set()
        group_id = 0

        def dfs(node, current_path, visited_in_path, visited_keys):
            current_key = pgs[node][0][1]
            logger.debug(f"current_key: {current_key}, visited_keys: {visited_keys}")
            if node in visited_in_path or current_key in visited_keys:
                logger.debug(f"visited_in_path: {visited_in_path}, visited_keys: {visited_keys}")
                logger.debug(f"Cycle detected, returning current path: {current_path}")
                return [current_path]  # Cycle detected, return current path

            visited_in_path.add(node)
            current_path.append(node)
            visited_keys.add(current_key)
            if not graph[node] or all(neighbor in visited_in_path for neighbor in graph[node]):
                # Leaf node or all neighbors visited, return this path
                if current_key in visited_keys:
                    visited_keys.remove(current_key)
                return [current_path.copy()]

            def find_type_val(key: tuple[str, str, int]) -> int:
                """
                Find the order index of a given process group type
                """
                type_name = key[0]
                type_val = key[1].split(',')[0]
                per_pg_seq = (int)(key[1].split(',')[1])
                parsed_key = (type_name, type_val, per_pg_seq)
                logger.debug(
                    f"key: {parsed_key}, self.collectives_to_order: {self.collectives_to_order[parsed_key]}"
                )
                return self.collectives_to_order.get(parsed_key, -1)

            all_paths = []
            for neighbor in graph[node]:
                if neighbor not in visited_in_path:
                    tail_key = pgs[node][0][:2]
                    tail_pg_type = find_type_val(tail_key)
                    new_node_key = pgs[neighbor][0][:2]
                    new_node_pg_type = find_type_val(new_node_key)
                    logger.debug(f"current_path: {current_path}, neighbor: {neighbor}")
                    logger.debug(
                        f"tail_pg_type: {tail_pg_type}, new_node_pg_type: {new_node_pg_type}"
                    )
                    if tail_pg_type <= new_node_pg_type:
                        paths_from_neighbor = dfs(
                            neighbor, current_path.copy(), visited_in_path.copy(), visited_keys
                        )
                        all_paths.extend(paths_from_neighbor)
                    else:
                        all_paths.append(current_path.copy())
            #            visited_in_path.remove(node)
            visited_keys.remove(current_key)
            return all_paths

        def find_valid_paths(graph, start_node, visited):
            """
            Find all longest paths starting from a given node using DFS.
            Returns a list of paths, where each path is a list of nodes.
            """
            return dfs(start_node, [], set(), set())

        sorted_pg_indices = sorted(pg_indices, key=lambda x: len(pg_rank_mapping[x]), reverse=True)
        logger.debug(f"sorted_pg_indices: {sorted_pg_indices}")

        for pg_idx in sorted_pg_indices:
            if pg_idx in visited:
                continue

            # Find all longest paths starting from this PG
            all_paths = find_valid_paths(graph, pg_idx, visited)
            if all_paths is None:
                logger.info(f"No paths from PG {pg_idx}. Skipping this PG")
                continue
            else:
                seen_paths = set()
                for path in all_paths:
                    # Convert path to tuple for hashing and duplicate detection
                    path_tuple = tuple(path)
                    if path_tuple not in seen_paths:
                        seen_paths.add(path_tuple)
                logger.debug(f"all_paths: {all_paths}")
                logger.debug(f"Filtered(excl. dup.) paths starting from {pg_idx}: {seen_paths}")
                for path in seen_paths:
                    for node in path:
                        visited.add(node)
                    grouped_pgs[group_id] = list(path)
                    group_id += 1

        logger.debug(f"grouped_pgs: {grouped_pgs}")
        # Remove paths that are subsets of other paths
        unique_paths = []
        path_tuples = list(grouped_pgs.values())
        logger.debug(f"path_tuples: {path_tuples}")
        for i, path1 in enumerate(path_tuples):
            is_subset = False
            for j, path2 in enumerate(path_tuples):
                if i != j:
                    if set(path1) < (set(path2)):
                        logger.debug(f"path1: {path1} is a subset of path2: {path2}")
                        is_subset = True
                        break
                    elif set(path1) == (set(path2)):
                        if path1 not in unique_paths and path2 not in unique_paths:
                            unique_paths.append(path1)
                        is_subset = True
                        break
            if not is_subset:
                logger.debug(f"path1: {path1}")
                unique_paths.append(path1)
        grouped_pgs = {i: path for i, path in enumerate(unique_paths)}
        logger.debug(f"unique_paths: {unique_paths}")
        return grouped_pgs


    def process_file(self, filepath: str):
        """
        Process a single file to extract collective operations and other metadata
        """

        def load_trace_file(filename: str) -> Dict:
            if filename.lower().endswith('.json'):
                try:
                    with open(filename, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    raise ValueError(f"Error loading JSON file: {filename}")
            else:
                try:
                    with open(filename, 'rb') as f:
                        # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
                        # Severity: Medium   Confidence: High
                        # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
                        # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
                        data = pickle.load(f)  # nosec
                        # Convert pickle data to JSON-compatible format
                        converted_data = json.loads(json.dumps(data))
                    if getattr(self.args, 'debug', False):
                        with open(filename + '.json', 'w') as f:
                            f.write(json.dumps(converted_data, indent=2))
                            f.write('\n')
                    return converted_data
                except (pickle.PickleError, FileNotFoundError, json.JSONDecodeError):
                    raise ValueError(f"Error loading pickle file: {filename}")

        def extract_collectives(data: Dict, file_id: str) -> List[Collective]:
            """
            Extract collective operations from the JSON data
            """
            collectives = []
            for entry in data['entries']:
                if 'collective_seq_id' in entry and entry['state'] == 'scheduled':
                    collective = Collective(
                        record_id=entry.get('record_id', -1),
                        file_id=file_id,
                        collective_seq_id=entry['collective_seq_id'],
                        p2p_seq_id=entry.get('p2p_seq_id', -1),
                        pg_id=entry['pg_id'],
                        op_id=entry['op_id'],
                        profiling_name=entry['profiling_name'],
                        time_created_ns=entry['time_created_ns'],
                        time_discovered_started_ns=entry.get(
                            'time_discovered_started_ns', entry['time_created_ns']
                        ),
                        time_discovered_completed_ns=entry.get(
                            'time_discovered_completed_ns', entry['time_created_ns']
                        ),
                        process_group=entry['process_group'],
                        state=entry['state'],
                        input_sizes=entry['input_sizes'],
                        output_sizes=entry['output_sizes'],
                        input_dtypes=entry['input_dtypes'],
                        output_dtypes=entry['output_dtypes'],
                    )
                    collectives.append(collective)
            return collectives

        try:
            file_id = Path(filepath).stem
            # Extract rank ID from the filename assuming it's the last part after an underscore
            rank_id = file_id.split('_')[-1]
            data = load_trace_file(filepath)

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
                        mapping['ranks'] = set(map(int, ranks))
                    else:
                        mapping['ranks'] = set()
                    if group not in self.pg_configs:
                        self.pg_configs[group] = mapping
                    else:
                        self.pg_configs[group]['ranks'] = (
                            self.pg_configs[group]['ranks'] | mapping['ranks']
                        )
            collectives = extract_collectives(data, rank_id)
            self.pg_status[rank_id] = data['pg_status']
            self.collectives_by_file[rank_id] = collectives

            if 'health_check_results' in data:
                self.node_health_status[rank_id] = data['health_check_results']

            # self.collective_groups = self.matching_collectives_by_ordering_of_appearance_of_process_group_types()
            # logger.info(f"collective_groups: {self.collective_groups.keys()}")
            '''
            # Group collectives by process_group[0] and process_group[1] only (remove seq_id and op_id)
            for collective in collectives:
                # Use process_group[0] and process_group[1] as they contain the group identifiers
                key = (collective.process_group[0], collective.process_group[1])
                self.collective_groups[key].append(collective)
            '''
            return True
        except Exception as e:
            eprint(f"Error processing {filepath}: {str(e)}")
            return False

    def print_node_health_status(self, verbose: bool = False):
        """
        Print the node health status of the ranks
        """
        eprint("\n=== Node Health Status ===\n")
        healthy_ranks = defaultdict(list)
        unhealthy_ranks = defaultdict(list)
        status_parts = defaultdict(list)
        for rank_id, status in self.node_health_status.items():
            # TODO: Add host name to the output
            for device, result in status.items():
                if result['status'] == 'Healthy':
                    healthy_ranks[device].append(rank_id)
                else:
                    unhealthy_ranks[device].append(rank_id)
                    if verbose:
                        status_parts[device].append(f"({rank_id}: {result['output'].strip()})")
        for device, ranks in healthy_ranks.items():
            eprint(f"Healthy ranks {device}: {sorted(map(int, healthy_ranks[device]))}")
            eprint(f"Unhealthy ranks {device}: {sorted(map(int, unhealthy_ranks[device]))}")
            if len(unhealthy_ranks[device]) > 0:
                for status in status_parts[device]:
                    eprint(f"Unhealthy, {status}")

    def print_pg_configs(self, verbose: bool = False):
        """Print process group configurations in a more readable format."""
        eprint("\n=== Process Group Configurations ===\n")

        # Table header
        eprint(f"{'Group ID':<10} {'Description':<35} {'Ranks':<50}")
        eprint("-" * 95)
        # Sort by group ID numerically
        for group_id in sorted(self.pg_configs.keys(), key=lambda x: int(x)):
            group = self.pg_configs[group_id]
            ranks = str(group['ranks'])
            eprint(f"{group_id:<10} {group['desc']:<35} {ranks:<50}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze collective operations across JSON dump files.'
    )
    parser.add_argument(
        '--fr-path', type=str, help='Path to JSON files or directories containing JSON files'
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
        '-m',
        '--model',
        default="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
        help='Model to use for LLM analysis',
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Convert the trace file to json file, if the trace is binary, for debugging',
    )

    args = parser.parse_args()

    analyzer = CollectiveAnalyzer(args)
    analyzer.run_sync(args)


if __name__ == "__main__":
    main()
