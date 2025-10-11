"""
Utility functions for FR attribution test comparison and validation.
"""

import re
from typing import Any, Dict, Set


class FRAttributionOutputParser:
    """Parser for FR attribution output to extract key information for comparison."""

    def __init__(self):
        self.missing_ranks_pattern = r'(\d+)\s+\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([^|]+?)(?:\s|$)'
        self.processed_files_pattern = r'Successfully processed (\d+) files'
        self.group_types_pattern = r'Found group types: ([^\n]+)'

    def parse_output(self, output: str) -> Dict[str, Any]:
        """
        Parse FR attribution output and extract key information.

        Args:
            output: The output string from fr_attribution

        Returns:
            Dictionary containing parsed information
        """
        result = {
            'processed_files': 0,
            'group_types': [],
            'missing_ranks_by_pg': {},
            'process_groups': [],
            'scheduling_order': {},
            'raw_output': output,
        }

        # Extract processed files count
        processed_match = re.search(self.processed_files_pattern, output)
        if processed_match:
            result['processed_files'] = int(processed_match.group(1))

        # Extract group types
        group_types_match = re.search(self.group_types_pattern, output)
        if group_types_match:
            group_types_str = group_types_match.group(1)
            result['group_types'] = [gt.strip() for gt in group_types_str.split(',')]

        # Extract missing ranks information
        missing_ranks_matches = re.findall(self.missing_ranks_pattern, output)
        for match in missing_ranks_matches:
            pg_id = match[0].strip()
            pg_desc = match[1].strip()
            op_type = match[2].strip()
            size = match[3].strip()
            dtype = match[4].strip()
            missing_ranks_str = match[5].strip()

            if missing_ranks_str and missing_ranks_str.strip() != '':
                # Clean up the missing ranks string and extract only numeric values
                missing_ranks_str = missing_ranks_str.strip()
                # Split by comma and extract only numeric values
                missing_ranks = []
                for rank in missing_ranks_str.split(','):
                    rank = rank.strip()
                    # Extract only the numeric part (in case there's extra text)
                    numeric_match = re.search(r'\d+', rank)
                    if numeric_match:
                        missing_ranks.append(int(numeric_match.group()))

                if missing_ranks:  # Only add if we found valid ranks
                    result['missing_ranks_by_pg'][pg_id] = {
                        'pg_desc': pg_desc,
                        'op_type': op_type,
                        'size': size,
                        'dtype': dtype,
                        'missing_ranks': missing_ranks,
                    }
                    result['process_groups'].append(
                        {
                            'pg_id': pg_id,
                            'pg_desc': pg_desc,
                            'op_type': op_type,
                            'size': size,
                            'dtype': dtype,
                            'missing_ranks': missing_ranks,
                        }
                    )

        # Extract scheduling order
        scheduling_order_match = re.search(r'Using scheduling order: ({[^}]+})', output)
        if scheduling_order_match:
            import ast

            result['scheduling_order'] = ast.literal_eval(scheduling_order_match.group(1))

        return result

    def extract_missing_ranks_set(self, output: str) -> Set[int]:
        """
        Extract all missing ranks as a set from the output.

        Args:
            output: The output string from fr_attribution

        Returns:
            Set of missing rank numbers
        """
        missing_ranks = set()
        missing_ranks_matches = re.findall(self.missing_ranks_pattern, output)

        for match in missing_ranks_matches:
            missing_ranks_str = match[5].strip()
            if missing_ranks_str and missing_ranks_str.strip() != '':
                # Clean up the missing ranks string and extract only numeric values
                missing_ranks_str = missing_ranks_str.strip()
                # Split by comma and extract only numeric values
                for rank in missing_ranks_str.split(','):
                    rank = rank.strip()
                    # Extract only the numeric part (in case there's extra text)
                    numeric_match = re.search(r'\d+', rank)
                    if numeric_match:
                        missing_ranks.add(int(numeric_match.group()))

        return missing_ranks
