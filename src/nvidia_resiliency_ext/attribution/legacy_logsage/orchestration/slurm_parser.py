# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse SLURM output files to extract LOGS_DIR and detect scheduler restarts.

This module parses SLURM output files looking for:
- << START PATHS >> / << END PATHS >> markers (scheduler restarts)
- LOGS_DIR= declarations (splitlog mode log directory)
- ``Writing logs to <abs-path>`` lines (fallback when markers / LOGS_DIR= are absent)
- Requeue indicators (job restart capability)

See spec Section 13 for marker parsing details.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .parser_base import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# Markers for parsing SLURM output (see spec Section 3.3)
START_PATHS_MARKER = "<< START PATHS >>"
END_PATHS_MARKER = "<< END PATHS >>"
LOGS_DIR_PATTERN = re.compile(r"^LOGS_DIR=(.+)$", re.MULTILINE)
# e.g. "Writing logs to /path/to/run/logs" (common launcher convention without << START PATHS >>)
_WRITING_LOGS_TO_PATTERN = re.compile(r"^Writing logs to\s+(\S+)", re.MULTILINE)


@dataclass
class SlurmOutputInfo:
    """Information extracted from a SLURM output file.

    ParseResult from parser_base.py is the parser-wide return shape; this dataclass
    carries SLURM-specific details for helper callers.
    """

    logs_dir: Optional[str]  # Path to LOGS_DIR if found
    cycle_count: int  # Number of << START PATHS >> blocks (sched_restarts)
    has_requeue: bool  # Whether job has Requeue=1 (can restart)


class SlurmParser(BaseParser):
    """Parser for SLURM scheduler output files.

    Detects:
    - Scheduler restarts via << START PATHS >> markers
    - LOGS_DIR declarations within marker blocks
    - Requeue capability (Requeue=1 or #SBATCH --requeue)

    Example:
        parser = SlurmParser()
        result = parser.parse_file("/path/to/slurm-12345.out")
        if result and result.logs_dir:
            print(f"Splitlog mode: {result.logs_dir}")
    """

    def parse(self, content: str) -> ParseResult:
        """Parse SLURM output content.

        Args:
            content: Full content of the SLURM output file

        Returns:
            ParseResult with extracted information
        """
        info = parse_slurm_output(content)
        return ParseResult(
            logs_dir=info.logs_dir,
            restart_count=info.cycle_count,
            can_restart=info.has_requeue,
        )

    def parse_file(self, path: str) -> Optional[ParseResult]:
        """Read and parse a SLURM output file.

        Args:
            path: Path to the SLURM output file

        Returns:
            ParseResult or None if file cannot be read
        """
        info = read_and_parse_slurm_output(path)
        if info is None:
            return None
        return ParseResult(
            logs_dir=info.logs_dir,
            restart_count=info.cycle_count,
            can_restart=info.has_requeue,
        )


def parse_slurm_output(content: str) -> SlurmOutputInfo:
    """
    Parse a SLURM output file to extract LOGS_DIR and cycle information.

    Args:
        content: Full content of the SLURM output file

    Returns:
        SlurmOutputInfo with extracted information
    """
    # Count cycles by counting lines that ARE the marker (not just contain it)
    # This avoids false positives from log output containing the marker text
    cycle_count = _count_marker_lines(content, START_PATHS_MARKER)

    # Extract LOGS_DIR from the LAST << START PATHS >> block
    # (spec Section 13.4: use latest LOGS_DIR if it changes between restarts)
    logs_dir = _extract_logs_dir_from_path_blocks(content)
    if logs_dir is None:
        logs_dir = _extract_writing_logs_dir(content)

    # Check for Requeue=1 indicator (can restart)
    has_requeue = _check_requeue(content)

    return SlurmOutputInfo(
        logs_dir=logs_dir,
        cycle_count=cycle_count,
        has_requeue=has_requeue,
    )


def _count_marker_lines(content: str, marker: str) -> int:
    """
    Count lines that ARE the marker (with optional surrounding whitespace).

    This avoids false positives from log output that happens to contain
    the marker text as part of a longer line.

    Args:
        content: File content to search
        marker: Marker string to match

    Returns:
        Number of lines that match the marker exactly
    """
    count = 0
    for line in content.splitlines():
        if line.strip() == marker:
            count += 1
    return count


def _extract_writing_logs_dir(content: str) -> Optional[str]:
    """Last ``Writing logs to <path>`` line with an absolute path (splitlog fallback)."""
    last: Optional[str] = None
    for m in _WRITING_LOGS_TO_PATTERN.finditer(content):
        p = m.group(1).strip().rstrip("/")
        if p.startswith("/"):
            last = p
    if last:
        logger.debug("_extract_writing_logs_dir: final path=%s", last)
    return last


def _extract_logs_dir_from_path_blocks(content: str) -> Optional[str]:
    """
    Extract LOGS_DIR from the SLURM output content.

    Looks for LOGS_DIR= within << START PATHS >> blocks.
    Uses the LAST occurrence per spec Section 13.4.

    Uses line-based parsing to avoid false positives from log output
    that contains the marker text as part of a longer line.

    Args:
        content: SLURM output content

    Returns:
        LOGS_DIR path or None if not found
    """
    logs_dir = None
    lines = content.splitlines()
    in_block = False
    block_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped == START_PATHS_MARKER:
            in_block = True
            block_count += 1
            logger.debug(
                "_extract_logs_dir_from_path_blocks: entered START PATHS block #%s",
                block_count,
            )
            continue

        if stripped == END_PATHS_MARKER:
            in_block = False
            logger.debug(
                "_extract_logs_dir_from_path_blocks: exited END PATHS block #%s",
                block_count,
            )
            continue

        # Look for LOGS_DIR= within a block
        if in_block:
            match = LOGS_DIR_PATTERN.match(stripped)
            if match:
                logs_dir = match.group(1).strip().rstrip("/")
                logger.debug(
                    "_extract_logs_dir_from_path_blocks: found LOGS_DIR=%s in block #%s",
                    logs_dir,
                    block_count,
                )

    if logs_dir:
        logger.debug(
            "_extract_logs_dir_from_path_blocks: final LOGS_DIR=%s (from %s blocks)",
            logs_dir,
            block_count,
        )
    elif block_count > 0:
        logger.debug(
            "_extract_logs_dir_from_path_blocks: no LOGS_DIR in %s START PATHS blocks",
            block_count,
        )

    return logs_dir


def _check_requeue(content: str) -> bool:
    """
    Check if the job has Requeue=1 (can be restarted).

    This can appear in the SLURM output as part of job info or
    in the environment variables.

    Uses line-based matching to reduce false positives, but allows
    these patterns to appear as part of a line (e.g., in scontrol output).

    Args:
        content: SLURM output content

    Returns:
        True if Requeue=1 is found
    """
    for line in content.splitlines():
        stripped = line.strip()
        # Check for Requeue=1 pattern (appears in scontrol output)
        # Allow as part of line since scontrol shows "Requeue=1 Restarts=0"
        if "Requeue=1" in stripped:
            return True
        # Check for #SBATCH --requeue directive
        if stripped.startswith("#SBATCH") and "--requeue" in stripped:
            return True
    return False


def read_and_parse_slurm_output(path: str) -> Optional[SlurmOutputInfo]:
    """
    Read a SLURM output file and parse it.

    Args:
        path: Path to the SLURM output file

    Returns:
        SlurmOutputInfo or None if file cannot be read
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        content_len = len(content)
        logger.debug(f"read_and_parse_slurm_output: read {content_len} bytes from {path}")
        result = parse_slurm_output(content)
        logger.debug(
            f"read_and_parse_slurm_output: result - "
            f"logs_dir={result.logs_dir}, cycle_count={result.cycle_count}, "
            f"has_requeue={result.has_requeue}"
        )
        return result
    except FileNotFoundError:
        logger.warning(f"SLURM output file not found: {path}")
        return None
    except PermissionError:
        logger.warning(f"Permission denied reading SLURM output: {path}")
        return None
    except Exception as e:
        logger.error(f"Error reading SLURM output {path}: {e}")
        return None
