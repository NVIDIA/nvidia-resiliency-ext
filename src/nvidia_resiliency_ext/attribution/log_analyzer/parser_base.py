#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Abstract base class for scheduler log parsers.

This module defines the interface for parsing scheduler output files.
Each scheduler (SLURM, Kubernetes, PBS) should implement BaseParser.

See spec Section 2.1 for architecture overview.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParseResult:
    """Common result structure for scheduler output parsing.

    Attributes:
        logs_dir: Path to logs directory if found (e.g., from LOGS_DIR= marker)
        restart_count: Number of scheduler restarts detected (e.g., << START PATHS >> markers)
        can_restart: Whether the job can be restarted (e.g., SLURM Requeue=1)
    """

    logs_dir: Optional[str] = None
    restart_count: int = 0
    can_restart: bool = False


class BaseParser(ABC):
    """Abstract base class for scheduler log parsers.

    Implementations should:
    - Parse scheduler-specific output formats
    - Extract LOGS_DIR or equivalent paths
    - Detect scheduler restarts (sched_restarts in spec terminology)
    - Check if job can be restarted

    Example:
        parser = SlurmParser()
        result = parser.parse(content)
        if result.logs_dir:
            print(f"Found logs at: {result.logs_dir}")
    """

    @abstractmethod
    def parse(self, content: str) -> ParseResult:
        """Parse scheduler output content.

        Args:
            content: Full content of the scheduler output file

        Returns:
            ParseResult with extracted information
        """
        pass

    @abstractmethod
    def parse_file(self, path: str) -> Optional[ParseResult]:
        """Read and parse a scheduler output file.

        Args:
            path: Path to the scheduler output file

        Returns:
            ParseResult or None if file cannot be read
        """
        pass
