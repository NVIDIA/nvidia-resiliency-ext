#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""SLURM subprocess interactions for nvrx_smonsvc."""

import logging
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SlurmStats:
    """Statistics counters for SLURM operations."""

    squeue_calls: int = 0
    scontrol_calls: int = 0
    sacct_calls: int = 0
    squeue_failures: int = 0
    scontrol_failures: int = 0
    sacct_failures: int = 0


class SlurmClient:
    """
    Client for SLURM subprocess calls (squeue, scontrol, sacct).

    Handles batching and parsing of SLURM command output.
    """

    # Default timeouts
    DEFAULT_SQUEUE_TIMEOUT = 30  # seconds
    DEFAULT_SCONTROL_TIMEOUT = 45  # seconds (also used for sacct)

    # Batch sizes
    SCONTROL_BATCH_SIZE = 50
    SACCT_BATCH_SIZE = 50

    # Job states to query (skip PENDING - no output yet)
    SQUEUE_JOB_STATES = "RUNNING,COMPLETING"

    def __init__(
        self,
        partitions: list[str],
        user: str | None = None,
        all_users: bool = True,
        squeue_timeout: int = DEFAULT_SQUEUE_TIMEOUT,
        scontrol_timeout: int = DEFAULT_SCONTROL_TIMEOUT,
    ):
        """
        Initialize SLURM client.

        Args:
            partitions: List of SLURM partitions to monitor
            user: Filter jobs by user (ignored if all_users=True)
            all_users: If True, monitor jobs from all users
            squeue_timeout: Timeout for squeue commands
            scontrol_timeout: Timeout for scontrol/sacct commands
        """
        self.partitions = partitions
        self.user = user
        self.all_users = all_users
        self.squeue_timeout = squeue_timeout
        self.scontrol_timeout = scontrol_timeout
        self.stats = SlurmStats()

    def check_slurm_available(self) -> bool:
        """
        Check if SLURM commands are available.

        Returns:
            True if squeue is available and working, False otherwise.
        """
        try:
            result = subprocess.run(
                ["squeue", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired as e:
            logger.error(f"SLURM health check (squeue --version) timed out after {e.timeout}s")
            return False

    @staticmethod
    def get_available_partitions() -> list[str] | None:
        """
        Get list of available SLURM partitions using sinfo.

        Returns:
            List of partition names, or None if sinfo fails.
        """
        try:
            result = subprocess.run(
                ["sinfo", "--noheader", "-o", "%P"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            # Parse partition names (may have '*' suffix for default)
            partitions = []
            for line in result.stdout.strip().split("\n"):
                name = line.strip().rstrip("*")  # Remove default marker
                if name:
                    partitions.append(name)
            return partitions
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired as e:
            logger.warning(f"sinfo timed out after {e.timeout}s")
            return None

    def validate_partitions(self) -> tuple[list[str], list[str]]:
        """
        Validate configured partitions against available SLURM partitions.

        Returns:
            Tuple of (valid_partitions, invalid_partitions).
            If sinfo fails, returns ([], []) - caller should handle gracefully.
        """
        available = self.get_available_partitions()
        if available is None:
            logger.warning("Could not validate partitions - sinfo failed")
            return [], []

        available_set = set(available)
        valid = [p for p in self.partitions if p in available_set]
        invalid = [p for p in self.partitions if p not in available_set]
        return valid, invalid

    def get_running_jobs(
        self, existing_paths: dict[str, tuple[str, str]]
    ) -> dict[str, dict] | None:
        """
        Get currently queued/running jobs from squeue.

        Args:
            existing_paths: Dict of job_id -> (stdout, stderr) for jobs we already have paths for

        Returns:
            Dict of job_id -> job_info dict on success, None on failure.
            job_info contains: name, user, partition, state, stdout_path, stderr_path
        """
        cmd = [
            "squeue",
            "-o",
            "%i|%j|%u|%P|%T",  # job_id|name|user|partition|state
            "--noheader",
            "-p",
            ",".join(self.partitions),
            "-t",
            self.SQUEUE_JOB_STATES,
        ]

        if not self.all_users and self.user:
            cmd.extend(["-u", self.user])

        try:
            self.stats.squeue_calls += 1
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.squeue_timeout
            )
            if result.returncode != 0:
                logger.warning(f"squeue failed: {result.stderr}")
                self.stats.squeue_failures += 1
                return None

            # First pass: parse jobs and collect IDs needing path fetch
            parsed_jobs: list[tuple[str, str, str, str, str]] = []
            jobs_needing_paths: list[str] = []
            skipped_special_jobs = 0

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 5:
                    continue

                job_id, name, user, partition, state = parts[:5]
                parsed_jobs.append((job_id, name, user, partition, state))

                # Skip bracket notation (array job summaries like "1234[0-10]")
                # But allow het jobs with "+" notation (like "1234+0") - these are valid
                if "[" in job_id:
                    skipped_special_jobs += 1
                    continue

                # Need to fetch paths if we don't have them and job is running
                if job_id not in existing_paths and state in ("RUNNING", "COMPLETING"):
                    jobs_needing_paths.append(job_id)

            if skipped_special_jobs > 0:
                logger.info(
                    f"Skipped {skipped_special_jobs} array summary job IDs (bracket notation)"
                )

            # Batch fetch paths for jobs that need them
            fetched_paths = self.get_job_output_paths(jobs_needing_paths)

            # Second pass: create job info dicts
            jobs: dict[str, dict] = {}
            for job_id, name, user, partition, state in parsed_jobs:
                if job_id in existing_paths:
                    stdout_path, stderr_path = existing_paths[job_id]
                else:
                    stdout_path, stderr_path = fetched_paths.get(job_id, ("", ""))

                jobs[job_id] = {
                    "name": name,
                    "user": user,
                    "partition": partition,
                    "state": state,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                }

            return jobs

        except subprocess.TimeoutExpired as e:
            logger.warning(f"squeue timed out after {e.timeout}s")
            self.stats.squeue_failures += 1
            return None
        except FileNotFoundError:
            logger.error("squeue command not found. Is SLURM installed?")
            self.stats.squeue_failures += 1
            return None

    def get_job_output_paths(self, job_ids: list[str]) -> dict[str, tuple[str, str]]:
        """
        Get stdout and stderr paths for multiple jobs using batched scontrol/sacct calls.

        Args:
            job_ids: List of job IDs to fetch paths for

        Returns:
            Dict mapping job_id -> (stdout_path, stderr_path)
        """
        if not job_ids:
            return {}

        result_paths: dict[str, tuple[str, str]] = {}
        batch_size = self.SCONTROL_BATCH_SIZE
        total_batches = (len(job_ids) + batch_size - 1) // batch_size

        logger.info(
            f"Fetching output paths for {len(job_ids)} jobs via scontrol "
            f"({total_batches} batches of {batch_size})"
        )

        # Try scontrol first
        for i in range(0, len(job_ids), batch_size):
            batch = job_ids[i : i + batch_size]
            self._fetch_paths_scontrol_batch(batch, result_paths)

        # Use sacct for jobs scontrol couldn't find
        missing_job_ids = [jid for jid in job_ids if jid not in result_paths]
        if missing_job_ids:
            logger.info(f"scontrol missed {len(missing_job_ids)} jobs, trying sacct fallback")
            self._fetch_paths_sacct(missing_job_ids, result_paths)

        logger.debug(f"Fetched output paths for {len(result_paths)} jobs")
        return result_paths

    def _fetch_paths_scontrol_batch(
        self, job_ids: list[str], result_paths: dict[str, tuple[str, str]]
    ) -> None:
        """Fetch paths for a batch of jobs via scontrol."""
        job_ids_str = ",".join(job_ids)
        sample = job_ids[:3] + ["..."] + job_ids[-2:] if len(job_ids) > 5 else job_ids
        logger.debug(f"scontrol batch: {len(job_ids)} jobs, sample: {sample}")

        start_time = time.time()
        try:
            self.stats.scontrol_calls += 1
            result = subprocess.run(
                ["scontrol", "show", "job", job_ids_str],
                capture_output=True,
                text=True,
                timeout=self.scontrol_timeout,
            )
            elapsed = time.time() - start_time
            logger.debug(f"scontrol batch of {len(job_ids)} completed in {elapsed:.1f}s")

            if result.returncode != 0:
                logger.warning(f"scontrol failed (rc={result.returncode}): {result.stderr[:200]}")
                self.stats.scontrol_failures += 1
                return

            # Parse output
            current_job_id = None
            current_stdout = ""
            current_stderr = ""

            for line in result.stdout.split("\n"):
                if "JobId=" in line:
                    # Save previous job
                    if current_job_id:
                        result_paths[current_job_id] = (current_stdout, current_stderr)

                    # Extract job ID - handle array jobs
                    job_match = re.search(r"JobId=(\S+)", line)
                    array_job_match = re.search(r"ArrayJobId=(\d+)", line)
                    array_task_match = re.search(r"ArrayTaskId=(\d+)", line)

                    if array_job_match and array_task_match:
                        current_job_id = f"{array_job_match.group(1)}_{array_task_match.group(1)}"
                    elif job_match:
                        current_job_id = job_match.group(1)
                    else:
                        current_job_id = None

                    if current_job_id:
                        current_stdout = ""
                        current_stderr = ""
                elif "StdOut=" in line:
                    match = re.search(r"StdOut=(\S+)", line)
                    if match:
                        current_stdout = match.group(1)
                elif "StdErr=" in line:
                    match = re.search(r"StdErr=(\S+)", line)
                    if match:
                        current_stderr = match.group(1)

            # Save last job
            if current_job_id:
                result_paths[current_job_id] = (current_stdout, current_stderr)

        except subprocess.TimeoutExpired as e:
            logger.warning(
                f"scontrol timed out after {e.timeout}s fetching paths for {len(job_ids)} jobs"
            )
            self.stats.scontrol_failures += 1
        except FileNotFoundError:
            logger.error("scontrol command not found")
            self.stats.scontrol_failures += 1

    def _fetch_paths_sacct(
        self, job_ids: list[str], result_paths: dict[str, tuple[str, str]]
    ) -> None:
        """Fetch paths via sacct with batching."""
        if not job_ids:
            return

        batch_size = self.SACCT_BATCH_SIZE
        total_batches = (len(job_ids) + batch_size - 1) // batch_size

        logger.info(
            f"Fetching output paths for {len(job_ids)} jobs via sacct "
            f"({total_batches} batches of {batch_size})"
        )

        for i in range(0, len(job_ids), batch_size):
            batch = job_ids[i : i + batch_size]
            self._fetch_paths_sacct_batch(batch, result_paths)

    def _fetch_paths_sacct_batch(
        self, job_ids: list[str], result_paths: dict[str, tuple[str, str]]
    ) -> None:
        """Fetch paths for a batch of jobs via sacct."""
        if not job_ids:
            return

        # Normalize het job IDs: sacct uses base ID (1234) not component ID (1234+0)
        # Build mapping from normalized ID back to original IDs
        normalized_to_original: dict[str, list[str]] = {}
        for job_id in job_ids:
            base_id = job_id.split("+")[0] if "+" in job_id else job_id
            if base_id not in normalized_to_original:
                normalized_to_original[base_id] = []
            normalized_to_original[base_id].append(job_id)

        normalized_ids = list(normalized_to_original.keys())
        job_ids_str = ",".join(normalized_ids)
        sample = job_ids[:3] + ["..."] + job_ids[-2:] if len(job_ids) > 5 else job_ids
        logger.debug(f"sacct batch: {len(job_ids)} jobs, sample: {sample}")

        start_time = time.time()
        try:
            self.stats.sacct_calls += 1
            result = subprocess.run(
                [
                    "sacct",
                    "-j",
                    job_ids_str,
                    "--noheader",
                    "--parsable2",
                    "--format=JobID,StdOut,StdErr",
                ],
                capture_output=True,
                text=True,
                timeout=self.scontrol_timeout,
            )
            elapsed = time.time() - start_time
            logger.debug(f"sacct batch of {len(job_ids)} completed in {elapsed:.1f}s")

            if result.returncode != 0:
                logger.warning(f"sacct failed (rc={result.returncode}): {result.stderr[:200]}")
                self.stats.sacct_failures += 1
                return

            # Parse output
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    job_id_raw = parts[0]
                    stdout_path = parts[1] if parts[1] else ""
                    stderr_path = parts[2] if parts[2] else ""

                    # Handle job steps
                    if ".batch" in job_id_raw:
                        base_id = job_id_raw.replace(".batch", "")
                    elif "." in job_id_raw:
                        continue  # Skip other job steps
                    else:
                        base_id = job_id_raw

                    # Map results back to original het job IDs
                    if stdout_path and base_id in normalized_to_original:
                        for original_id in normalized_to_original[base_id]:
                            if original_id not in result_paths:
                                result_paths[original_id] = (stdout_path, stderr_path)

        except subprocess.TimeoutExpired as e:
            logger.warning(f"sacct timed out after {e.timeout}s for {len(job_ids)} jobs")
            self.stats.sacct_failures += 1
        except FileNotFoundError:
            logger.error("sacct command not found")
            self.stats.sacct_failures += 1


def expand_slurm_patterns(
    path: str,
    job_id: str,
    job_name: str,
    job_user: str,
    on_unexpanded: Callable[[str], None] | None = None,
) -> str:
    """
    Expand SLURM filename pattern placeholders in a path.

    Common SLURM patterns:
        %j - Job ID (or ArrayJobId_ArrayTaskId for array jobs)
        %J - Same as %j
        %A - Array job ID (master job ID, or job ID if not array)
        %a - Array task ID (or "4294967294" if not array job)
        %t - Same as %a (task ID)
        %x - Job name
        %u - User name

    Args:
        path: Path potentially containing SLURM patterns
        job_id: Job ID (e.g., "1600406_239" for array, "1602234" for regular)
        job_name: Job name
        job_user: Job user
        on_unexpanded: Optional callback for unexpanded patterns

    Returns:
        Path with patterns expanded
    """
    if "%" not in path:
        return path

    # Parse array job ID and task ID
    if "_" in job_id:
        parts = job_id.split("_", 1)
        array_job_id = parts[0]
        array_task_id = parts[1]
    else:
        array_job_id = job_id
        array_task_id = "4294967294"  # SLURM's "not an array" value

    substitutions = {
        "%j": job_id,
        "%J": job_id,
        "%A": array_job_id,
        "%a": array_task_id,
        "%t": array_task_id,
        "%x": job_name,
        "%u": job_user,
    }

    result = path
    for pattern, value in substitutions.items():
        result = result.replace(pattern, value)

    if result != path:
        logger.debug(f"[{job_id}] Expanded path: {path} -> {result}")

    # Warn about remaining unexpanded patterns
    remaining = re.findall(r"%[a-zA-Z]", result)
    if remaining:
        logger.warning(f"[{job_id}] Cannot expand SLURM patterns {remaining} in path: {result}")
        if on_unexpanded:
            on_unexpanded(result)

    return result
