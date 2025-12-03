# Auto-generated for consolidated logging with O_APPEND
import os
import sys


class _RankPrefixWriter:
    '''Wraps a file object to add rank: prefix to each line (like srun -l).'''

    def __init__(self, wrapped, rank, world_size=None):
        self._wrapped = wrapped
        # Format like srun -l: right-aligned rank number without brackets
        # Determine padding width based on world_size (like srun -l does)
        if world_size is not None and world_size > 1:
            # Calculate width needed for the max rank (world_size - 1)
            width = len(str(world_size - 1))
            self._prefix = f'{str(rank):>{width}}: '
        else:
            # No padding if world_size is unknown
            self._prefix = f'{rank}: '
        self._at_line_start = True

    def write(self, text):
        if not text:
            return 0

        # Handle line-by-line prefixing
        lines = text.split('\n')
        result = []

        for i, line in enumerate(lines):
            # Add prefix at line start (or after previous newline)
            if i == 0 and self._at_line_start and line:
                result.append(self._prefix + line)
            elif i > 0 and line:  # Not first line and not empty
                result.append(self._prefix + line)
            else:
                result.append(line)

        output = '\n'.join(result)

        # Update state: are we at a line start for next write?
        self._at_line_start = text.endswith('\n')

        return self._wrapped.write(output)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# Get the log file path from environment
log_file = os.environ.get('TORCHELASTIC_CONSOLIDATED_LOG')
if log_file:
    try:
        # Get rank for prefix (like srun -l, we use global rank)
        rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '?'))

        # Get world_size to determine padding width (like srun -l)
        world_size_str = os.environ.get('WORLD_SIZE')
        world_size = int(world_size_str) if world_size_str else None

        # Open the log file with O_APPEND flag (atomic appends, safe for concurrent writes)
        # Each child opens the file since spawn doesn't inherit file descriptors
        log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

        # Redirect stdout and stderr to the log file
        # This happens BEFORE any user code or PyTorch code runs
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())

        # Close the extra fd (stdout/stderr now point to it via dup2)
        os.close(log_fd)

        # Reopen Python's stdout/stderr with line buffering
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

        # Wrap with rank prefix writer (like srun -l)
        sys.stdout = _RankPrefixWriter(sys.stdout, rank, world_size)
        sys.stderr = _RankPrefixWriter(sys.stderr, rank, world_size)
    except Exception as e:
        # Can't write to redirected stderr at this point, so write to original stderr (fd 2)
        import traceback

        orig_stderr = os.fdopen(2, 'w')
        print(
            f"[PID {os.getpid()}] NVRx FT: Failed to redirect to consolidated log '{log_file}': {e}",
            file=orig_stderr,
            flush=True,
        )
        traceback.print_exc(file=orig_stderr)
        orig_stderr.flush()
