================================================================================
NVRX ATTRIBUTION SERVICE - TECHNICAL SPECIFICATION
================================================================================

TABLE OF CONTENTS
--------------------------------------------------------------------------------

FOUNDATION
    1.  Overview (1.1-1.5)
    2.  Project Structure (2.1-2.2)
    3.  Configuration (3.1-3.6)
    4.  Startup / Shutdown (4.1-4.2)
    5.  Data Structures (5.1-5.6)

HTTP INTERFACE
    6.  HTTP API (6.1-6.3)
    7.  Error Responses

VALIDATION & FLOWS
    8.  Path Validation (8.1-8.4)
    9.  POST Flow (9.1-9.3)
    10. GET Flow (10.1-10.5)
    11. Background Poll

PROCESSING
    12. LLM Analysis (12.1-12.8)
    13. Log File Marker Parsing (13.1-13.5)

SPLITLOG MODE
    14. Splitlog Overview (14.1-14.3)
    15. SplitlogTracker (15.1-15.4)
    16. Restart Detection & Log File Discovery (16.1-16.7)
    17. Splitlog GET Flow (17.1-17.4)

MAINTENANCE & INTERNALS
    18. Cleanup (18.1-18.2)
    19. Concurrency & Thread/Async Coordination (19.1-19.6)
    20. Counters (20.1-20.2)

OPTIONAL
    21. Dataflow (Optional)
    22. Logging
    23. Dependencies

DEVELOPMENT
    24. Testing Strategy (24.1-24.4)
    25. Deployment (25.1-25.6)

APPENDIX
    A.  Glossary
    B.  Quick Reference

================================================================================
                                  FOUNDATION
================================================================================

1. OVERVIEW
--------------------------------------------------------------------------------

Attribution service for analyzing job log files using LLM.
- POST /logs: Submit a log file for tracking
- GET /logs: Retrieve analysis results
- Background poll handles pending jobs, splitlog tracking, and cleanup

HIGH-LEVEL ARCHITECTURE:

    ┌─────────────┐     POST/GET     ┌─────────────────┐
    │   Client    │ ───────────────► │   attrsvc       │
    │  (smonsvc)  │ ◄─────────────── │   (FastAPI)     │
    └─────────────┘                  └────────┬────────┘
                                              │
                     ┌────────────────────────┼────────────────────────┐
                     │                        │                        │
               ┌─────▼─────┐          ┌───────▼───────┐        ┌───────▼───────┐
               │ Background│          │  MCP Client   │        │  Filesystem   │
               │ Poll      │          │  (LLM)        │        │  (LOGS_DIR)   │
               └───────────┘          └───────────────┘        └───────────────┘

1.1 OPERATING MODES
.......................

| Mode     | When Used                          | What's Analyzed              |
|----------|------------------------------------|-----------------------------|
| PENDING  | File not yet available or too short| (waiting for file)          |
| SINGLE   | No job_id, or LOGS_DIR not found   | Job output file directly    |
| SPLITLOG | job_id provided + LOGS_DIR found   | Log files in LOGS_DIR       |

Mode detection flow:
    POST with job_id → parse job output for LOGS_DIR
        → LOGS_DIR found + readable → SPLITLOG
        → LOGS_DIR found + not readable → ERROR
        → LOGS_DIR not found yet → PENDING (poll will re-check)
    
    POST without job_id → SINGLE (markers checked for warning, but SPLITLOG requires job_id)

1.2 TERMINOLOGY
................

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ SCHEDULER RESTART (sched_restart)                                       │
    │   - External orchestrator (SLURM, Kubernetes, etc.) restarts the job    │
    │   - Detected by: << START PATHS >> marker in job output file            │
    │   - Creates: New log file in LOGS_DIR                                   │
    │   - Field: job.sched_restarts (count)                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ WORKLOAD RESTART (abbreviated: wl_restart)                                │
    │   - Training framework restarts within same scheduler allocation        │
    │   - Detected by: Cycle: N marker in log file content                    │
    │   - Creates: New section within same log file (or new file if _cycleN)  │
    │   - Field: file_info[filename].wl_restarts (count per file)             │
    │   - Abbreviation used in: field names, query params, code               │
    └─────────────────────────────────────────────────────────────────────────┘

    Hierarchy:
        Job
        └── Scheduler restart 0       ← << START PATHS >> in job output
            ├── Workload restart 0    ← Cycle: 0 in log file
            ├── Workload restart 1    ← Cycle: 1 in log file
            └── Workload restart 2    ← Cycle: 2 in log file
        └── Scheduler restart 1       ← << START PATHS >> in job output
            ├── Workload restart 0    ← Cycle: 0 in log file
            └── Workload restart 1    ← Cycle: 1 in log file

1.3 USE CASES
..............

SUMMARY:

    | Use Case | Mode | Files Analyzed | LLM Calls | Sorting |
    |----------|------|----------------|-----------|---------|
    | 1: Single-file | SINGLE | Job output file | 1 | N/A |
    | 2a: Per sched restart | SPLITLOG | N files in LOGS_DIR | N | By timestamp |
    | 2b: Per wl restart | SPLITLOG | N files in LOGS_DIR | N | By cycle number |

USE CASE 1: Single-file mode (no LOGS_DIR)

    All output goes to job output file. May have scheduler restarts and/or
    workload restarts, all in the same file.

    Job output file (e.g., slurm-12345.out):
    ┌─────────────────────────────────────────────────────────────┐
    │ << START PATHS >>                                           │ ← Sched restart 0
    │ << END PATHS >>                                             │
    │ profiling.py:... Cycle: 0                                   │ ← WL restart 0
    │ ... training output ...                                     │
    │ profiling.py:... Cycle: 1                                   │ ← WL restart 1
    │ ... training output ...                                     │
    │                                                             │
    │ << START PATHS >>                                           │ ← Sched restart 1
    │ << END PATHS >>                                             │
    │ profiling.py:... Cycle: 0                                   │ ← WL restart 0
    │ ... training output ...                                     │
    └─────────────────────────────────────────────────────────────┘

    Behavior:
    - Mode: SINGLE (no LOGS_DIR marker)
    - Analyzed: Entire job output file
    - LLM calls: 1
    - Content splitting: See Section 1.4

USE CASE 2a: Splitlog - One log file per scheduler restart (multi-wl-restart per file)

    Workload creates one log file per scheduler restart. Each file may contain
    multiple workload restarts, delimited by Cycle: N markers.

    Job output file (e.g., slurm-1523572.out):
    ┌─────────────────────────────────────────────────────────────┐
    │ << START PATHS >>                                           │ ← Sched restart 0
    │ LOGS_DIR=/logs/job_1523572                                  │
    │ << END PATHS >>                                             │
    │                                                             │
    │ << START PATHS >>                                           │ ← Sched restart 1
    │ LOGS_DIR=/logs/job_1523572                                  │
    │ << END PATHS >>                                             │
    └─────────────────────────────────────────────────────────────┘

    LOGS_DIR contents (one file per scheduler restart):
    ├── 55B_model_1523572_date_26-01-19_time_01-51-12.log  ← Sched restart 0
    └── 55B_model_1523572_date_26-01-19_time_04-22-45.log  ← Sched restart 1

    Inside each log file (multiple workload restarts):
    ┌─────────────────────────────────────────────────────────────┐
    │ profiling.py:... Cycle: 0                                   │ ← WL restart 0
    │ ... training output ...                                     │
    │ profiling.py:... Cycle: 1                                   │ ← WL restart 1
    │ ... training output ...                                     │
    │ profiling.py:... Cycle: 2                                   │ ← WL restart 2
    │ ... training output ...                                     │
    └─────────────────────────────────────────────────────────────┘

    Behavior:
    - Mode: SPLITLOG
    - File sorting: By date/time in filename
    - LLM calls: 1 per log file
    - Content splitting: See Section 1.4

USE CASE 2b: Splitlog - One log file per workload restart (single per file)

    Workload creates one log file per workload restart. Each file contains
    only one workload restart (no Cycle: N markers needed).

    Job output file (e.g., slurm-1617268.out):
    ┌─────────────────────────────────────────────────────────────┐
    │ << START PATHS >>                                           │ ← Sched restart 0
    │ LOGS_DIR=/logs/job_1617268                                  │
    │ << END PATHS >>                                             │
    └─────────────────────────────────────────────────────────────┘

    LOGS_DIR contents (one file per workload restart):
    ├── 55B_model_1617268_date_26-01-27_time_12-50-58_cycle0.log  ← WL restart 0
    ├── 55B_model_1617268_date_26-01-27_time_14-30-22_cycle1.log  ← WL restart 1
    └── 55B_model_1617268_date_26-01-27_time_16-05-11_cycle2.log  ← WL restart 2

    Behavior:
    - Mode: SPLITLOG
    - File sorting: By cycle number in filename (preferred)
    - LLM calls: 1 per log file
    - Content splitting: Not needed (single workload restart per file)

1.4 CONTENT SPLITTING
.....................

Uses chunk_logs_strict from nvidia_resiliency_ext.attribution.log_analyzer module.

When a log file contains multiple workload restarts, the LLM analyzer
splits content by Cycle: N markers:

    Pattern: profiling.py:.*Cycle: N

    Algorithm:
        1. Find all Cycle: N markers in file
        2. Extract lines from Cycle: N to Cycle: N+1 (or EOF)
        3. Each chunk analyzed separately
        4. Results combined in output

    When no markers found: Entire file treated as single workload restart

1.5 DEPLOYMENT MODEL
....................

- Single instance per cluster (no multi-instance state sharing)
- State is primarily in-memory with optional cache persistence (see 3.7)
- With CACHE_FILE configured: Analysis results survive restarts
- Without CACHE_FILE: Clients must re-submit jobs after restart

================================================================================

2. PROJECT STRUCTURE
--------------------------------------------------------------------------------

Two-layer architecture: reusable library + service-specific HTTP wrapper.
Design is scheduler-agnostic to support SLURM, Kubernetes, PBS, etc.

2.1 LIBRARY LAYER
.................

Shared, reusable components in src/nvidia_resiliency_ext/attribution/:

src/nvidia_resiliency_ext/attribution/
├── __init__.py              # Re-exports from log_analyzer
├── base.py                  # Base classes
├── utils.py                 # Common utilities
│
├── # ─── LOG ANALYZER (core functionality) ───
├── log_analyzer/
│   ├── __init__.py          # Exports all (LogAnalyzer, ErrorCode, Job, etc.)
│   ├── analyzer.py          # LogAnalyzer - main API (usable without HTTP)
│   ├── config.py            # ErrorCode enum, TTL/limit constants
│   ├── coalescer.py         # RequestCoalescer - request deduplication & caching
│   ├── job.py               # Job, FileInfo, JobMode dataclasses
│   ├── splitlog.py          # SplitlogTracker - background poll, file discovery
│   ├── parser_base.py       # Abstract parser interface (BaseParser, ParseResult)
│   ├── slurm_parser.py      # SLURM output parsing (<< START PATHS >>)
│   ├── nvrx_logsage.py      # Log analysis prompts
│   └── utils.py             # Patterns (CYCLE_LOG_PATTERN, JOB_ID_PATTERN)
│
├── postprocessing/          # Result posting and notifications
│   ├── __init__.py          # Exports config, configure(), ResultPoster, post_results, Slack
│   ├── config.py            # PostprocessingConfig singleton and configure()
│   ├── base.py              # ResultPoster, post_results (generic framework)
│   └── slack.py             # Slack notifications for terminal failures
│
└── mcp_integration/         # MCP client/server for LLM communication
    ├── __init__.py
    ├── mcp_client.py        # MCP client wrapper
    ├── mcp_server.py        # MCP server for log_analyzer
    └── ...

2.2 SERVICE LAYER
.................

HTTP API service in services/nvrx_attrsvc/. Run and deployment scripts live under
deploy/ (run_attrsvc.sh, snapshot_attrsvc.sh, Dockerfile, kubernetes.yaml, slurm.sbatch).

See README.md for:
    - "Files" section: File descriptions
    - "Architecture" section: Two-layer design overview
    - "Python API" section: Usage examples

File organization by layer:

    SERVICE CORE:
        service.py, config.py

    HTTP ACCESS LAYER:
        app.py

    POSTPROCESSING:
        config.setup() wires lib postprocessing (ResultPoster(dataflow.post), Slack); dataflow.py

PYTHON API REFERENCE:

    See README.md "Python API" section for usage examples.

    # ─── Class Signatures ───

    @dataclass
    class AnalyzerConfig:
        allowed_root: str                    # Required: base directory for paths
        compute_timeout: float = 300.0       # LLM timeout in seconds
        llm_model: str = "..."               # LLM model identifier
        llm_temperature: float = 0.0
        llm_top_p: float = 1.0
        llm_max_tokens: int = 8192
        # cluster_name / dataflow_index live in postprocessing.config (set at service startup)

    class LogAnalyzer:
        def __init__(self, config: AnalyzerConfig) -> None: ...
        def shutdown(self) -> None: ...
        
        async def submit(
            self, log_path: str, user: str = "unknown", job_id: str | None = None
        ) -> SubmitResult | AnalyzerError: ...
        
        async def analyze(
            self, log_path: str, file: str | None = None, wl_restart: int | None = None
        ) -> AnalysisResult | SplitlogAnalysisResult | AnalyzerError: ...
        
        def read_file_preview(self, log_path: str, max_bytes: int = 4096) -> FilePreviewResult | AnalyzerError: ...
        async def get_stats(self) -> Dict[str, Any]: ...
        def get_all_jobs(self) -> Dict[str, Any]: ...

    # ─── HTTP Service Wrapper ───
    
    class AttributionService:
        """Thin wrapper around LogAnalyzer for HTTP service."""
        def __init__(self, cfg: Settings) -> None: ...
        def shutdown(self) -> None: ...
        # Delegates to LogAnalyzer, returns AnalyzerError on failure

    # ─── Result Types ───

    @dataclass
    class SubmitResult:
        submitted: bool
        normalized_path: str
        mode: str                 # "pending", "single", "splitlog"
        logs_dir: str | None
        sched_restarts: int
        files_analyzed: int

    @dataclass
    class AnalysisResult:
        result: Dict[str, Any]    # LLM analysis result
        status: str               # "completed"

    @dataclass
    class SplitlogAnalysisResult:
        result: Dict[str, Any]
        status: str
        mode: str                 # "splitlog"
        sched_restarts: int
        log_file: str
        wl_restart: int

    @dataclass
    class AnalyzerError:
        error_code: ErrorCode
        message: str

================================================================================

3. CONFIGURATION
--------------------------------------------------------------------------------

3.1 ENVIRONMENT VARIABLES
.........................

See README.md for the complete list of environment variables and their defaults.

Prefix: NVRX_ATTRSVC_ (except NVIDIA_API_KEY)

LLM settings (LLM_MODEL, LLM_TEMPERATURE, etc.) are optional - library defaults
from AnalyzerConfig are used if not set. Service can override via environment.

Rate limiting uses slowapi library with custom exception handler 
(increments counters, logs, returns 429).

SETTINGS VALIDATION (config.py):

    Use pydantic-settings with @field_validator decorators for fail-fast startup.

    | Field            | Validation                                              |
    |------------------|---------------------------------------------------------|
    | ALLOWED_ROOT     | Required, absolute path, exists, readable/traversable  |
    | NVIDIA_API_KEY   | Required, non-empty, starts with "nvapi-"               |
    | LOG_LEVEL_NAME   | Must be DEBUG/INFO/WARNING/ERROR/CRITICAL               |
    | PORT             | Range 1-65535                                           |
    | COMPUTE_TIMEOUT  | Must be positive (if set)                               |
    | LLM_TEMPERATURE  | Must be 0.0-2.0 (if set)                                |
    | LLM_TOP_P        | Must be 0.0-1.0 (if set)                                |
    | LLM_MAX_TOKENS   | Must be positive (if set)                               |
    | RATE_LIMIT_*     | Must match "N/period" format (e.g., "60/minute")        |

    BEHAVIOR:
        - Validators run at Settings() instantiation
        - Any failure → startup fails with clear error message
        - NVIDIA_API_KEY loaded from (in order):
            1. NVIDIA_API_KEY environment variable
            2. NVIDIA_API_KEY_FILE environment variable (path to file)
            3. ~/.nvidia_api_key file
            4. ~/.config/nvrx/nvidia_api_key file

3.2 CONSTANTS
.............

Defined in log_analyzer/config.py to avoid magic numbers/strings.

Timeouts (TTL):
    TTL_PENDING_SECONDS         = 604800    # 1 week - pending jobs expiry
    TTL_TERMINATED_SECONDS      = 3600      # 1 hour - terminated jobs (after GET)
    TTL_MAX_JOB_AGE_SECONDS     = 15552000  # 6 months - non-terminated safety net
    TTL_IN_FLIGHT_STUCK_SECONDS = 600       # 2 × COMPUTE_TIMEOUT - safety net

Intervals:
    POLL_INTERVAL_SECONDS           = 300   # 5 minutes
    GRACEFUL_SHUTDOWN_TIMEOUT       = 30    # uvicorn shutdown
    DEFAULT_COMPUTE_TIMEOUT_SECONDS = 300   # 5 minutes - LLM analysis timeout

Thresholds:
    MIN_FILE_SIZE_KB            = 4     # Minimum file size (KB) for classification
    HEALTH_DEGRADED_THRESHOLD   = 0.20  # 20% error rate → degraded
    HEALTH_FAIL_THRESHOLD       = 0.50  # 50% error rate → fail

Pagination:
    JOBS_DEFAULT_LIMIT      = 100   # Default page size for /jobs
    JOBS_MAX_LIMIT          = 1000  # Maximum page size
    RESULT_TRUNCATE_LENGTH  = 1000  # Chars in /jobs result field

Memory:
    MAX_JOBS = 100000   # ~500 MB at capacity (100K × 5KB per job)
    
    When limit reached: POST returns 503 JOB_LIMIT_REACHED

3.3 LOG FILE MARKERS
....................

Written to jobOutputLogFile by job wrapper script at job start.
Used for mode detection (SPLITLOG vs SINGLE). See section 13.

    MARKER_START_PATHS = "<< START PATHS >>"   # Paths block start
    MARKER_END_PATHS   = "<< END PATHS >>"     # Paths block end
    MARKER_LOGS_DIR    = "LOGS_DIR="           # Logs directory path prefix

Example:
    << START PATHS >>
    LOGS_DIR=/path/to/logs
    << END PATHS >>
    ... training logs ...

3.4 ENUMS
.........

class JobMode(StrEnum):
    PENDING  = "pending"
    SINGLE   = "single"
    SPLITLOG = "splitlog"

# Note: No AnalysisState enum - errors return AnalyzerError
# Success returns AnalysisResult with status="completed"

class ErrorCode(StrEnum):
    INVALID_PATH          = "invalid_path"
    OUTSIDE_ROOT          = "outside_root"
    NOT_FOUND             = "not_found"
    NOT_READABLE          = "not_readable"
    NOT_REGULAR           = "not_regular"
    EMPTY_FILE            = "empty_file"
    LOGS_DIR_NOT_READABLE = "logs_dir_not_readable"
    JOB_LIMIT_REACHED     = "job_limit_reached"
    INTERNAL_ERROR        = "internal_error"

class HealthStatus(StrEnum):
    OK       = "ok"
    DEGRADED = "degraded"
    FAIL     = "fail"

3.5 ERROR CODE TO HTTP STATUS
.............................

See Section 7 ERROR RESPONSES for the complete error code table with HTTP status
codes and trigger conditions.

3.6 PATH-BASED JOB_ID EXTRACTION
................................

Best-effort extraction when job_id not provided (GET-without-POST).
Uses extract_job_metadata() from log_analyzer module.

JOB_ID_PATTERNS (tried in order):
    r"_(\d+)_date_"                   # Generic: foo_12345_date_2024...
    r"[/\\]job_(\d+)[/\\]"            # Generic: job_12345/output.log
    r"slurm-(\d+)\.(out|err|log)$"    # SLURM-specific: slurm-12345.out
    r"[/\\](\d{6,})\.(out|err|log)$"  # Generic: /12345678.out
    r"_(\d{6,})\.(out|err|log)$"      # Generic: prefix_12345678.out

Note: Patterns are scheduler-agnostic where possible. SLURM-specific pattern
is included for backward compatibility. Additional patterns can be added for
other schedulers (Kubernetes, PBS, LSF, etc.) as needed.

User: NOT extractable from path → always "unknown" for GET-without-POST

3.7 PROCESSED FILES LEDGER (CACHE PERSISTENCE)
...............................................

Optional feature to track processed files across restarts. Set CACHE_FILE to enable.

PURPOSE:

    The cache acts as a "processed files ledger" - tracking which files have been
    analyzed and posted to Elasticsearch. This prevents duplicate processing when:
    
    1. Service restarts and smonsvc resubmits recently completed jobs
    2. Same file is requested multiple times
    
    Without the ledger, service restart would trigger re-analysis of all recently
    completed files, wasting LLM calls for files already posted to ES.

ENVIRONMENT VARIABLES:
    NVRX_ATTRSVC_CACHE_FILE=/path/to/cache.json
    NVRX_ATTRSVC_CACHE_GRACE_PERIOD_SECONDS=600  # default 10 min

WHAT'S STORED:

    For each processed file: (path, mtime, size, result)
    
    - path: File that was analyzed
    - mtime, size: File metadata at analysis time (for validation)
    - result: Analysis result (also in ES, kept here for convenience)

VALIDATION STRATEGY:

    | Phase              | Behavior                                          |
    |--------------------|---------------------------------------------------|
    | Grace period       | Serve from cache without file validation          |
    | After grace period | stat() file on hit; invalidate if (mtime, size) changed |
    | Eviction           | Remove entries when file.mtime > 14 days          |

    The configurable grace period (default 10 min) absorbs straggling writes
    at end of files, preventing unnecessary re-analysis while still detecting
    genuine file changes after the file stabilizes.

FUTURE OPTIMIZATION:

    If stat() overhead becomes significant for immutable files (older files that
    won't change), add mutable/immutable distinction:
    - Immutable: File where a newer file exists for the same job (skip stat)
    - Mutable: Latest file for a job (keep stat validation)
    
    Currently not implemented as stat() cost is negligible.

LIFECYCLE:

    1. ANALYSIS COMPLETES:
       - Add to ledger with file (mtime, size)
       - Entry starts grace period countdown

    2. LEDGER HIT (within grace period):
       - Return cached result immediately
       - No file validation (absorbs straggling writes)

    3. LEDGER HIT (after grace period):
       - stat() file to get current (mtime, size)
       - If unchanged: return cached result (file already processed)
       - If changed: invalidate entry, re-analyze

    4. CLEANUP (every 10 min):
       - Evict entries where file.mtime > 14 days (safeguard)

    5. SHUTDOWN:
       - Save ledger to CACHE_FILE (JSON format)
       - Includes (mtime, size) per entry

    6. STARTUP (if CACHE_FILE exists):
       - Skip if file.mtime > 14 days
       - Skip if file (mtime, size) changed or file gone
       - Valid entries restored (treated as freshly cached for grace period)

STATS (in /stats response):

    "cache": {
        "hits": 100,          // Ledger hits (file already processed)
        "misses": 25,         // Ledger misses (new file or invalidated)
        "invalidated": 5      // Entries invalidated (file changed after grace period)
    },
    "cleanup": {
        "cache_expired": 2    // Entries evicted (file.mtime > 14 days)
    },
    "persistence": {
        "imported": 15,                 // Entries restored from disk
        "import_skipped_changed": 3,    // Entries with file changes or gone
        "import_skipped_old_file": 2    // Entries with file.mtime > 14 days
    }

LIMITATIONS:

    1. SAVE ONLY ON SHUTDOWN:
       - Ledger is persisted only during graceful shutdown (SIGTERM)
       - If service is killed (SIGKILL) or crashes, in-memory entries are lost
       - Lost entries will be re-analyzed on next request (duplicate LLM calls)

    2. CORRUPTION HANDLING:
       - Corrupt cache file is logged and ignored (service starts with empty ledger)
       - Atomic writes (temp file + rename) prevent partial corruption

    MITIGATIONS:

    - Use systemd or process manager that sends SIGTERM before SIGKILL
    - Set adequate stop timeout (e.g., TimeoutStopSec=30 in systemd)
    - Monitor service stability to reduce unexpected crashes
    - Results are also in Elasticsearch (ledger is optimization, not source of truth)

    FUTURE: If needed, implement periodic saves (e.g., every 5-10 min) to reduce
    data loss window. Not currently implemented as graceful shutdown is typical.

================================================================================

4. STARTUP / SHUTDOWN
--------------------------------------------------------------------------------

4.1 STARTUP SEQUENCE
....................

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Step │ Action                     │ On Failure                          │
    ├──────┼────────────────────────────┼─────────────────────────────────────┤
    │  1   │ Load configuration         │ Fail startup                        │
    │      │ └─ Validate ALLOWED_ROOT   │ (required, must exist)              │
    │      │ └─ Validate NVIDIA_API_KEY │ (required, non-empty)               │
    ├──────┼────────────────────────────┼─────────────────────────────────────┤
    │  2   │ Initialize AttributionSvc  │ -                                   │
    │      │ └─ Create _jobs dict       │                                     │
    │      │ └─ Create _in_flight dict  │                                     │
    │      │ └─ Create threading.Lock   │                                     │
    │      │ └─ Initialize counters     │                                     │
    ├──────┼────────────────────────────┼─────────────────────────────────────┤
    │  3   │ Start background poll      │ -                                   │
    │      │ └─ daemon=True             │                                     │
    ├──────┼────────────────────────────┼─────────────────────────────────────┤
    │  4   │ Create FastAPI app         │ -                                   │
    │      │ └─ Register handlers       │                                     │
    │      │ └─ Register rate limiter   │                                     │
    ├──────┼────────────────────────────┼─────────────────────────────────────┤
    │  5   │ Load persisted cache       │ Log warning, continue               │
    │      │ (if CACHE_FILE configured) │ (cache starts empty)                │
    ├──────┼────────────────────────────┼─────────────────────────────────────┤
    │  6   │ Start uvicorn server       │ -                                   │
    │      │ └─ Listen on HOST:PORT     │                                     │
    └──────┴────────────────────────────┴─────────────────────────────────────┘

    Order matters:
        - Service initialized before app created (app.state.service)
        - Poll thread can start immediately (handles empty _jobs)
        - Cache loaded in startup hook (before serving requests)

4.2 SHUTDOWN SEQUENCE
.....................

GRACEFUL (SIGTERM, SIGINT, Ctrl+C):

    Signal received (SIGTERM, SIGINT, Ctrl+C):

    1. Uvicorn stops accepting new connections
        - timeout_graceful_shutdown=30s to drain in-flight HTTP requests
        - Existing requests continue until complete or timeout

    2. In-flight analysis requests:
        - Requests waiting on _in_flight futures will complete
        - No new analyses started after shutdown begins
        - If timeout exceeded, requests may be interrupted

    3. Stop background poll thread
        - AttributionService.shutdown() called via lifespan hook
        - Sets stop flag on poller
        - Thread exits on next loop iteration (up to POLL_INTERVAL_SECONDS delay)
        - Since daemon=True, thread also killed when process exits

    4. Save cache (if CACHE_FILE configured):
        - Exports cache entries to JSON file
        - Includes mtime, size, immutable flag per entry
        - Saved atomically (write to temp, then rename)

    5. State persistence:
        - _jobs dict is lost (job tracking not persisted)
        - Jobs must be re-submitted after restart
        - Analysis results in cache ARE persisted (if CACHE_FILE set)
        - Immutable entries restored on next startup

    Lifespan hook (handles cache persistence):
        @app.on_event("startup")
        async def startup_event():
            if cfg.CACHE_FILE:
                app.state.service.load_cache(cfg.CACHE_FILE)

        @app.on_event("shutdown")
        async def shutdown_event():
            if app.state.cache_file:
                app.state.service.save_cache(app.state.cache_file)

UNGRACEFUL SHUTDOWN (SIGKILL, OOM, crash):

    - Process terminates immediately
    - In-flight requests interrupted mid-analysis
    - Background thread killed
    - State lost (same as graceful)
    - Clients may see connection reset errors

================================================================================

5. DATA STRUCTURES
--------------------------------------------------------------------------------

5.1 CORE DATA STRUCTURES
.........................

_jobs: Dict[jobOutputLogFile, Job]
    Job:
        # Common fields (all modes)
        - path: str              # Client's original path (job output, used as key)
        - mode: JobMode          # PENDING | SINGLE | SPLITLOG
        - user: str              # From POST, or "unknown" if GET-without-POST
        - job_id: str | None     # From POST, or extracted from path if GET-without-POST
        - created_at: float      # time.monotonic() at job creation
        - sched_restarts: int    # Scheduler restart count (all modes)
                                 # (count of << START PATHS >> markers in job output)
        - terminated: bool       # Job has been terminated via GET (all modes)
        - terminated_at: float | None  # When terminated (for cleanup TTL)
        - file_info: Dict[str, FileInfo]  # Per-file tracking, keyed by filename
        
        # Splitlog-specific fields (only populated when mode=SPLITLOG)
        - logs_dir: str | None           # Path to LOGS_DIR from markers
        - last_poll_at: float | None     # When job was last polled (for observability)
                                         # Poll checks: job output for markers + LOGS_DIR for files

    FileInfo (tracks each analyzed file - ALL modes):
        See Section 5.4 for dataclass definition.

    MODE DIFFERENCES:

        | Aspect          | Single Mode                    | Splitlog Mode                   |
        |-----------------|--------------------------------|---------------------------------|
        | file_info size  | 1 entry (job.path)             | N entries (discovered files)    |
        | log_file value  | job.path (job output itself)   | Files discovered in LOGS_DIR   |
        | logs_dir        | None                           | Path from LOGS_DIR marker       |

    TERMINOLOGY: See Section 1.2 TERMINOLOGY for full definitions of sched_restarts
    and wl_restarts.

_in_flight: Dict[path, InFlightEntry]
    InFlightEntry:
        - future: asyncio.Future[Dict]  # Shared future for waiters
        - started_at: float             # time.monotonic() when analysis started
    
    Key semantics:
        - SINGLE mode: key = job output path (jobOutputLogFile)
        - SPLITLOG mode: key = log file path in LOGS_DIR

_lock: threading.Lock
    - Protects concurrent access to _jobs and _in_flight

CACHING ARCHITECTURE:

    The service uses TWO caching mechanisms:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. REQUEST COALESCER (_in_flight)                                       │
    │    Prevents duplicate LLM calls for concurrent GETs                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Purpose:  Deduplicate concurrent GET requests for same file             │
    │ Lifetime: Only while analysis is in-flight (seconds to minutes)         │
    │ Storage:  _in_flight dict with asyncio.Future                           │
    │ Key:      log_file path                                                 │
    │ Cleanup:  Removed immediately after analysis completes                  │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2. RESULT CACHE (file_info)                                             │
    │    Persists results beyond request lifecycle, aligns with Job lifecycle │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Purpose:  Store completed analysis results for subsequent GETs          │
    │ Lifetime: Until job is cleaned up (TTL_TERMINATED_SECONDS after GET)    │
    │ Storage:  job.file_info[filename].wl_restart_results                    │
    │ Key:      filename (log_file basename)                                  │
    │ Cleanup:  When job is removed from _jobs                                │
    └─────────────────────────────────────────────────────────────────────────┘

    LOOKUP FLOW:
        GET arrives
         ├─→ Check file_info → if complete → return cached result
         └─→ if not complete
              ├─→ Check _in_flight → if exists → await existing future
              └─→ if not exists
                   ├─→ Create _in_flight entry
                   ├─→ Start LLM analysis
                   ├─→ On complete: store in file_info
                   └─→ Remove from _in_flight

    COORDINATION BETWEEN POLL AND GET:

        Both poll thread and GET can trigger analysis. Use file_info.analysis_triggered
        as a shared coordination flag to prevent duplicates:

        | Context     | Before Analysis                  | After Analysis              |
        |-------------|----------------------------------|------------------------------|
        | GET         | Check analysis_triggered         | Store in file_info           |
        |             | If false → set true, start       | Set analysis_complete=True   |
        |             | If true → wait on _in_flight     |                              |
        | Poll        | Check analysis_triggered         | Store in file_info           |
        |             | If false → set true, start       | Set analysis_complete=True   |
        |             | If true → skip (GET handles it)  |                              |

        WHY POLL DOESN'T USE _in_flight:
            - _in_flight uses asyncio.Future (not thread-safe from poll thread)
            - Poll thread runs in sync context, can't await futures
            - Poll thread creates its own event loop for analysis

        RESULT: No duplicate analysis if coordination via analysis_triggered is correct.
        External (API) and internal (poll) requests use same file_info storage.

5.2 DATACLASS DEFINITIONS
..........................

In job.py:

    from dataclasses import dataclass, field
    from typing import Dict, List, Optional
    import time

    @dataclass(slots=True)
    class FileInfo:
        """Tracks a single log file and its analysis results.
        Uses slots=True for memory efficiency with many instances."""
        log_file: str                            # Absolute path to log file
        analysis_triggered: bool = False         # Analysis started
        analysis_complete: bool = False          # Analysis finished
        analyzed_at: Optional[float] = None      # time.time() when analysis completed
        wl_restarts: int = 0                     # Count of Cycle: N markers
        wl_restart_results: List[Dict] = field(default_factory=list)
        
        # NOTE: job.file_info dict is keyed by FILENAME (basename), not full path.
        # Example: job.file_info["model_12345_cycle0.log"] = FileInfo(log_file="/logs/job/model_12345_cycle0.log", ...)

    @dataclass(slots=True)
    class Job:
        """Unified job model for all modes.
        Uses slots=True for memory efficiency (MAX_JOBS can be 100,000+)."""
        # Required fields
        path: str                                # Job output path (dict key)
        user: str                                # Job owner
        mode: str                                # "pending" | "single" | "splitlog"
        
        # Timestamps
        created_at: float = field(default_factory=time.monotonic)
        
        # Optional identification
        job_id: Optional[str] = None             # From POST (enables splitlog)
        
        # Scheduler restart tracking (all modes)
        sched_restarts: int = 0                  # Count of << START PATHS >>
        
        # Splitlog-specific
        logs_dir: Optional[str] = None           # Path from LOGS_DIR marker
        last_poll_at: Optional[float] = None     # Last poll timestamp
        
        # File tracking (all modes), keyed by filename
        file_info: Dict[str, FileInfo] = field(default_factory=dict)
        
        # Termination (all modes)
        terminated: bool = False
        terminated_at: Optional[float] = None

        def mark_terminated(self) -> None:
            if not self.terminated:
                self.terminated = True
                self.terminated_at = time.monotonic()

    @dataclass
    class InFlightEntry:
        """Tracks an in-progress analysis."""
        future: asyncio.Future                   # Shared future for waiters
        started_at: float                        # time.monotonic()

5.3 DATA STRUCTURE RELATIONSHIPS
.................................

    SINGLE MODE (file_info has 1 entry):
        Job
        ├── sched_restarts: 2           # From << START PATHS >> count
        ├── logs_dir: None              # No separate log directory
        └── file_info:
            └── "slurm-12345.out": FileInfo
                ├── log_file: "/logs/slurm-12345.out"  # Same as job.path
                ├── analysis_complete: True
                ├── wl_restarts: 3
                └── wl_restart_results: [result0, result1, result2]

    SPLITLOG MODE (file_info has N entries):
        Job
        ├── sched_restarts: 2           # From << START PATHS >> count
        ├── logs_dir: "/logs/job_123"   # Separate log directory
        └── file_info:
            ├── "model_123_time_01.log": FileInfo
            │   ├── log_file: "/logs/job_123/model_123_time_01.log"
            │   ├── analysis_complete: True
            │   ├── wl_restarts: 3
            │   └── wl_restart_results: [result0, result1, result2]
            └── "model_123_time_02.log": FileInfo
                ├── log_file: "/logs/job_123/model_123_time_02.log"
                ├── analysis_complete: False
                └── wl_restarts: 0      # Not yet analyzed

5.4 ANALYSIS RESULT FORMAT
...........................

    Single LLM analysis result (per log file):
    {
        "cycles": [
            {"cycle": 0, "analysis": {...}},
            {"cycle": 1, "analysis": {...}},
            {"cycle": 2, "analysis": {...}}
        ],
        "file_path": "/path/to/log/file",
        "analyzed_at": 1234567890.123
    }

    When no Cycle: N markers found:
    {
        "cycles": [
            {"cycle": 0, "analysis": {...}}
        ],
        "file_path": "/path/to/log/file",
        "analyzed_at": 1234567890.123
    }

    MAPPING TO FILE_INFO:
        file_info[filename].wl_restarts = len(result["cycles"])
        file_info[filename].wl_restart_results = result["cycles"]

5.5 FILEINFO LIFECYCLE
.......................

    SINGLE MODE:
        Created:  On first GET (after file validation passes)
        Content:  file_info[basename(job.path)] = FileInfo(log_file=job.path, ...)
        Analysis: Triggered on first GET, results stored in file_info

    SPLITLOG MODE:
        Created:  When background poll discovers file in LOGS_DIR
        Content:  file_info[filename] = FileInfo(log_file=discovered_path, ...)
        Analysis: Triggered when next file appears OR job terminated via GET

    STATE TRANSITIONS:
        FileInfo created → analysis_triggered=False, analysis_complete=False, analyzed_at=None
        Analysis starts  → analysis_triggered=True
        Analysis ends    → analysis_complete=True, analyzed_at=time.time(), wl_restart_results populated

5.6 COMMON GOTCHAS
...................

    1. GET-without-POST creates job with user="unknown"
       - job_id extracted from path if possible (best effort)
       - Mode detected from file content (markers checked)

    2. job_id from first POST wins
       - Subsequent POSTs with different job_id log WARNING
       - job.job_id is NOT updated (prevents splitlog confusion)

    3. Single mode: file_info has single entry keyed by filename (basename of job.path)
       - Same file, just stored in unified structure
       - Only 1 entry in file_info dict

    4. Terminated jobs still return cached results
       - Until TTL_TERMINATED_SECONDS expires (1 hour)
       - Then job removed, GET returns NOT_FOUND

    5. Analysis errors don't prevent result caching
       - Timeout/error stored as result with state field
       - Retry requires waiting for TTL_TERMINATED_SECONDS

================================================================================
                               HTTP INTERFACE
================================================================================

6. HTTP API
--------------------------------------------------------------------------------

6.1 ENDPOINTS
..............

Authentication: None (open endpoints)
    - All endpoints are unauthenticated
    - Rely on network-level security (firewall, VPC, etc.)
    - Rate limiting provides basic abuse protection

GET /healthz
    Health check endpoint for load balancers and monitoring.
    
    Success response (200):
        {
            "status": HealthStatus (OK | DEGRADED | FAIL),
            "issues": ["elevated compute error rate: 25%", ...]
        }
    
    CALCULATION:
    
        # LLM/compute error rate
        if total_computes > 0:
            compute_error_rate = (compute_errors + compute_timeouts) / total_computes
        else:
            compute_error_rate = 0  # No requests yet = healthy
        
        # Dataflow error rate (only if configured)
        if total_dataflow_posts > 0:
            dataflow_error_rate = failed_posts / total_posts
        else:
            dataflow_error_rate = 0  # No posts = healthy
    
    STATUS DETERMINATION:
    
        | Condition                  | Status   | Issue message                       |
        |----------------------------|----------|-------------------------------------|
        | error_rate < 0.20          | OK       | (no issue)                          |
        | 0.20 <= error_rate < 0.50  | DEGRADED | "elevated compute error rate: N%"   |
        | error_rate >= 0.50         | FAIL     | "high compute error rate: N%"       |
    
        Final status = FAIL if any "high" issue, else DEGRADED if any issue, else OK
    
    EDGE CASES:
        - No requests yet (total_computes = 0): Returns OK (no denominator issue)
        - All requests succeeded: Returns OK with empty issues
        - Mixed issues: Returns worst status (FAIL > DEGRADED > OK)
    
    LIMITATION:
        - Uses cumulative counters, not sliding window
        - After long healthy period, sudden LLM failure takes many errors
          before rate crosses threshold (slow to detect)
        - Future improvement: sliding window or consecutive failure detection

GET /stats
    Returns all counters and gauges (see section 20).
    
    Success response (200):
        {
            "jobs_tracked": 42,
            "pending_jobs": 3,
            "single_jobs": 25,
            "splitlog_jobs": 14,
            "analyses_completed": 120,
            "coalescer_cache_size": 38,
            "inflight_requests": 2
        }

GET /jobs
    Dump tracked jobs (for debugging/introspection) with pagination and filtering.
    
    Query params:
        mode: JobMode (optional)    # Filter: PENDING, SINGLE, SPLITLOG
        limit: int (default=100)    # Max jobs to return (max: 1000)
        offset: int (default=0)     # Pagination offset
    
    Success response (200):
        {
            "total_counts": {
                "pending": N,   # JobMode.PENDING count
                "single": N,    # JobMode.SINGLE count
                "splitlog": N   # JobMode.SPLITLOG count
            },
            "filter": { "mode": "...", "limit": N, "offset": N },
            "returned": N,
            "jobs": [...]
        }
    
    Job fields: path, mode, user, job_id, created_at, terminated, sched_restarts,
                 files_count (len of file_info), files_analyzed (with analysis_complete=True)
    
    Note: Full file_info and results available via GET /logs?log_path=...
    
    Examples:
        GET /jobs                        # First 100 jobs (all modes), with total counts
        GET /jobs?mode=pending           # First 100 pending jobs (value is string, not enum)
        GET /jobs?limit=50&offset=100    # Jobs 100-149

GET /print
    Preview first 4KB of a log file.
    
    Query params:
        log_path: string  # Required: path to file under ALLOWED_ROOT
    
    Success response (200): text/plain
        <first 4KB of file content>
    
    Error response (4xx):
        { "error_code": string, "message": string }

GET /inflight
    Dump all in-flight requests (for debugging/introspection).
    
    Success response (200):
        {
            "count": N,
            "entries": [
                { "path": "...", "started_at": timestamp, "elapsed_seconds": N }
            ]
        }

POST /logs
    Request body:
        {
            "log_path": string,      # Required: path to job output file
            "user": string,          # Required: job owner
            "job_id": string | null  # Optional: enables splitlog detection
        }
    
    Success response (200):
        { "mode": JobMode (SINGLE | PENDING | SPLITLOG) }
    
    Error response (4xx):
        { "error_code": string, "message": string }

GET /logs
    Query params:
        log_path: string       # Required: path to job output file
        file: string (optional)  # Filename (for splitlog or single)
        wl_restart: int (optional)  # Workload restart index within file
        all_files: bool (optional)  # Return all completed files (default: false)
    
    Success response (200) - All files:
        {
            "mode": "SINGLE" | "SPLITLOG",
            "sched_restarts": N,
            "files": [                         // Sorted chronologically
                {
                    "log_file": "model_12345_cycle0.log",
                    "analysis_complete": true,
                    "wl_restarts": N,
                    "wl_restart_results": [...]
                },
                ...
            ]
        }
    
    Success response (200) - Specific file (?file=model_12345_cycle0.log):
        {
            "log_file": "model_12345_cycle0.log",
            "analysis_complete": true,
            "wl_restarts": N,
            "wl_restart_results": [...]
        }
    
    Success response (200) - Specific wl_restart (?file=model_12345_cycle0.log&wl_restart=2):
        {
            "log_file": "model_12345_cycle0.log",
            "wl_restart": 2,
            "result": {...}
        }
    
    Error response (4xx):
        { "error_code": string, "message": string }

6.2 PYDANTIC REQUEST/RESPONSE MODELS
.....................................

In app.py (or separate models module):

    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any

    # Request models
    class PostLogsRequest(BaseModel):
        log_path: str = Field(..., description="Absolute path to job output file")
        user: str = Field(..., description="Job owner")
        job_id: Optional[str] = Field(None, description="Job ID for splitlog detection")

    # Response models
    class PostLogsResponse(BaseModel):
        mode: str  # "pending" | "single" | "splitlog"

    class FileInfoResponse(BaseModel):
        log_file: str                # Filename (stable identifier, use in ?file= param)
        analysis_complete: bool
        analyzed_at: Optional[float]  # time.time() when analysis completed
        wl_restarts: int
        wl_restart_results: List[Dict[str, Any]]

    class GetLogsResponse(BaseModel):
        mode: str
        sched_restarts: int
        files: List[FileInfoResponse]  # Sorted chronologically for display

    # NOTE: files[] is sorted chronologically. Use log_file (filename) in API calls:
    # GET /logs?log_path=...&file=model_12345_cycle0.log

    class ErrorResponse(BaseModel):
        error_code: str
        message: str

    class HealthResponse(BaseModel):
        status: str  # "ok" | "degraded" | "fail"
        issues: List[str]

    class StatsResponse(BaseModel):
        jobs_tracked: int
        pending_jobs: int
        single_jobs: int
        splitlog_jobs: int
        analyses_completed: int
        coalescer_cache_size: int
        inflight_requests: int
        # Poll thread observability
        poll_last_run_at: Optional[float]   # Timestamp of last poll (epoch)
        poll_duration_seconds: Optional[float]  # Duration of last poll
        poll_is_running: bool               # True if poll in progress

    class InflightResponse(BaseModel):
        count: int
        entries: List[Dict[str, Any]]  # [{"log_file": str, "started_at": float}, ...]

6.3 CURL EXAMPLES
..................

# Health check
curl http://localhost:8000/healthz

# Health check (pretty-printed)
curl "http://localhost:8000/healthz?pretty=true"

# Get all stats
curl http://localhost:8000/stats

# List jobs (first 100)
curl http://localhost:8000/jobs

# List pending jobs only
curl "http://localhost:8000/jobs?mode=pending"

# List jobs with pagination
curl "http://localhost:8000/jobs?limit=50&offset=100"

# Preview log file (first 4KB)
curl "http://localhost:8000/print?log_path=/data/logs/slurm-12345.out"

# List in-flight requests
curl http://localhost:8000/inflight

# Submit job for tracking (without job_id)
curl -X POST http://localhost:8000/logs \
  -H "Content-Type: application/json" \
  -d '{"log_path": "/data/logs/slurm-12345.out", "user": "alice"}'

# Submit job for tracking (with job_id for splitlog detection)
curl -X POST http://localhost:8000/logs \
  -H "Content-Type: application/json" \
  -d '{"log_path": "/data/logs/slurm-12345.out", "user": "alice", "job_id": "12345"}'

# Get analysis result (triggers LLM if not cached)
curl "http://localhost:8000/logs?log_path=/data/logs/slurm-12345.out"

# Get analysis result (with jq for pretty output)
curl -s "http://localhost:8000/logs?log_path=/data/logs/slurm-12345.out" | jq .

================================================================================

7. ERROR RESPONSES
--------------------------------------------------------------------------------

| Error                | HTTP | Code                  | When                          |
|----------------------|------|-----------------------|-------------------------------|
| Path not absolute    | 400  | INVALID_PATH          | Path doesn't start with /     |
| Outside allowed root | 403  | OUTSIDE_ROOT          | Path outside ALLOWED_ROOT     |
| File not found       | 404  | NOT_FOUND             | File doesn't exist at GET     |
| Not readable         | 403  | NOT_READABLE          | File exists but permission denied |
| Not regular file     | 400  | NOT_REGULAR           | Path is directory, device, etc. |
| Empty file           | 400  | EMPTY_FILE            | File is empty (GET only)      |
| LOGS_DIR not readable| 403  | LOGS_DIR_NOT_READABLE | LOGS_DIR permission denied    |
| Job limit reached    | 503  | JOB_LIMIT_REACHED     | MAX_JOBS exceeded, try later  |
| Internal error       | 500  | INTERNAL_ERROR        | Unexpected server error       |

Note: 
- Analysis timeout/error returns error response (AnalyzerError with appropriate ErrorCode)
- Invalid ?file= parameter (filename not in file_info) returns 404 NOT_FOUND
- Empty files are accepted for POST (pending mode), rejected only on GET

ERROR HANDLING APPROACH:

    Success: Returns AnalysisResult or SplitlogAnalysisResult
        - result: Dict containing LLM analysis output
        - status: "completed"
    
    Failure: Returns AnalyzerError (mapped to HTTP status codes at API boundary)
        - error_code: ErrorCode enum value (INTERNAL_ERROR, NOT_FOUND, etc.)
        - message: Human-readable error description
    
    HTTP layer maps ErrorCode to status codes (see table above)

================================================================================
                             VALIDATION & FLOWS
================================================================================

8. PATH VALIDATION
--------------------------------------------------------------------------------

8.1 VALIDATION STEPS
....................

    1. Check path is absolute (starts with /)
    2. Resolve to real path: os.path.realpath(user_path)
       - Resolves symlinks to final target
       - Normalizes . and .. components
       - Returns absolute path
    3. Check resolved path is under ALLOWED_ROOT:
       - os.path.commonpath([real_path, allowed_root]) == allowed_root

    Key: Client path used as job key (not normalized), but file ops use resolved path

8.2 EDGE CASES
..............

    | Case                  | Behavior                                          |
    |-----------------------|---------------------------------------------------|
    | Symlink inside root   | ALLOWED (resolved path checked under ALLOWED_ROOT)|
    | Symlink outside root  | REJECTED (OUTSIDE_ROOT, 403)                      |
    | Symlink chain         | All links resolved, final target checked          |
    | Broken symlink        | NOT_FOUND (404) at GET                            |
    | Non-regular file      | REJECTED (NOT_REGULAR, 400) - directories, etc.   |
    | Path traversal (../)  | realpath() normalizes, then root checked          |
    | Trailing slashes      | realpath normalizes, file check handles           |
    | Case sensitivity      | Follows filesystem (case-sensitive on Linux)      |
    | Unicode/special chars | Passed through to filesystem                      |
    | Null bytes            | ValueError → INVALID_PATH (400)                   |
    | TOCTOU race           | Acceptable, error returned to client              |
    | Network FS delays     | Handled by pending mode and background poll       |

    EMPTY/SMALL FILE HANDLING:
        | Context | File State                    | Behavior                          |
        |---------|-------------------------------|-----------------------------------|
        | POST    | Empty or < MIN_FILE_SIZE_KB   | PENDING mode (poll will re-check) |
        | GET     | Empty                         | REJECTED (EMPTY_FILE, 400)        |
        | GET     | < MIN_FILE_SIZE_KB            | Analyze anyway (file may be valid)|

8.3 ALLOWED_ROOT VALIDATION
...........................

    At startup:
    - Must be absolute path
    - Must exist and be a directory
    - Must be readable and traversable (R_OK | X_OK)

8.4 PATH VALIDATION EXCEPTION HANDLING
......................................

    Map filesystem exceptions to ErrorCode in path validation:

    | Exception/Check            | ErrorCode      | HTTP | Notes                        |
    |----------------------------|----------------|------|------------------------------|
    | ValueError (null bytes)    | INVALID_PATH   | 400  | os.path.realpath raises this |
    | FileNotFoundError          | NOT_FOUND      | 404  | File doesn't exist           |
    | PermissionError            | NOT_READABLE   | 403  | Permission denied            |
    | stat.S_ISREG() = False     | NOT_REGULAR    | 400  | Not a regular file           |
    | st.st_size == 0 (GET only) | EMPTY_FILE     | 400  | Empty file at GET            |
    | OSError (other)            | INTERNAL_ERROR | 500  | Unexpected filesystem error  |

================================================================================

9. POST FLOW
--------------------------------------------------------------------------------

9.1 REQUEST PROCESSING
......................

POST /logs (path, user, job_id?):

    1. VALIDATE PATH
        - Use client's path as job key (not normalized)
        - Validate internally: normalize for file ops, check under ALLOWED_ROOT

    2. CHECK EXISTING JOB
        - If _jobs[path] exists:
            - Update job.user = user
            - If job_id provided AND differs from job.job_id → log WARNING
            - job.job_id is NOT updated (first POST wins)
            - job.file_info is preserved (existing results kept)
            - Return: {mode: job.mode}

    3. CHECK CAPACITY
        - If len(_jobs) >= MAX_JOBS:
            - Reject with ErrorCode.JOB_LIMIT_REACHED (503)

    4. CHECK FILE STATE
        - If file not found:
            - Create Job(mode=JobMode.PENDING, ...)
            - Return: {mode: JobMode.PENDING}
            - Background poll will check (section 11)
        
        - If file exists but size < MIN_FILE_SIZE_KB:
            - Create Job(mode=JobMode.PENDING, ...)
            - Return: {mode: JobMode.PENDING}
            - Background poll will re-check (section 11)

    5. CLASSIFY MODE (file exists + readable + size >= MIN_FILE_SIZE_KB)
        - If job_id NOT provided:
            - Check for MARKER_START_PATHS (optional, for warning)
            - If LOGS_DIR found → log WARNING "LOGS_DIR found but no job_id, using SINGLE mode"
            - Create Job(mode=JobMode.SINGLE, ...)
            - Return: {mode: JobMode.SINGLE}
            - Note: Can't use SPLITLOG without job_id (file pattern requires it)
        
        - If job_id provided:
            - Check for MARKER_START_PATHS in first 4KB of file
            - If markers found AND LOGS_DIR readable:
                - Create Job(mode=JobMode.SPLITLOG, ...)
                - Return: {mode: JobMode.SPLITLOG}
            - If markers found AND LOGS_DIR not readable:
                - Reject with ErrorCode.LOGS_DIR_NOT_READABLE
            - If no markers found:
                - Create Job(mode=JobMode.SINGLE, ...)
                - Return: {mode: JobMode.SINGLE}

9.2 MODE SELECTION LOGIC
........................

    | job_id | File State | Markers | LOGS_DIR | Result        |
    |--------|------------|---------|----------|---------------|
    | No     | Any        | None    | -        | SINGLE        |
    | No     | Any        | Found   | Any      | SINGLE + WARN |
    | Yes    | Not ready  | -       | -        | PENDING       |
    | Yes    | Ready      | None    | -        | SINGLE        |
    | Yes    | Ready      | Found   | Readable | SPLITLOG      |
    | Yes    | Ready      | Found   | Not OK   | ERROR         |

    Note: "SINGLE + WARN" logs warning "LOGS_DIR found but no job_id provided"

9.3 SEQUENCE DIAGRAM
....................

    Client                Service                    FileSystem
      |                      |                           |
      |-- POST /logs ------->|                           |
      |   {path, user, id}   |                           |
      |                      |-- validate path --------->|
      |                      |<-- ok/error --------------|
      |                      |                           |
      |                      |-- check _jobs[path] ----->|
      |                      |   (exists? return mode)   |
      |                      |                           |
      |                      |-- check MAX_JOBS -------->|
      |                      |   (limit? reject 503)     |
      |                      |                           |
      |                      |-- check file exists? ---->|
      |                      |<-- yes/no ----------------|
      |                      |                           |
      |                      |-- if yes: read lines ---->|
      |                      |<-- content ---------------|
      |                      |                           |
      |                      |-- check markers --------->|
      |                      |   (SPLITLOG/SINGLE/PENDING)|
      |                      |                           |
      |                      |-- create Job ------------>|
      |<-- {mode} -----------|                           |

================================================================================

10. GET FLOW
--------------------------------------------------------------------------------

10.1 REQUEST PROCESSING
.......................

GET /logs (jobOutputLogFile, file?, wl_restart?):

    1. Validate jobOutputLogFile:
        - Must be under ALLOWED_ROOT
        - Must exist, be readable, and be a regular file
        - Must not be empty (reject with EMPTY_FILE)
        - If not found or unreadable → reject with error

    2. Lookup job by jobOutputLogFile:
        - If no job exists (GET-without-POST):
          - Check for MARKER_START_PATHS in file content
          - Extract job_id from path using JOB_ID_PATTERNS (best effort)
          - If markers + LOGS_DIR found → Create Job(mode=JobMode.SPLITLOG, ...)
          - If not found → Create Job(mode=JobMode.SINGLE, user="unknown", ...)

    3. If job.mode == JobMode.PENDING:
        - File exists and is readable (validated in step 1)
        - Final check for MARKER_START_PATHS in content
        - If markers + LOGS_DIR found → promote to JobMode.SPLITLOG
        - If not found → set job.mode = JobMode.SINGLE

10.2 MODE-SPECIFIC HANDLING
...........................

    SINGLE MODE (job.mode == SINGLE):
        - Initialize file_info[filename] if not exists:
            file_info[basename(job.path)] = FileInfo(log_file=job.path, ...)
        - If file_info[filename].analysis_complete:
            → Return cached results from file_info[filename].wl_restart_results
        - If in _in_flight[job.path]:
            → Wait on future, then return results
        - Else: Trigger analysis (see 10.3)

    SPLITLOG MODE (job.mode == SPLITLOG):
        - See Section 17 SPLITLOG GET FLOW for detailed handling
        - Summary: Mark terminated, trigger final file analysis, return results
        - Query params: ?file=, ?wl_restart=, ?all_files= (see Section 17.2)

10.3 ANALYSIS TRIGGER
.....................

    When analysis needed for target file:
        - Add to _in_flight[target_file_path]
        - file_info[filename].analysis_triggered = True
        - Run LLM analysis (see section 12)
        - On success: 
            file_info[filename].wl_restart_results = parsed_results
            file_info[filename].wl_restarts = len(parsed_results)
            file_info[filename].analysis_complete = True
            file_info[filename].analyzed_at = time.time()
        - On timeout/error: 
            file_info[filename].wl_restart_results = [{state: "error/timeout", ...}]
            file_info[filename].analysis_complete = True
            file_info[filename].analyzed_at = time.time()
        - Remove from _in_flight
        - Mark job.terminated = True, job.terminated_at = now

    Return: {files: [...], mode: job.mode, ...}

10.4 REQUEST COALESCING
.......................

    | Scenario          | Behavior                                     |
    |-------------------|----------------------------------------------|
    | First GET         | Triggers analysis, creates _in_flight entry  |
    | Subsequent GETs   | Wait on shared future                        |
    | All GETs          | Receive same result from file_info           |
    | Client disconnect | Analysis continues, result cached            |

10.5 SEQUENCE DIAGRAM
.....................

    Client1              Client2              Service                LLM
      |                    |                    |                     |
      |-- GET /logs ------>|                    |                     |
      |                    |                    |                     |
      |                    |                    |-- lookup job ------>|
      |                    |                    |   (no file_info)    |
      |                    |                    |                     |
      |                    |                    |-- add to _in_flight |
      |                    |                    |                     |
      |                    |-- GET /logs ------>|                     |
      |                    |                    |-- check _in_flight  |
      |                    |                    |   (exists! wait)    |
      |                    |                    |                     |
      |                    |                    |-- analyze --------->|
      |                    |                    |<-- result ----------|
      |                    |                    |                     |
      |                    |                    |-- store in file_info|
      |                    |                    |-- notify waiters -->|
      |                    |                    |                     |
      |<-- {files:[...]} --|<-- {files:[...]} --|                     |
      |                    |                    |                     |
    (Both clients receive same result - single LLM call)

================================================================================

11. BACKGROUND POLL
--------------------------------------------------------------------------------

Runs in separate thread (daemon=True).

Loop:
    1. Sleep(POLL_INTERVAL)
       (Note: sleep first means POLL_INTERVAL delay before first poll on startup)
    
    2. For each job with mode=JobMode.PENDING:
        a. Check if file exists and readable
        b. If not exists or not readable:
           - Keep polling (network FS delay)
        c. If file size < MIN_FILE_SIZE_KB:
           - Keep polling (file still being written)
        d. If file size >= MIN_FILE_SIZE_KB:
           - Classify mode (check MARKER_START_PATHS)
           - Update job.mode accordingly
    
    3. Run cleanup (see section 18)
    
    4. If file never appears/readable after TTL_PENDING_SECONDS → job removed by cleanup

================================================================================
                                  PROCESSING
================================================================================

12. LLM ANALYSIS
--------------------------------------------------------------------------------

Function: _run_llm_analysis(jobOutputLogFile) → Dict

12.1 ARCHITECTURE
.................

Uses MCP (Model Context Protocol) to communicate with log_analyzer module:

    ┌─────────────┐      MCP       ┌─────────────────┐      API      ┌─────────────┐
    │  attrsvc    │ ──────────────►│  log_analyzer   │ ─────────────►│  NVIDIA LLM │
    │  (client)   │                │  (MCP server)   │               │  Endpoint   │
    └─────────────┘                └─────────────────┘               └─────────────┘

    Connection type: STDIO (subprocess)
        - MCP server runs as child process of attrsvc
        - Communication via stdin/stdout pipes
        - No network configuration required
        - Server launched on-demand per analysis request

Benefits:
    - Separation of concerns (attrsvc doesn't contain LLM logic)
    - log_analyzer can be updated independently
    - Reusable across different access methods (HTTP, CLI, etc.)
    - No external server process to manage

12.2 IMPLEMENTATION
...................

    from nvidia_resiliency_ext.attribution.mcp_integration import create_mcp_client

    async def _run_llm_analysis(self, path: str, user: str) -> Dict:
        client = create_mcp_client()
        async with client:
            result = await client.run_module(
                module_name="log_analyzer",
                log_path=path,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p,
                max_tokens=self.config.llm_max_tokens,
            )
            return result

LLM defaults (in library AnalyzerConfig):
    llm_model = "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1"
    llm_temperature = 0.0   # Deterministic output
    llm_top_p = 1.0         # No nucleus sampling
    llm_max_tokens = 8192   # Max response tokens

Service can override via environment (NVRX_ATTRSVC_LLM_MODEL, etc.)

MCP CLIENT CREATION (mcp_integration/mcp_client.py):

    def get_server_command() -> List[str]:
        """Resolve path to server_launcher.py in package."""
        pkg = "nvidia_resiliency_ext.attribution.mcp_integration"
        resource = pkg_files(pkg).joinpath("server_launcher.py")
        return ["python", str(resource), "--log-level", "WARNING"]

    def create_mcp_client() -> NVRxMCPClient:
        """Create MCP client with default server command."""
        return NVRxMCPClient(get_server_command())

    class NVRxMCPClient:
        """Async context manager for MCP server communication."""
        
        async def __aenter__(self):
            # Launch subprocess with StdioServerParameters
            server_params = StdioServerParameters(
                command="python",
                args=[str(server_launcher_path), "--log-level", "WARNING"],
                env=dict(os.environ)  # Pass PYTHONPATH, NVIDIA_API_KEY, etc.
            )
            # Connect via mcp.client.stdio.stdio_client
            self._context = await stdio_client(server_params)
            # Initialize ClientSession
            await self.session.initialize()
            return self

        async def run_module(self, module_name: str, **kwargs) -> Dict:
            """Call a tool on the MCP server."""
            result_str = await self.session.call_tool(module_name, kwargs)
            return deserialize_result(result_str)

    ERROR HANDLING:
        - If server_launcher.py not found → FileNotFoundError at client creation
        - If subprocess fails to start → Exception in __aenter__
        - If MCP call times out → asyncio.TimeoutError (wrapped by COMPUTE_TIMEOUT)
        - If LLM API fails → Error propagated in result dict

12.3 LLM RESPONSE PARSING
.........................

The log_analyzer module returns raw text. Parse with parse_llm_response():

    from nvidia_resiliency_ext.attribution.log_analyzer import parse_llm_response

    @dataclass
    class ParsedLLMResponse:
        auto_resume: str              # "YES" | "NO" | "ERRORS NOT FOUND"
        auto_resume_explanation: str  # Reason for decision
        attribution_text: str         # Root cause attribution
        checkpoint_saved_flag: int    # 0 or 1

Expected LLM output format:
    <auto_resume_decision>           # Line 1: YES/NO/ERRORS NOT FOUND
    <auto_resume_explanation>        # Line 2: Explanation
    ...
    Attribution: <attribution_text>  # Attribution section
    
    <checkpoint_saved>               # true/false

12.4 RESULT FORMAT
..................

Success:
    {
        "module": "log_analyzer",
        "state": "complete",
        "result": [
            ["<raw_llm_text>", {...metadata...}]
        ]
    }

Timeout:
    {
        "module": "log_analyzer",
        "state": "timeout",
        "result": [],
        "error": "Analysis timed out after 300s"
    }

Error:
    {
        "module": "log_analyzer",
        "state": "error",
        "result": [],
        "error": "<error message>"
    }

12.5 FILE SIZE HANDLING
.......................

No hard limit enforced. Implementation considerations:
    - log_analyzer reads entire file into memory
    - Very large files (>100MB) may cause memory pressure
    - LLM context window limits effective analysis (~32K tokens typical)
    - Future improvement: streaming reads for large files

12.6 FILE READING WHILE WRITING
...............................

    Log files may be actively written during analysis. Handling:

    SINGLE MODE:
        - GET triggers analysis only after job terminates
        - By then, file writing should be complete
        - If still writing: analyze partial file (acceptable - shows current state)

    SPLITLOG MODE:
        - File is analyzed when NEXT file appears (implies previous complete)
        - Final file analyzed on GET (job terminated)
        - Partial read possible but rare

    SAFEGUARDS:
        - MIN_FILE_SIZE_KB check prevents analyzing empty/tiny files
        - No file locking (would interfere with writer)
        - Read entire file at once (no streaming - avoids partial line issues)
        - UTF-8 decode with errors='ignore' handles incomplete writes

    EDGE CASES:
        - File truncated during read: get partial content (acceptable)
        - File grows during read: get content as of read start (acceptable)
        - NFS caching: may see stale content (poll handles eventual consistency)

12.7 LLM SERVICE AVAILABILITY
.............................

    - NVIDIA_API_KEY not found (env var or file) → fail startup (service unhealthy)
    - NVIDIA_API_KEY invalid → detected on first analysis (auth error)
    - LLM service down → analysis errors increase → health check degraded/fail
    - No proactive health probe (reactive via error rate only)

12.8 RETRY POLICY
.................

    - NO automatic retries on analysis failure (timeout or error)
    - Error/timeout result is cached in file_info[filename].wl_restart_results
    - Subsequent GETs return cached error (no re-analysis)
    - To retry: wait for TTL_TERMINATED_SECONDS (1 hour), then GET triggers fresh analysis
    - Rationale: Simplicity; avoids retry storms on persistent failures

================================================================================

13. LOG FILE MARKER PARSING
--------------------------------------------------------------------------------

Parses markers written to jobOutputLogFile by the job wrapper script.

13.1 MARKER PURPOSES
....................

    | Marker           | Purpose                                              |
    |------------------|------------------------------------------------------|
    | << START PATHS >>| Scheduler restart counting (each = one sched_restart)|
    | LOGS_DIR=        | Mode detection (present = SPLITLOG, absent = SINGLE) |
    | << END PATHS >>  | Block terminator (optional, for parsing clarity)     |

13.2 FILE READING
.................

    - Encoding: UTF-8
    - Error handling: errors='ignore' (invalid bytes silently dropped)

13.3 ALGORITHMS
...............

    MODE DETECTION (POST flow):
        1. Read first 4KB of file (efficiency - markers at top)
        2. Find FIRST << START PATHS >> block
        3. If LOGS_DIR= found in that block → SPLITLOG mode
        4. If no LOGS_DIR= → SINGLE mode

    SCHEDULER RESTART COUNTING (poll/GET):
        1. Read entire jobOutputLogFile
        2. Count lines that ARE the marker (line.strip() == marker)
           - Uses line-based parsing to avoid false positives from log output
             that contains the marker text as part of a longer line
        3. sched_restarts = count

13.4 MULTI-RESTART FILES
........................

    - Each scheduler restart writes its own << START PATHS >> block
    - For splitlog: LOGS_DIR should be same path in each block (directory, not file)
    - For single: No LOGS_DIR in any block
    - Mode is determined by FIRST block only; subsequent blocks increment sched_restarts

    LOGS_DIR CHANGE BETWEEN RESTARTS:
        If LOGS_DIR differs between << START PATHS >> blocks:
        - Use LATEST (last) LOGS_DIR found in file
        - Log WARNING: "LOGS_DIR changed between restarts, using latest"
        - Rationale: Latest directory most likely contains current log files

13.5 EXAMPLES
.............

    Single mode (no LOGS_DIR):
        << START PATHS >>
        RANK=0
        << END PATHS >>
    
    Splitlog mode (with LOGS_DIR):
        << START PATHS >>
        LOGS_DIR=/logs/job_12345
        RANK=0
        << END PATHS >>

================================================================================
                               SPLITLOG MODE
================================================================================

14. SPLITLOG OVERVIEW
--------------------------------------------------------------------------------

Splitlog mode is used when jobs write logs to a separate LOGS_DIR folder.
Enables per-cycle analysis for multi-restart jobs.

14.1 ACTIVATION CONDITIONS
..........................

    | Condition                        | Required |
    |----------------------------------|----------|
    | POST includes job_id             | Yes      |
    | Job output has << START PATHS >> | Yes      |
    | LOGS_DIR= found in block         | Yes      |
    | LOGS_DIR is readable             | Yes      |

14.2 ARCHITECTURE
.................

    ┌─────────────┐    POST     ┌─────────────┐    poll    ┌─────────────┐
    │  smonsvc    │ ──────────► │  attrsvc    │ ─────────► │  LOGS_DIR   │
    │ (job_id)    │             │ (splitlog)  │            │ (log files) │
    └─────────────┘             └─────────────┘            └─────────────┘

14.3 FLOW
.........

    1. smonsvc POSTs job output path with job_id and user
    2. attrsvc parses job output for LOGS_DIR in << START PATHS >> block
    3. SplitlogTracker polls LOGS_DIR for log files (every POLL_INTERVAL)
    4. Background poll triggers analysis when new cycle detected
    5. smonsvc GETs results when job terminal (COMPLETED, FAILED, etc.)
    6. GET marks job terminated, triggers final cycle analysis

KEY DIFFERENCES FROM SINGLE MODE: See Section 5.1 MODE DIFFERENCES table.

================================================================================

15. SPLITLOG TRACKER
--------------------------------------------------------------------------------

A helper class that manages splitlog-mode jobs. Does NOT own job storage -
operates on Job objects stored in LogAnalyzer._jobs (accessed via callbacks).

15.1 RESPONSIBILITIES
.....................

    | Responsibility       | Description                                      |
    |----------------------|--------------------------------------------------|
    | Background polling   | Check splitlog jobs every POLL_INTERVAL          |
    | Restart detection    | Count << START PATHS >> markers in job output    |
    | Log file discovery   | Scan LOGS_DIR for matching files                 |
    | Analysis triggering  | Trigger LLM when cycle complete                  |

15.2 INITIALIZATION
...................

    tracker = SplitlogTracker(
        poll_interval=POLL_INTERVAL_SECONDS,           # 5 minutes
        log_pattern="*_{job_id}_*.log",                # Glob pattern for log files
        terminated_job_ttl=TTL_TERMINATED_SECONDS,     # 1 hour
        max_job_age=TTL_MAX_JOB_AGE_SECONDS,           # 6 months
    )

    Internal fields:
        _lock: threading.Lock          # Protects job state access
        _poll_thread: threading.Thread # Background poll (daemon=True)
        _stop_event: threading.Event   # Signals poll thread to stop
        _executor: ThreadPoolExecutor  # For non-blocking analysis (max_workers=4)
        _jobs_cleaned: int             # Counter for cleaned up jobs

15.3 CALLBACKS
..............

    Set by LogAnalyzer to integrate tracker with main service:

    | Callback                    | Signature                          | Purpose                    |
    |-----------------------------|------------------------------------|----------------------------|
    | analyze_callback            | (log_file, user, job_id) → None    | Trigger analysis (fire-and-forget) |
    | pending_check_callback      | () → None                          | Check pending for LOGS_DIR |
    | get_splitlog_jobs_callback  | () → List[Job]                     | Get all splitlog jobs      |
    | cleanup_job_callback        | (path) → None                      | Remove job from _jobs      |

    NOTE: analyze_callback uses fire-and-forget pattern - it schedules analysis
    on the main event loop but returns immediately. Results are stored in the
    RequestCoalescer cache. See Section 19.5 for concurrency details.

15.4 POLL LOOP
..............

    Runs every POLL_INTERVAL_SECONDS:

    1. Check pending jobs for LOGS_DIR (may promote to splitlog mode)
    2. For each splitlog job:
        a. Read job output for << START PATHS >> count → sched_restarts
        b. Scan LOGS_DIR for log files matching pattern → update file_info
        c. If new log file found → previous file complete → trigger analysis
    3. Cleanup jobs that have exceeded TTL

================================================================================

16. RESTART DETECTION & LOG FILE DISCOVERY
--------------------------------------------------------------------------------

16.1 RESTART DETECTION
......................

    - Scheduler restarts: See Section 1.2 TERMINOLOGY and Section 13 for algorithm
    - Workload restarts: See Section 1.2 TERMINOLOGY and Section 1.4 CONTENT SPLITTING

16.2 LOG FILE DISCOVERY
.......................

    Pattern: *_{job_id}_*.log (excludes *.env.log)

16.3 FILE SORTING
.................

    ALGORITHM: Use most specific pattern available per file.

    | Priority | Pattern                                    | Example                    |
    |----------|--------------------------------------------|-----------------------------|
    | 1        | _cycle(\d+)\.log$                          | model_12345_cycle0.log      |
    | 2        | _date_YY-MM-DD_time_HH-MM-SS\.log$         | model_12345_date_26-01-19_time_01-51-12.log |
    | 3        | Fallback to modification time              | (any .log file)             |

    MIXED FILE HANDLING:
        - Files with _cycleN.log → sort by cycle number (extracted as int)
        - Files with timestamp only → sort by date/time (parsed as datetime)
        - Files with neither → sort by mtime
        - If mix of patterns in same LOGS_DIR → each file uses its own best pattern

    EXAMPLE (mixed patterns in same directory):
        ├── model_12345_cycle0.log          → sort key: cycle=0
        ├── model_12345_cycle1.log          → sort key: cycle=1
        ├── model_12345_date_26-01-20.log   → sort key: datetime(2026-01-20)
        └── model_12345_other.log           → sort key: mtime

        Result order: cycle0, cycle1, then timestamp/mtime files interleaved by time

    INDEX STABILITY PROBLEM:
        If a file with an older timestamp is discovered late (network lag, FS latency),
        re-sorting would shift indices of all subsequent files.
        
        Example:
            T1: Discover file_A (10:00) → file_info[0]
            T2: Discover file_B (11:00) → file_info[1]
            T3: Discover file_C (10:30) LATE → if re-sorted: A=0, C=1, B=2
            API call ?file=1 now returns different data!

    SOLUTION - USE FILENAME AS IDENTIFIER:
        Instead of ?file=0, use ?file=model_12345_cycle0.log
        
        Why filename is better:
            - Stable: filename doesn't change
            - Self-documenting: API call shows exactly what file
            - No ambiguity: no index shifting possible
            - Simpler data model: no discovery-order tracking needed
        
        Trade-off: Longer URLs, requires URL encoding (acceptable)
        
        Internal storage:
            job.file_info: Dict[str, FileInfo]  # key = filename (not index)
        
        Response format:
            {
                "files": [
                    {"log_file": "model_12345_cycle0.log", ...},
                    {"log_file": "model_12345_cycle1.log", ...}
                ]
            }
            # files[] sorted chronologically for display
        
        API usage:
            GET /logs?log_path=...&file=model_12345_cycle0.log

16.4 WORKLOAD LOG FILE STRATEGIES
.................................

    Strategy A: One file per scheduler restart (multi-cycle per file)
        - File: 55B_model_1523572_date_26-01-19_time_01-51-12.log
        - Contains: Multiple Cycle: N markers inside
        - Sorting: By date/time in filename
        - Analysis: See Section 1.4 CONTENT SPLITTING

    Strategy B: One file per cycle (single-cycle per file)
        - File: 55B_model_1617268_date_26-01-27_time_12-50-58_cycle0.log
        - Contains: Single cycle, no Cycle: N markers needed
        - Sorting: By cycle number in filename (preferred)
        - Analysis: Entire file is one cycle

16.5 FILE COMPLETION LOGIC
..........................

    A log file is ready for analysis when:
        - A NEXT log file exists, OR
        - Job is terminated (GET called)
    
    When file is ready for analysis:
        - Trigger analysis for that file
        - file_info[filename].analysis_triggered = True
        - Result stored in file_info[filename].wl_restart_results
        - After analysis: file_info[filename].wl_restarts, analyzed_at populated

16.6 LOG FILE EXAMPLES
......................

    Strategy A (one file per scheduler restart, sorted by date/time):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ File 0: 55B_model_1523572_date_26-01-19_time_01-51-12.log               │
    │         Contains: Cycle: 0, Cycle: 1, Cycle: 2 (3 wl restarts)          │
    │                                                                         │
    │ File 1: 55B_model_1523572_date_26-01-19_time_04-22-45.log               │
    │         Contains: Cycle: 0, Cycle: 1 (2 wl restarts)                    │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Strategy B (one file per workload restart, sorted by cycle number):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ File 0: 55B_model_1617268_date_26-01-27_time_12-50-58_cycle0.log        │
    │ File 1: 55B_model_1617268_date_26-01-27_time_14-30-22_cycle1.log        │
    │ File 2: 55B_model_1617268_date_26-01-27_time_16-05-11_cycle2.log        │
    └─────────────────────────────────────────────────────────────────────────┘

16.7 MISMATCH HANDLING
......................

    If sched_restarts != len(file_info):
        - Log WARNING but continue
        - Analyze available log files
        - May indicate job failure before log file written

================================================================================

17. SPLITLOG GET FLOW
--------------------------------------------------------------------------------

17.1 REQUEST PROCESSING
.......................

    GET /logs (jobOutputLogFile) for SPLITLOG mode:

    1. Validate jobOutputLogFile (same as single mode)

    2. Lookup job by jobOutputLogFile:
        - Job must exist with mode=SPLITLOG
        - If job not found → treat as single mode (GET-without-POST)

    3. Mark job terminated:
        - job.terminated = True
        - job.terminated_at = now
        - Triggers analysis of final file (if not already done)

17.2 QUERY PARAMETERS
.....................

    | Parameter  | Type | Default | Description                              |
    |------------|------|---------|------------------------------------------|
    | file       | str  | -       | Filename (e.g., model_12345_cycle0.log)  |
    | wl_restart | int  | -       | 0-indexed workload restart within file   |
    | all_files  | bool | false   | Return complete history                  |

    VALIDATION:
        - wl_restart must be >= 0 (validated at HTTP layer with ge=0, library returns AnalyzerError)

    PARAMETER HANDLING:

    | Query                              | Result                                        |
    |------------------------------------|-----------------------------------------------|
    | (no params)                        | Latest completed file, all workload restarts  |
    | ?all_files=true                    | All completed files with their wl_restarts    |
    | ?file=model_12345_cycle0.log       | All workload restarts from that file          |
    | ?file=model_12345_cycle0.log&wl_restart=2 | Third workload restart from that file  |
    | ?file=nonexistent.log              | 404 "file not found"                          |
    | ?wl_restart >= count               | 404 "wl_restart not found"                    |

17.3 RESULT RETRIEVAL
.....................

    If result not in cache:
        - Check _in_flight for log_file
        - If exists → wait on future
        - If not → trigger analysis, wait for result

    Return: {result: <analysis>}

17.4 CURL EXAMPLES
..................

    # Get latest file result (default - all workload restarts in that file)
    curl "http://localhost:8000/logs?log_path=/data/logs/slurm-12345.out"
    
    # Get complete history (all completed files)
    curl "http://localhost:8000/logs?log_path=/data/logs/slurm-12345.out&all_files=true"
    
    # Get specific file by filename
    curl "http://localhost:8000/logs?log_path=/data/logs/slurm-12345.out&file=model_12345_cycle0.log"
    
    # Get specific workload restart within a file
    curl "http://localhost:8000/logs?log_path=/data/logs/slurm-12345.out&file=model_12345_cycle0.log&wl_restart=2"

TERMINATION BEHAVIOR:
    - First GET marks job terminated
    - Subsequent GETs return cached results
    - After TTL_TERMINATED (1 hour), job cleaned up
    - Further GETs fail (job not found)

================================================================================
                           MAINTENANCE & INTERNALS
================================================================================

18. CLEANUP
--------------------------------------------------------------------------------

Runs every POLL_INTERVAL_SECONDS in background thread.

18.1 CLEANUP RULES
..................

    | Job State     | Condition                                          | TTL             | Action      |
    |---------------|----------------------------------------------------|-----------------|-------------|
    | PENDING       | mode=PENDING AND age > TTL_PENDING_SECONDS         | 1 week          | Remove job  |
    | Non-terminated| terminated=False AND age > TTL_MAX_JOB_AGE_SECONDS | 6 months        | Remove job  |
    | Terminated    | terminated=True AND age > TTL_TERMINATED_SECONDS   | 1 hour          | Remove job  |
    | In-flight     | entry age > TTL_IN_FLIGHT_STUCK_SECONDS            | 10 min          | Remove entry|

    Notes:
    - `age` for non-terminated jobs = (now - created_at)
    - `age` for terminated jobs = (now - terminated_at)
    - `terminated` is set on successful GET for both SINGLE and SPLITLOG modes

18.2 TTL CONSTANTS
..................

    See Section 3.2 for authoritative values.

================================================================================

19. CONCURRENCY & THREAD/ASYNC COORDINATION
--------------------------------------------------------------------------------

19.1 THREADING MODEL
....................

    | Thread          | Purpose                          | Concurrency Type       |
    |-----------------|----------------------------------|------------------------|
    | Main (uvicorn)  | HTTP requests, event loop        | Cooperative (async)    |
    | Background      | Poll loop, file checks, cleanup  | Thread-parallel        |

    HTTP requests are NOT separate threads:
        - All requests run as coroutines on the single event loop thread
        - When one coroutine awaits (I/O, future), another runs
        - This is cooperative multitasking, not thread-per-request

19.2 SHARED STATE
.................

    | Variable     | Type                | Access Pattern                  |
    |--------------|---------------------|---------------------------------|
    | _jobs        | Dict[str, Job]      | Both contexts, protected by lock|
    | _in_flight   | Dict[str, Entry]    | Both contexts, protected by lock|
    | _lock        | threading.Lock      | Standard threading lock         |

19.3 LOCK STRATEGY
..................

    Use threading.Lock (not asyncio.Lock) for simplicity:
        - Works in both async and sync contexts
        - Only in-memory dict operations under lock (no I/O), so blocking is acceptable

    Pattern:
        with self._lock:
            # Quick read/modify only, no I/O
            job = self._jobs.get(path)
        # Log and I/O after releasing lock
        logger.debug(f"Retrieved job: {path}")

    Rules:
        - Do NOT perform I/O while holding lock
        - Do NOT log while holding lock
        - Keep critical section minimal (read/modify only)

19.4 FUTURE HANDLING
....................

    _in_flight entries use asyncio.Future for request coalescing:

        InFlightEntry:
            - future: asyncio.Future[Dict]
            - started_at: float

    First GET creates future:
        entry = InFlightEntry(future=loop.create_future(), started_at=now)
        self._in_flight[path] = entry
        # Start analysis...
        entry.future.set_result(result)

    Subsequent GETs wait on same future:
        entry = self._in_flight.get(path)
        if entry:
            result = await entry.future  # Multiple awaiters allowed
            return result

    Thread safety note:
        - asyncio.Future is NOT thread-safe for set_result()
        - Only call set_result() from async context (event loop thread)
        - Poll thread should NOT touch futures directly

19.5 POLL THREAD vs ASYNC CONTEXT
.................................

    | Context      | Creates Jobs | Triggers Analysis | Uses _in_flight | Uses Coalescer |
    |--------------|--------------|-------------------|-----------------|----------------|
    | Poll thread  | No           | Yes (proactive)   | No              | Yes (write)    |
    | Async (HTTP) | Yes          | Yes (on-demand)   | Yes             | Yes (read)     |

    DETAILS:

    Poll thread (background - SplitlogTracker):
        - Reads _jobs with lock
        - Updates job.mode (pending → single/splitlog)
        - Discovers new log files in LOGS_DIR
        - Triggers analysis for completed files (fire-and-forget)
        - Runs cleanup (removes expired jobs)
        
        ANALYSIS TRIGGERING FROM POLL THREAD (Fire-and-Forget):

            Uses non-blocking pattern to avoid deadlocking the async event loop:
                - Poll thread calls analyze_callback(log_file, user, job_id)
                - Callback (_fire_and_forget_analyze) uses run_coroutine_threadsafe()
                - Does NOT wait for result - returns immediately
                - Main loop executes the analysis asynchronously
                - Results stored in RequestCoalescer cache
                - GET requests retrieve results from cache

            Implementation in SplitlogTracker._trigger_analysis():
                - Uses ThreadPoolExecutor to run analysis in separate thread
                - FileInfo.analysis_triggered set immediately (before completion)
                - FileInfo.analysis_complete set when done
                - Error handling logs failures but doesn't block poll

            Benefits:
                + Poll thread never blocks waiting for LLM
                + Multiple files can be discovered and analyzed concurrently
                + High job volume supported (analysis parallelized)
                + No event loop deadlock risk

            Requires: LogAnalyzer.set_event_loop() called during app startup
        
        Why fire-and-forget:
            - POLL_INTERVAL_SECONDS is 5 minutes
            - LLM analysis can take 1-5 minutes per file
            - Blocking would delay discovery of other files
            - Results available via GET (coalescer cache lookup)

    Async context (HTTP requests):
        - Creates/updates jobs
        - Creates _in_flight entries (for request coalescing)
        - Triggers LLM analysis for SINGLE mode and on-demand splitlog requests
        - Sets futures with results
        - Note: Poll thread also triggers analysis proactively (see above)

19.6 RACE CONDITIONS
....................

    POST then immediate GET (same path):
        - POST creates job with mode=JobMode.PENDING
        - GET arrives before poll classifies mode
        - GET should check file and classify inline (not wait for poll)

    Concurrent GETs (same path, no result):
        - First GET creates _in_flight entry
        - Second GET sees _in_flight, waits on future
        - Both receive same result (no duplicate analysis)

    Cleanup during GET:
        - GET holds reference to job
        - Cleanup removes job from _jobs
        - GET completes with stale job reference (acceptable)
        - Next GET will create new job

    Poll thread vs GET modifying same job:
        PROBLEM:
            - Poll thread scans LOGS_DIR, prepares file_info updates
            - GET arrives, promotes mode, initializes file_info
            - Poll thread overwrites GET's file_info → data loss

        SOLUTION - IDEMPOTENT UPDATES:
            1. Mode promotion: check-then-set pattern
               - Only promote if mode == PENDING (not if already SINGLE/SPLITLOG)
               - First to promote wins

            2. file_info updates: merge, never overwrite
               - Only add new file_info entries (don't replace existing)
               - Only set wl_restart_results if currently empty
               - Only set analysis_complete if currently False

            3. Lock scope (LogAnalyzer._jobs_lock - threading.Lock):
               - Protects all access to self._jobs dict
               - Hold lock during check-and-modify (atomic decision)
               - Release lock during I/O (file scan, LLM call)
               - Re-acquire lock for final update

        PATTERN:
            with self._jobs_lock:
                job = self._jobs.get(path)
                if job and job.mode != PENDING:
                    return  # Already promoted, skip
                if job:
                    job.mode = new_mode  # Claim the promotion
            
            # Do expensive work (file scan) without lock
            new_files = scan_logs_dir()
            
            with self._jobs_lock:
                job = self._jobs.get(path)
                if job:
                    for filename, filepath in new_files:
                        if filename not in job.file_info:  # Only add new
                            job.file_info[filename] = FileInfo(log_file=filepath, ...)

        INVARIANT: Once file_info[filename].wl_restart_results is set, it's never overwritten.

================================================================================

20. COUNTERS (for /stats endpoint)
--------------------------------------------------------------------------------

20.1 CUMULATIVE COUNTERS
........................

Cumulative counters always increase over the service lifetime.

    | Category    | Counter                      | Description                         |
    |-------------|------------------------------|-------------------------------------|
    | Submissions | total_posts                  | Total POST requests received        |
    |             | total_posts_single           | POSTs that created single-mode      |
    |             | total_posts_pending          | POSTs that created pending-mode     |
    |             | total_posts_splitlog         | POSTs that created splitlog-mode    |
    |             | total_posts_duplicate        | POSTs for existing jobs             |
    |             | total_posts_rejected_limit   | POSTs rejected (MAX_JOBS)           |
    | Requests    | total_gets                   | Total GET requests received         |
    |             | total_gets_cache_hit         | GETs served from cache              |
    |             | total_gets_coalesced         | GETs that waited on in_flight       |
    |             | total_gets_triggered         | GETs that triggered analysis        |
    | Analysis    | total_analysis               | Total LLM analyses started          |
    |             | total_analysis_success       | Analyses completed successfully     |
    |             | total_analysis_timeout       | Analyses that timed out             |
    |             | total_analysis_error         | Analyses that failed                |
    | Cleanup     | total_cleanup_pending        | Pending jobs removed (TTL)          |
    |             | total_cleanup_terminated     | Terminated jobs removed (TTL)       |
    |             | total_cleanup_abandoned      | Non-terminated jobs removed (TTL)   |
    |             | total_cleanup_in_flight      | Stuck in_flight entries removed     |
    | Dataflow    | total_dataflow_posts         | Total dataflow post attempts        |
    |             | total_dataflow_success       | Successful dataflow posts           |
    |             | total_dataflow_failed        | Failed dataflow posts               |
    | Rate Limit  | total_rate_limited           | Total requests rejected (429)       |
    |             | total_rate_limited_submit    | POST /logs rate limited             |
    |             | total_rate_limited_analyze   | GET /logs rate limited              |
    |             | total_rate_limited_preview   | GET /print rate limited             |

20.2 GAUGES
...........

Gauges reflect current state and can increase or decrease.

    | Gauge            | Description                                    |
    |------------------|------------------------------------------------|
    | jobs_pending     | Current count of pending jobs                  |
    | jobs_single      | Current count of single-mode jobs              |
    | jobs_splitlog    | Current count of splitlog-mode jobs            |
    | jobs_terminated  | Current count of terminated jobs               |
    | files_analyzed   | Total files with analysis_complete=True        |
    | in_flight_count  | Current count of in_flight entries             |

Poll thread observability (for debugging):

    | Gauge                   | Description                                    |
    |-------------------------|------------------------------------------------|
    | poll_last_run_at        | Timestamp of last poll iteration (epoch)       |
    | poll_duration_seconds   | Duration of last poll iteration                |
    | poll_is_running         | True if poll iteration currently in progress   |

    Usage:
        - If (now - poll_last_run_at) > 2 × POLL_INTERVAL_SECONDS → poll may be stuck
        - If poll_duration_seconds consistently high → investigate performance
        - Exposed via GET /stats

Splitlog-specific counters:
    - total_files_discovered: Total log files discovered in LOGS_DIR (all jobs)
    - total_poll_cycles: Total background poll iterations
    - total_poll_promotions: Jobs promoted from pending → splitlog via poll

================================================================================
                            OPTIONAL & REFERENCE
================================================================================

21. DATAFLOW (OPTIONAL - PROPRIETARY)
--------------------------------------------------------------------------------

Posts LLM analysis results to Elasticsearch via NVIDIA nvdataflow API.
This is optional and proprietary - implemented in separate module for easy
exclusion or replacement.

Files (see section 2 PROJECT STRUCTURE):
    - nvrx_attrsvc/dataflow.py                # Elasticsearch posting via nvdataflow
    - nvrx_attrsvc/config.py setup()         # Wires lib postprocessing via configure(poster, cluster_name, dataflow_index, slack_*)
    - nvidia_resiliency_ext/attribution/postprocessing/  # config, configure(), ResultPoster, post_results, Slack

Configuration:
    - CLUSTER_NAME, DATAFLOW_INDEX: env prefix NVRX_ATTRSVC_ (e.g. NVRX_ATTRSVC_CLUSTER_NAME)
    - SLACK_BOT_TOKEN, SLACK_CHANNEL: no prefix (env vars SLACK_BOT_TOKEN, SLACK_CHANNEL)
    - If DATAFLOW_INDEX empty, dataflow posting disabled; if SLACK_BOT_TOKEN empty, Slack disabled
    - Slack notifications sent for auto_resume = "STOP - DONT RESTART IMMEDIATE"

When triggered:
    - After successful LLM analysis (when result contains valid data)
    - Called from _run_llm_analysis() after parsing LLM response

Record format (built by build_dataflow_record):
    {
        "job_id": string,
        "log_path": string,
        "cluster_name": string,
        "user": string,
        "processing_time": float,
        "auto_resume": bool,
        "auto_resume_explanation": string,
        "attribution_text": string,
        ...
    }

Retry logic:
    - MAX_RETRIES: 3
    - Exponential backoff: 0.5s, 1s, 2s
    - Returns success/failure (does not throw)

Stats (exposed via /stats → dataflow):
    - total_posts: Total post attempts
    - successful_posts: Successful posts
    - failed_posts: Failed posts (after all retries)

To disable:
    - Leave DATAFLOW_INDEX empty (default)

To replace with custom implementation:
    1. Create new module implementing post(data, index) → bool
    2. In config.setup(), pass default_poster=ResultPoster(post_fn=your_post) to configure()

================================================================================

22. LOGGING
--------------------------------------------------------------------------------

| Level   | Category        | Message Template                                    |
|---------|-----------------|-----------------------------------------------------|
| INFO    | Startup         | Starting NVRX Attribution Service on {host}:{port}  |
| INFO    | Startup         | API Documentation: http://{host}:{port}/docs        |
| INFO    | Shutdown        | AttributionService shutdown complete                |
| INFO    | Request         | POST /logs path={path} user={user} job_id={job_id}  |
| INFO    | Request         | POST complete: mode={mode}                          |
| INFO    | Request         | GET /logs path={path}                               |
| INFO    | Request         | GET complete: cached={bool} coalesced={bool}        |
| INFO    | Analysis        | Analysis started: path={path}                       |
| INFO    | Analysis        | Analysis complete: path={path} state={state}        |
| DEBUG   | State           | Job created: path={path} mode={mode}                |
| DEBUG   | State           | Job updated: path={path} field={field} value={val}  |
| DEBUG   | State           | In_flight added/removed: path={path}                |
| DEBUG   | Poll            | Poll iteration: pending_jobs={count}                |
| DEBUG   | Cleanup         | Cleanup: removed {count} jobs                       |
| WARNING | File            | File not found at GET: path={path}                  |
| WARNING | Analysis        | Analysis timeout/error: path={path}                 |
| WARNING | Cleanup         | Stuck in_flight cleaned: path={path} age={seconds}s |
| WARNING | Duplicate       | Duplicate POST with different job_id: path={path}   |
| WARNING | Dataflow        | Dataflow post failed, retrying: {error}             |
| WARNING | Rate Limit      | Rate limit exceeded: endpoint={endpoint}            |
| ERROR   | Dataflow        | Failed to post to dataflow after {max} attempts     |
| ERROR   | Import          | Can't import nvdataflow                             |

================================================================================

23. DEPENDENCIES
--------------------------------------------------------------------------------

See services/pyproject.toml for the authoritative dependency list.

================================================================================
                                  DEVELOPMENT
================================================================================

24. TESTING STRATEGY
--------------------------------------------------------------------------------

24.1 TEST STRUCTURE
...................

    tests/
    ├── unit/
    │   ├── test_job.py              # Job, FileInfo dataclass tests
    │   ├── test_path_validation.py  # Path validation logic
    │   ├── test_mode_detection.py   # Mode classification logic
    │   ├── test_marker_parsing.py   # Marker extraction
    │   └── test_file_discovery.py   # Splitlog file sorting
    ├── integration/
    │   ├── test_service.py          # AttributionService with mocked LLM
    │   ├── test_http_api.py         # HTTP endpoints with TestClient
    │   ├── test_background_poll.py  # Poll thread behavior
    │   └── test_coalescing.py       # Request coalescing
    ├── e2e/
    │   └── test_full_flow.py        # End-to-end with real LLM (optional)
    ├── conftest.py                  # Fixtures
    └── fixtures/
        ├── logs/                    # Sample log files
        └── responses/               # Mock LLM responses

24.2 MOCKING GUIDANCE
.....................

    WHAT TO MOCK:

    | Component           | Mock Strategy                                      |
    |---------------------|---------------------------------------------------|
    | LLM (MCP client)    | Mock create_mcp_client() → return fake responses  |
    | Filesystem          | Use tmp_path fixture, create real temp files      |
    | time.monotonic()    | Use freezegun or manual mock for TTL tests        |
    | Background thread   | Don't start in unit tests; test poll logic directly|

    LLM MOCK FIXTURE:

        @pytest.fixture
        def mock_llm_client(mocker):
            """Mock MCP client to return canned responses."""
            mock = mocker.patch("nvrx_attrsvc.llm.mcp_client.create_mcp_client")
            
            async def fake_run_module(**kwargs):
                return {
                    "module": "log_analyzer",
                    "state": "complete",
                    "result": [["YES\nNo errors found\nAttribution: Normal exit", {}]]
                }
            
            mock.return_value.__aenter__.return_value.run_module = fake_run_module
            return mock

    FILESYSTEM FIXTURE:

        @pytest.fixture
        def sample_logs(tmp_path):
            """Create sample log files for testing."""
            job_output = tmp_path / "slurm-12345.out"
            job_output.write_text("""
            << START PATHS >>
            LOGS_DIR={logs_dir}
            << END PATHS >>
            """.format(logs_dir=tmp_path / "logs"))
            
            logs_dir = tmp_path / "logs"
            logs_dir.mkdir()
            (logs_dir / "job_12345_date_26-01-01_time_00-00-00.log").write_text(
                "profiling.py:... Cycle: 0\nTraining started..."
            )
            
            return {"job_output": job_output, "logs_dir": logs_dir}

24.3 KEY TEST SCENARIOS
.......................

    UNIT TESTS:
        - Path validation: symlinks, traversal, case sensitivity, null bytes
        - Mode detection: all combinations from Section 9.2 table
        - Marker parsing: missing markers, malformed blocks, multiple blocks
        - File discovery: sorting by cycle, date/time, mtime fallback

    FILE DISCOVERY - MIXED PATTERN TEST:
        Test: Mixed file patterns in same LOGS_DIR sorted correctly
        Input: Files with _cycleN, timestamp-only, and no pattern in same directory
        Expected: Cycle files sorted by number, others by timestamp/mtime
        Verify: cycle0 < cycle1 < cycle2, regardless of other files present

    INTEGRATION TESTS:
        - POST → GET flow (single mode)
        - POST → poll → GET flow (splitlog mode)
        - Concurrent GETs (coalescing)
        - GET-without-POST
        - TTL cleanup (with mocked time)
        - Rate limiting

    EDGE CASES TO TEST:
        - File appears after POST but before first poll
        - File disappears between POST and GET
        - LLM timeout during GET
        - Concurrent POSTs with same path, different job_id
        - MAX_JOBS limit reached
        - Malformed LLM response

24.4 COVERAGE EXPECTATIONS
..........................

    Minimum coverage targets:
        - core/ modules: 90%+
        - http/ modules: 80%+
        - splitlog/ modules: 85%+
        - Overall: 85%+

================================================================================

25. DEPLOYMENT
--------------------------------------------------------------------------------

25.1 DOCKER
...........

    See services/nvrx_attrsvc/deploy/Dockerfile

25.2 RESOURCE REQUIREMENTS
..........................

    See nvrx_attrsvc/README.md "Resource Requirements" section.

25.3 KUBERNETES
...............

    See services/nvrx_attrsvc/deploy/kubernetes.yaml

25.4 RUN AND SNAPSHOT SCRIPTS
.............................

    All run and snapshot scripts live under deploy/:
    - deploy/run_attrsvc.sh [output_dir]   - Run service in background with logging
    - deploy/snapshot_attrsvc.sh [host] [port] - Periodic endpoint snapshot for debugging

25.5 GRACEFUL DEGRADATION
.........................

    LLM UNAVAILABLE:
        - Service remains healthy (accepts requests)
        - GET requests timeout after DEFAULT_COMPUTE_TIMEOUT_SECONDS (5 min)
        - Result stored with state=TIMEOUT
        - Health check degrades (elevated error rate)
        - No automatic retry (client must wait TTL_TERMINATED_SECONDS to retry)

    FILESYSTEM UNAVAILABLE:
        - POST fails with NOT_FOUND or NOT_READABLE
        - Existing jobs continue (in-memory state preserved)
        - Poll thread logs errors, continues polling
        - Service remains running

    MEMORY PRESSURE:
        - MAX_JOBS limit prevents unbounded growth
        - Large files may cause OOM (no mitigation - see Section 12.5)
        - Consider: file size limit or streaming reads (future)

25.6 OBSERVABILITY
..................

    METRICS EXPORT:

        GET /stats returns JSON counters. For Prometheus:
        
        Option A: Use prometheus-fastapi-instrumentator
            - Auto-exports request latency, counts
            - Custom metrics for service counters
        
        Option B: Separate /metrics endpoint (future)
            - Export counters in Prometheus format

    LOGGING:
        - JSON format recommended for production
        - Ship to centralized logging (ELK, Splunk, etc.)

    ALERTING RECOMMENDATIONS:
        - Alert on: health_status == FAIL for > 5 min
        - Alert on: compute_error_rate > 50% for > 10 min
        - Alert on: jobs_pending > 10000 (approaching MAX_JOBS)
        - Alert on: container restart loop

================================================================================
                                   APPENDIX
================================================================================

A. GLOSSARY
--------------------------------------------------------------------------------

Attribution:
    The process of analyzing a job's log file to determine the root cause of
    a failure, crash, or unexpected behavior.

Coalescing:
    Combining multiple concurrent GET requests for the same path into a single
    LLM analysis. All waiters receive the same result.

In-flight:
    An analysis that is currently being processed. Tracked in _in_flight dict
    to enable request coalescing.

Job:
    A tracked log file submission. Contains path, mode, user, and analysis result.

jobOutputLogFile:
    The main output log file for a job. Used as the unique key for job tracking.
    Examples: slurm-12345.out (SLURM), pod-abc123.log (Kubernetes).

Markers:
    Special strings written to jobOutputLogFile by job wrapper script at job
    start. Used for mode detection (SPLITLOG vs SINGLE). See section 13.

Mode:
    The classification of a job: PENDING (waiting for file), SINGLE (analyze
    jobOutputLogFile directly), or SPLITLOG (analyze separate log directory).

Pending:
    A job whose file is not yet available or too short. Background poll will
    check periodically until file is ready or TTL expires.

Splitlog:
    A mode where logs are written to a separate directory (LOGS_DIR) rather than
    the main jobOutputLogFile. Requires job_id in POST.

Scheduler restart (sched_restart):
    External scheduler restarts the job. See Section 1.2 TERMINOLOGY for full definition.

Workload restart (wl_restart):
    Training restarts within same scheduler allocation. See Section 1.2 TERMINOLOGY for
    full definition.

FileInfo:
    Tracks a single log file in LOGS_DIR. Contains file path, analysis status,
    and workload restart results (from chunk_logs_strict parsing).

chunk_logs_strict:
    LLM analyzer function that splits log file content by Cycle: N markers.
    Each chunk is analyzed separately. See Section 1.4 CONTENT SPLITTING.

SplitlogTracker:
    Background tracker class that manages splitlog-mode jobs. Handles scheduler restart
    detection, log file discovery, and triggering analysis via fire-and-forget pattern.
    Uses ThreadPoolExecutor for non-blocking analysis. See Section 15.

Terminated:
    A job (any mode) that has received a successful GET request. Marks the job
    for cleanup after TTL_TERMINATED_SECONDS. For splitlog: stops file monitoring.

TTL (Time To Live):
    Maximum time a job can exist in a given state before cleanup removes it.

================================================================================

B. QUICK REFERENCE
--------------------------------------------------------------------------------

Quick lookup for common values. See referenced sections for full details.

ENDPOINTS (Section 6):
    POST /logs          Submit job for tracking      → {mode}
    GET  /logs          Get analysis result          → {result}
    GET  /healthz       Health check                 → {status, issues}
    GET  /stats         All counters/gauges          → {...}
    GET  /jobs          List jobs (paginated)        → {total_counts, jobs}
    GET  /inflight      List in-flight requests      → {count, entries}
    GET  /print         Preview file (4KB)           → text

ERROR CODES (Section 7):
    INVALID_PATH          400    Path not absolute (client error)
    OUTSIDE_ROOT          403    Path outside ALLOWED_ROOT (forbidden)
    NOT_FOUND             404    File doesn't exist
    NOT_READABLE          403    File permission denied (server config)
    NOT_REGULAR           400    Path is not a regular file
    EMPTY_FILE            400    File is empty (GET only)
    LOGS_DIR_NOT_READABLE 403    LOGS_DIR permission denied (server config)
    JOB_LIMIT_REACHED     503    MAX_JOBS exceeded
    INTERNAL_ERROR        500    Unexpected server error

KEY CONSTANTS (Section 3.2-3.3):
    TTL_PENDING_SECONDS           1 week     Pending job expiry
    TTL_TERMINATED_SECONDS        1 hour     Terminated job expiry (after GET)
    TTL_MAX_JOB_AGE_SECONDS       6 months   Non-terminated job safety net
    POLL_INTERVAL_SECONDS         5 min      Background poll interval
    DEFAULT_COMPUTE_TIMEOUT_SECONDS  5 min   LLM analysis timeout
    MAX_JOBS                      100,000    Maximum tracked jobs
    MIN_FILE_SIZE_KB              4          Minimum file size (KB) for classification

MARKERS (Section 13):
    MARKER_START_PATHS    "<< START PATHS >>"    Paths block / cycle delimiter
    MARKER_END_PATHS      "<< END PATHS >>"      Paths block end
    MARKER_LOGS_DIR       "LOGS_DIR="            Logs directory prefix

MODE SELECTION: See Section 9.2 for full decision table.

RATE LIMITS (Section 3.1):
    POST /logs            1200/minute
    GET  /logs            60/minute
    GET  /print           120/minute

ENUMS (Section 3.4):
    JobMode:       PENDING, SINGLE, SPLITLOG
    ErrorCode:     INVALID_PATH, NOT_REGULAR, EMPTY_FILE, OUTSIDE_ROOT, 
                   NOT_READABLE, LOGS_DIR_NOT_READABLE, NOT_FOUND,
                   JOB_LIMIT_REACHED, INTERNAL_ERROR
    HealthStatus:  OK, DEGRADED, FAIL

================================================================================
