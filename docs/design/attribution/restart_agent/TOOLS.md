# Restart Agent Read-Only Tools

This document is canonical only for L1 evidence-tool availability, interfaces,
limits, rejection behavior, and observability.

`L0B.md` owns the initial model evidence view. `L1.md` owns the model loop and
prompt contract. `CONFIGURATION.md` owns route-specific advertisement and turn limits.

## Tool Boundary

Tools are read-only views over one immutable attempt:

- raw-log tools read the `LogSnapshot` created during preparation;
- evidence-object lookup reads the immutable `L0Bundle`;
- no tool rescans through a separate analyzer or mutates evidence;
- no tool reads history, policy state, credentials, or another log.

The model can call only tools advertised by the selected model route.
Implementation availability does not make a tool callable.

Unknown names, malformed arguments, unsupported requests, and calls outside the
advertised set are rejected and traced. The product does not dynamically create
executable tools from model requests.

## Advertised Set

The default route configuration advertises:

- `grep_log`;
- `read_window`.

`overview` and `get_evidence_objects` are implemented but not advertised by
default. L0B already provides initial orientation, so a route enables
`overview` only for controlled evaluation or a client that does not supply the
normal L0B view.

Tools remain generic inspection primitives. Failure-specific helpers such as
`find_device_side_assert` or `find_nccl_timeout` do not belong in this
interface.

## Fixed Response Limits

These are implementation bounds. Model routes control whether a tool is
advertised and how many model/tool rounds are allowed.

| Setting | Default |
| --- | ---: |
| `overview.head_lines` | 40 |
| `overview.tail_lines` | 80 |
| `overview.max_chars` | 12000 |
| `grep_log.result_mode` | `compact` |
| `grep_log.max_matches` | 50 |
| `grep_log.max_matches_hard_limit` | 200 |
| `read_window.before` | 20 |
| `read_window.after` | 80 |
| `read_window.max_lines` | 240 |
| `read_window.max_chars` | 50000 |
| `get_evidence_objects.max_refs` | 8 |
| `get_evidence_objects.max_chars` | 50000 |

Every response reports truncation and relevant applied limits. A cap never
silently changes source-line numbering.

## `overview`

Input: none.

This tool is opt-in because the normal L0B request already carries initial
source and evidence orientation.

Purpose: orient the model to file scale, bounded head/tail content, and the
existing deterministic evidence without recomputing L0.

Example output:

```json
{
  "line_count": 1234,
  "byte_size": 456789,
  "head": [{"line": 1, "text": "..."}],
  "tail": [{"line": 1200, "text": "..."}],
  "deterministic_summary": {
    "progress_lines": [],
    "registry_candidate_groups": [],
    "registry_candidate": null,
    "candidate_anchors": [],
    "failure_episodes": [],
    "cause_confirmations": [],
    "cascade_groups": [],
    "termination_candidates": []
  },
  "truncated": false
}
```

The output excludes absolute paths, basenames, eval labels, and path-derived
hints.

## `grep_log`

Input:

```json
{
  "pattern": "Traceback|RuntimeError",
  "ignore_case": true,
  "max_matches": 50,
  "result_mode": "compact"
}
```

Purpose: search the complete immutable source snapshot while preserving
original line numbers. Compact mode reuses L0 normalized occurrence groups
inside distributed-incident boundaries. Unclassified matches retain the
existing incident-local normalized compaction or remain individual lines. Raw
mode returns individual matching lines. Under a response limit, L0 error/cause
groups are retained before diagnostic and unclassified samples while scan
totals still describe the complete source boundary.

Example output:

```json
{
  "pattern": "Traceback|RuntimeError",
  "result_mode": "compact",
  "matches": [
    {
      "line": 1174,
      "text": "...",
      "group_kind": "normalized_occurrence_group",
      "incident_id": "di-2",
      "occurrence_group_id": "og-12",
      "classification": "error",
      "registry_id": "observed_distributed_operation_timeout",
      "occurrence_count": 12000,
      "occurrence_group_total_count": 12000,
      "distinct_rank_count": 12000,
      "first_line": 1174,
      "last_line": 18990
    }
  ],
  "total_raw_matches": 12000,
  "total_match_groups": 1,
  "collapsed_matches": 11999,
  "scan_complete": true,
  "samples_truncated": false,
  "initial_view_overlap": {
    "available": true,
    "matched_group_count": 1,
    "represented_group_count": 1,
    "new_group_count": 0,
    "new_evidence_beyond_initial_view": false
  }
}
```

`result_mode` defaults to `compact` and may be set to `raw`. In compact mode,
`max_matches` limits representative groups; in raw mode, it limits individual
lines. `scan_complete` states whether the entire immutable source boundary was
searched; `samples_truncated` states whether matched groups or lines were
omitted from the returned sample. The outer result-envelope `truncated` mirrors
sample truncation, not scan completion. `initial_view_overlap` reports whether
the search found normalized evidence beyond the initial L0B model view.

## `read_window`

Input:

```json
{
  "center_line": 1174,
  "before": 20,
  "after": 80
}
```

Purpose: retrieve original raw lines around a selected source location.

Example output:

```json
{
  "start_line": 1154,
  "end_line": 1254,
  "lines": [{"line": 1174, "text": "..."}],
  "truncated": false
}
```

The client bounds before/after context, total lines, and serialized characters.
The result should be large enough to show progress before a failure and
cascade/recovery after it without becoming an unbounded log dump.

If the range is unavailable because of source truncation, overwrite,
progressive eviction, or a configured cap, the tool returns a deterministic
non-crashing result:

```json
{
  "start_line": 1154,
  "end_line": 1254,
  "lines": [],
  "truncated": true,
  "error": "window_unavailable",
  "unavailable_reason": "progressive_window_evicted",
  "candidate_summary_refs": ["cand-17"]
}
```

Retained candidate references may orient the model but are never rendered as
fabricated raw lines.

## `get_evidence_objects`

Status: implemented, disabled in the default advertisement.

Input:

```json
{
  "refs": ["fe-1", "w-3", "og-4"]
}
```

Purpose: resolve attempt-scoped L0A object identifiers for occurrence groups,
windows, anchors, episodes, distributed incidents, and progress/setup markers
without rescanning the source log.

Example output:

```json
{
  "schema_version": "restart_agent_evidence_objects.v1",
  "requested_refs": ["fe-1", "w-3", "og-4"],
  "objects": [
    {
      "ref": "fe-1",
      "object_type": "failure_episode",
      "payload": {"episode_id": "fe-1", "start_line": 1000},
      "truncated": false
    }
  ],
  "missing_refs": [],
  "invalid_refs": [],
  "omitted_refs": [],
  "limits": {"max_refs": 8, "max_chars": 50000},
  "truncated": false
}
```

Missing, invalid, omitted, and truncated objects are represented explicitly.

## Error And Deadline Behavior

- Tool execution checks the remaining whole-analysis deadline.
- Tool calls do not begin after that deadline.
- Invalid arguments and unadvertised tools produce structured failures rather
  than exceptions escaping the route.
- Tool errors do not become evidence.
- Exhausting the route's tool-round limit makes L1 degraded if no valid final
  evidence was produced; the deterministic recommendation remains available.
- Tool result truncation is visible to the model and trace.

## Observability

Each accepted or rejected request records:

- tool-call id;
- route and model-turn ids;
- phase;
- tool name;
- redacted argument summary and normalized argument hash;
- start/end time and latency;
- visible source offset or line range;
- returned lines/characters and newly visible line ids;
- match count and truncation;
- effective caps and caps hit;
- timeout, execution error, or rejection reason.

Aggregates include:

- call count and extra model turns;
- tool latency;
- error and unsupported-request rates;
- truncation rate;
- duplicate-call count;
- no-new-line count;
- new decision-relevant context yield;
- incremental token cost.

Interpretation belongs in evaluation:

- new decision-relevant evidence absent from L0B suggests a projection gap;
- rereading lines already visible in L0B suggests model/route inefficiency;
- the same missing context requested by many models is a high-priority L0B
  improvement signal;
- one model overusing tools is not evidence that L0B is defective.

Tool metrics never directly select a policy action.
