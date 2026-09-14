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

## Executable Contract Registry

One in-code registry is the source of truth for each tool's advertised JSON
schema, runtime validation, implementation dispatch, result requirements, and
limits. Route advertisement selects entries from that registry; it does not
redefine them.

Arguments are validated without coercion. For example, `"false"` is not a
Boolean and `"20"` is not an integer. Unknown fields are rejected. Every tool
response uses this envelope:

```json
{
  "schema_version": "restart_agent_tool_result.v1",
  "tool": "grep_log",
  "status": "ok",
  "data": {},
  "error": null,
  "truncated": false,
  "limits": {}
}
```

For an error, `status` is `error`, `data` is null, and `error` contains
`code`, optional `field`, and a bounded message. The closed error codes are:

| Code | Meaning |
| --- | --- |
| `malformed_arguments_json` | Arguments are not valid JSON. |
| `invalid_arguments` | The decoded object violates the tool schema. |
| `invalid_regex` | `grep_log.pattern` is not a valid Python regular expression. |
| `tool_not_advertised` | The route did not advertise the requested tool. |
| `tool_not_implemented` | No executable registry entry exists. |
| `source_unavailable` | The immutable attempt source cannot be read. |
| `line_out_of_range` | `read_window.center_line` is outside the source snapshot. |
| `internal_tool_error` | An unexpected read or serialization failure occurred. |

Tool-name rejection uses advertisement-first precedence. A name outside the
route's advertised set returns `tool_not_advertised`, whether or not an
implementation exists. `tool_not_implemented` is reserved for the defensive
case where an advertised name has no executable registry entry; configuration
validation normally prevents that state.

Failed envelopes are visible in the interaction transcript but are not source
evidence and do not expand L2 grounding visibility.

Every per-tool block labeled `Example data` shows only the contents of this
common envelope's `data` member. A payload-specific `schema_version` inside
`data` does not replace the outer `restart_agent_tool_result.v1` envelope.

## Advertised Set

The default route configuration advertises:

- `grep_log`;
- `read_window`;
- `get_evidence_objects`.

`overview` is implemented but not advertised by default. L0B already provides
initial orientation, so a route enables `overview` only for controlled
evaluation or a client that does not supply the normal L0B view.

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
| `read_window.max_lines` | 241 |
| `read_window.max_chars` | 50000 |
| `get_evidence_objects.max_refs` | 8 |
| `get_evidence_objects.max_chars` | 50000 |

Every successful response reports truncation and relevant applied limits. A cap never
silently changes source-line numbering.

## `overview`

Input: an object with no fields (`{}`).

Purpose: orient the model to file scale, bounded head/tail content, and the
existing deterministic evidence without recomputing L0. This tool is opt-in
because the normal L0B request already carries that orientation.

Example `data`:

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

The output excludes the request's source-log path, its location components,
eval labels, and hints inferred from that source location. Workload artifact
paths present in returned log evidence are preserved.

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

`pattern` is a non-empty Python regular expression of at most 4096 characters.
`ignore_case` is a Boolean and defaults to `true`. `max_matches` is an integer
from 0 through 200. When omitted, `max_matches` defaults to 50; 50 is not an
additional cap. An explicit value in the supported range is honored exactly.
`result_mode` is `compact` by default and may be set to `raw`.

Purpose: search the complete immutable source snapshot while preserving
original line numbers. Compact mode reuses L0 normalized occurrence groups and
distributed-incident boundaries. Matching copies of an L0 occurrence group are
returned once per incident with the L0 group identity, classification, registry
provenance, occurrence count, and bounded source samples. Matches that are not
recognized by L0 still use normalized-message compaction inside a distributed
incident or remain individual matches. Raw mode returns individual matching
lines without compaction. When the response limit cannot retain every compact
match, failure-relevant L0 error/cause groups are selected before diagnostic and
unclassified matches; source-scan totals still cover every match.

Example `data`:

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
      "normalized_shape": "nccl_collective_timeout",
      "classification": "error",
      "registry_id": "observed_distributed_operation_timeout",
      "occurrence_count": 12000,
      "occurrence_group_total_count": 12000,
      "occurrence_group_distinct_rank_count": 12000,
      "distinct_rank_count": 12000,
      "unattributed_occurrence_count": 0,
      "first_line": 1174,
      "last_line": 18990,
      "sample_lines": [1174, 1175, 1176],
      "sample_ranks": ["0", "1", "2", "3", "4"]
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

Requests beyond the hard limit of 200 are rejected rather than clamped. Results
that exceed the effective requested match count truncate only the returned
samples. In compact mode, `max_matches` limits representative groups; in raw
mode, it limits individual lines. `scan_complete=true` means the regex was
evaluated against the entire immutable source boundary. `samples_truncated`
means at least one matched group or raw line was omitted from the response. The
outer envelope's `truncated` field mirrors `samples_truncated`; it does not mean
the source scan stopped early. A fully represented fanout therefore reports
`samples_truncated=false` even when `collapsed_matches` is large.

`initial_view_overlap` compares every matched compact group, including groups
omitted by the response limit, with the evidence initially rendered in L0B.
Occurrence-group and distributed-incident identity take precedence over exact
sample-line overlap. In raw mode the comparison is line-based. This makes
duplicate/no-new tool use observable without preventing the model from reading
additional raw context.

## `read_window`

Input:

```json
{
  "center_line": 1174,
  "before": 20,
  "after": 80
}
```

`center_line` is an integer from 1 through the immutable snapshot's line count.
`before` and `after` are integers from 0 through 120 and default to 20 and 80;
the largest symmetric request contains 241 lines: 120 before, the center line,
and 120 after.

Purpose: retrieve original raw lines around a selected source location.

Example `data`:

```json
{
  "start_line": 1154,
  "end_line": 1254,
  "lines": [{"line": 1174, "text": "..."}],
  "truncated": false
}
```

The client bounds total lines and serialized characters.
The result should be large enough to show progress before a failure and
cascade/recovery after it without becoming an unbounded log dump.

An out-of-range center is rejected with the common error envelope:

```json
{
  "schema_version": "restart_agent_tool_result.v1",
  "tool": "read_window",
  "status": "error",
  "data": null,
  "error": {
    "code": "line_out_of_range",
    "field": "center_line",
    "message": "center_line is outside the immutable source snapshot."
  },
  "truncated": false,
  "limits": {}
}
```

## `get_evidence_objects`

Status: implemented and advertised by default.

Input:

```json
{
  "refs": ["fe-1", "w-3", "og-4"]
}
```

`refs` contains 1 through 8 unique, non-empty strings, each no longer than 128
characters.

Purpose: expand object IDs listed in
`decision_evidence_view.selected_evidence_references` into their structured
current-log payloads. Use this before `grep_log` when the needed evidence may
already be represented by those IDs. Pass only IDs from the `*_ids` fields; do
not convert `source_lines` into `line-*` IDs. The tool does not read external
files or search arbitrary log text.

Example `data`:

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
- Exhausting the route's tool-round limit triggers one forced no-tool final
  evidence response with reason `forced_final_after_tool_exhaustion`. A
  contract-valid response remains usable-but-degraded. If that final response
  does not produce contract-valid evidence, L1 is unusable. This is distinct
  from `contract_repair`; the deterministic recommendation remains available in
  either case.
- Tool result truncation is visible to the model and trace.

## Observability

Each accepted or rejected request records:

- tool-call id;
- model-turn id (the containing route trace supplies route identity);
- tool name;
- bounded argument summary;
- latency;
- serialized result characters and returned line count;
- match count and truncation;
- result status and closed error/rejection code.

The transcript preserves the exact structured envelope, including applied
limits and any returned source-line range. Evaluation derives duplicate calls,
newly visible lines, and decision-context yield from that transcript.

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
