# Production Test Coverage

This inventory maps every `restart_agent_eval` production module to its behavioral tests. The suite
covers empty and malformed inputs, process failures, timeouts, configuration variants, artifact
publication, and simulated concurrent route publication. No production module is wholly untested.

## Module Inventory

| Production module | Test ownership | Covered failure and boundary cases |
|---|---|---|
| `__init__.py` | `test_schemas.py` | Public schema exports remain importable. |
| `artifact_io.py` | `test_artifact_io.py` | Missing/malformed JSON, nested writes, failed atomic replace, temporary-file cleanup. |
| `artifacts.py` | `test_artifacts.py` | Explicit/default roots, missing environment, invalid ancestry/equality, optional gold root. |
| `behavior.py` | `test_behavior.py` | Worker stderr/stdout failures, malformed/non-object output, missing product source, empty corpus, missing source log, fixture write/check/diff. |
| `behavior_worker.py` | `test_behavior_worker.py`, `test_behavior.py` | CLI success delegation and propagated worker failure. |
| `corpus.py` | `test_corpus.py`, `test_evaluate_discovery.py` | Private subtrees, absent logs, malformed/non-object labels, invalid decisions, legacy/default fields, hash mismatch. |
| `evaluate.py` | `test_evaluate_discovery.py`, `test_evaluate_scoring.py`, `test_evaluate_execution.py` | Empty corpus, unavailable log, analyzer failure, result cardinality, malformed result, optional trace, mismatch exit, aggregate empty rates. |
| `gold.py` | `test_gold_reader.py`, `test_gold_schema.py` | Required fields, review status, digest syntax/mismatch, malformed nested expectations and unsupported claims. |
| `inspect.py` | `test_inspect.py` | Every view, snapshots, missing views/files, malformed JSON, non-object traces, invalid snapshot values. |
| `panel.py` | `test_panel_aggregation.py`, `test_panel_cli.py`, `test_panel_rendering.py` | Empty/missing run, malformed reviews, route disagreement, missing routes, endpoint/tool concerns, fallback/enriched policy, shared L0 consistency. |
| `panel_diagnostics_markdown.py` | `test_panel_rendering.py` | Rich diagnostics, distributed incidents, L0/L1/L2/L3/L4 sections and optional data. |
| `panel_format.py` | `test_panel_rendering.py`, `test_panel_cli.py` | Empty values, labels, table-safe rendering through public panel output. |
| `panel_summary_markdown.py` | `test_panel_rendering.py` | Summary hierarchy, progressive latency, semantic and operational comparison, empty/partial routes. |
| `paths.py` | `test_paths.py`, `test_artifacts.py` | Missing and present environment values, default paths. |
| `product_contract.py` | `test_product_contract.py` | Collect-all config and route artifact manifest shape. |
| `product_process.py` | `test_product_process.py`, `test_review_process_lifecycle.py` | Success, nonzero exit, timeout, poll/wait, terminate/kill, command and environment propagation. |
| `product_trace.py` | `test_product_trace.py`, `test_review_scoring.py` | Supported schemas, malformed/non-object fields, file parsing, decision candidate extraction. |
| `profiles.py` | `test_profiles.py` | Empty targets, aliases, all-model expansion, ordering/deduplication, unknown target. |
| `repository_identity.py` | `test_repository_identity.py` | No repository, clean/dirty repository, failed Git commands. |
| `review.py` | `test_review_application.py`, `test_review_execution.py`, `test_review_artifacts.py`, `test_review_live_progress.py`, `test_review_process_lifecycle.py`, `test_review_route_planning.py` | Invalid log/repository, route config variations, missing credentials, product failure, incremental publication, timeout cleanup, malformed live events. |
| `review_context.py` | `test_review_context.py` | Complete and sparse traces, selected-analysis fallback, malformed optional fields, injected artifact store. |
| `review_markdown.py` | `test_review_markdown.py` | Minimal and rich summaries, stage KPIs, gold results, path leakage, provider errors, failure identity. |
| `runtime.py` | `test_runtime.py` | Clock and sleeper adapters. |
| `schemas.py` | `test_schemas.py`, `test_product_trace.py`, `test_panel_cli.py` | Correct, missing, and mismatched schema versions. |
| `scoring.py` | `test_scoring_*.py`, `test_redaction.py`, `test_review_scoring.py` | L0/L1/L2/L3/L4, gold and no-gold, malformed/empty inputs, policy paths, endpoint failures, token budgets, tools, redaction, mixed line endings. |
| `stability.py` | `test_stability.py` | Stable/unstable cohorts, insufficient samples, endpoint failures, input changes, dirty products, malformed runs, invalid discovery, unmatched routes, report publication. |

## Residual Integration And Performance Cases

These cases require the real product, operating system, or inference service and should not be
represented as passing unit tests with mocks:

| Case | Why it remains external | Intended test level |
|---|---|---|
| Successful `build_fixture_in_worker` against the installed Restart Agent | Its contract is the exact product import and full deterministic product payload. The checked gold behavior fixtures are the assertion. | Product/harness compatibility job using `capture_behavior_fixtures.py --check`. |
| Live inference authentication, HTTP retry, provider timeout, and model tool loop | A fake process validates harness lifecycle but cannot establish provider behavior. | Credentialed endpoint smoke test and repeated-run stability evaluation. |
| Real multi-route scheduling races | Unit tests deterministically interleave publication, polling, deadline, terminate, and kill behavior; they do not stress OS scheduling. | Repeated collect-all integration run under load. |
| Filesystem permission denial and storage exhaustion | Tests cover propagated `OSError` and atomic cleanup; chmod and ENOSPC behavior is platform/filesystem dependent. | Containerized filesystem fault-injection test. |
| Very large log/corpus latency and memory use | There is no semantic max-size boundary; the requirement is operational rather than a different return value. | Benchmark with representative multi-GB logs and corpus fanout. |

## Coverage Gate

The current suite contains 222 tests and 157 parameterized subtests. Local branch coverage is 86%
overall. Lower percentages in renderer and scoring modules represent optional presentation and
cross-product combinations; behavioral contracts are tested through public APIs rather than by
calling private formatting helpers.
