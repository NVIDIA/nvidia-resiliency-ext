# Test Layout

The test suite follows production-module ownership. Keep behavior in the narrowest matching
module; reserve cross-module tests for `test_boundaries.py` and the review workflow tests.

The suite intentionally uses a flat directory. Responsibility prefixes provide the hierarchy
without adding package, import, and fixture boundaries:

- `test_product_*` covers the product process and payload boundary.
- `test_review_*` covers review orchestration and artifact publication.
- `test_panel_*` covers cross-route aggregation and presentation.
- `test_scoring_*` covers stage-specific evaluation behavior.
- `test_gold_*` covers corpus-label storage and validation.

| Area | Test modules |
|---|---|
| Artifact primitives and run layout | `test_artifact_io.py`, `test_artifacts.py`, `test_schemas.py` |
| Product process and payload adapters | `test_product_process.py`, `test_product_contract.py`, `test_product_trace.py` |
| Corpus labels and evaluation | `test_corpus.py`, `test_gold_schema.py`, `test_gold_reader.py`, `test_evaluate_discovery.py`, `test_evaluate_execution.py`, `test_evaluate_scoring.py` |
| Trace inspection and redaction | `test_inspect.py`, `test_redaction.py` |
| Profiles and route planning | `test_profiles.py`, `test_review_route_planning.py` |
| Review execution and publication | `test_review_application.py`, `test_review_execution.py`, `test_review_artifacts.py`, `test_review_context.py`, `test_review_live_progress.py`, `test_review_process_lifecycle.py` |
| Panel aggregation and rendering | `test_panel_aggregation.py`, `test_panel_cli.py`, `test_panel_rendering.py`, `test_review_markdown.py` |
| L0, L1, L2, and L4 scoring | `test_scoring_boundaries.py`, `test_scoring_l0.py`, `test_scoring_semantics.py`, `test_scoring_audit.py`, `test_scoring_policy.py` |
| Endpoint and final-review scoring | `test_scoring_endpoint.py`, `test_review_scoring.py` |
| Tool-efficiency scoring | `test_scoring_tool_efficiency.py` |
| Behavior fixtures and repository identity | `test_behavior.py`, `test_behavior_worker.py`, `test_repository_identity.py` |
| Stability, runtime, and boundaries | `test_stability.py`, `test_runtime.py`, `test_boundaries.py` |

The module-by-module coverage and residual integration inventory is in
[`TEST_COVERAGE.md`](TEST_COVERAGE.md).

Within a module, tests are grouped by the production function they exercise. Closely related happy,
error, and edge cases remain together so the behavioral contract can be read in one place.

Shared test code is separated by responsibility:

- `_bootstrap.py` and `conftest.py` provide the same source-tree import setup to unittest and pytest.
- `_builders.py` creates repeated contract-shaped policy and recovery payloads.
- `_assertions.py` validates structured fields, JSON artifacts, and artifact paths with focused diagnostics.
- `_mocks.py` owns deterministic process-boundary fakes.
- `_panel_fixtures.py`, `_evaluation_fixtures.py`, and `_review_execution_fixtures.py` own
  domain-specific test data and process scaffolding.

Helpers must not hide assertions or business rules. Reusable trace and process layouts belong in the
matching fixture module; one-off scenario data stays beside the test that explains it.

Tests exercise public composition roots, artifact APIs, or typed business-rule boundaries. They do
not call module-private production functions. Exact collaborator calls are asserted only for
adapters whose contract is delegation, such as subprocess execution, clocks, and sleeping.
Cross-module behavior uses published artifacts or typed boundary objects.

## Test Method Structure

Each test method follows a visible precondition, execution, and verification sequence. Blank lines
separate these phases; `# Arrange`, `# Act`, and `# Assert` comments are not required.

- Exercise one public behavior or failure contract per test method.
- Assign the behavior result to a descriptive local such as `actual`, `result`, or `summary` before
  asserting on it.
- Use `subTest` only for input variations that share the same contract and expected result shape.
- Setup helpers may construct fixtures and inputs, but must not execute the behavior or hide
  assertions unless their name explicitly describes an orchestration action.

## When To Add Directories

Keep the suite flat while filenames identify ownership unambiguously. Introduce a subsystem
directory only when that subsystem has its own fixture/configuration boundary or grows to roughly
five to eight substantial test modules. Move the whole subsystem in one change; do not maintain a
mixed layout where tests for the same responsibility are split between the root and a subdirectory.

Until then, run a responsibility group directly by filename prefix, for example:

```bash
pytest -q tools/restart_agent_eval/tests/test_scoring_*.py
```

Prefer behavior-driven names such as `test_should_publish_route_artifacts_when_collect_all_finishes`.
Assert exception types and observable side effects; pin exact error text only when that text is a
documented CLI contract.

Run the suite from the repository root:

```bash
python3 -m unittest discover -s tools/restart_agent_eval/tests -v
```

Pytest uses the same bootstrap through `conftest.py`:

```bash
pytest -q tools/restart_agent_eval/tests
```
