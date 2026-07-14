# Restart Agent Evaluation Harness

This is offline developer and QA tooling for qualifying the NVRx Restart Agent.
It does not implement log analysis or ship in the NVRx package. Every review
invokes the product checkout under test, then scores and renders its artifacts.

The harness exists to make analyzer tuning measurable. It separates:

- product correctness by stage;
- model semantic quality;
- model/tool behavioral efficiency;
- endpoint reliability and latency;
- repeated-run decision stability.

## Read In This Order

1. [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) - use cases and qualification requirements.
2. [docs/DESIGN.md](docs/DESIGN.md) - product/harness boundary and execution flow.
3. [docs/SCHEMA.md](docs/SCHEMA.md) - human gold, run, score, and report contracts.
4. [docs/TUNING.md](docs/TUNING.md) - one-log investigation through holdout promotion.
5. [docs/PANEL_SUMMARY.md](docs/PANEL_SUMMARY.md) - reviewer-facing model comparison.

[docs/STABILITY.md](docs/STABILITY.md) defines repeated-run metrics,
[docs/STATUS.md](docs/STATUS.md) records open work, and
[docs/REVIEW_PANEL.md](docs/REVIEW_PANEL.md) defines technical-document review.
Deck source notes live under `docs/internal/`; they are explanatory material,
not contracts.

## Architecture

```text
source log + optional human gold
  -> product collect_all (shared L0, parallel model routes)
  -> canonical product results and traces
  -> per-route stage scoring and review
  -> panel summary / corpus aggregate / stability report
```

`docs/REQUIREMENTS.md` owns the three-root isolation rule; `docs/SCHEMA.md` owns
the exact mirrored paths. This README keeps the commands needed to operate that
layout. Gold is never sent to the product or model.

## Code Map

```text
src/restart_agent_eval/
  corpus.py          source-log/gold discovery
  gold.py            strict label validation
  artifacts.py       mirrored roots and run placement
  profiles.py        named model-panel profiles
  product_contract.py versioned product CLI payloads
  product_process.py product command/process adapter
  product_trace.py   typed product trace boundary
  repository_identity.py repository provenance adapter
  runtime.py         injectable clock and sleep boundaries
  artifact_io.py     local artifact storage adapter
  schemas.py         harness-owned artifact versions
  review_context.py  normalized product-stage inputs for review
  behavior.py        deterministic golden fixture capture/check
  behavior_worker.py isolated product-import worker
  review.py          one-log execution and review publication
  scoring.py         gold scoring and runtime KPI derivation
  review_markdown.py per-route Markdown rendering
  evaluate.py        corpus execution and aggregate scoring
  panel.py           panel normalization and concern derivation
  panel_*markdown.py compact and diagnostic panel rendering
  inspect.py         L0A/DecisionEvidence/L0B inspection
  stability.py       repeated-run stability analysis
src/*.py             thin command wrappers
```

## Setup

From the repository root:

```bash
cd tools/restart_agent_eval
cp restart_agent.env.example restart_agent.env
# Edit local paths, then load the untracked environment file.
source restart_agent.env
```

The equivalent explicit environment is:

```bash
export RESTART_AGENT_EVAL_LOG_ROOT=/abs/path/to/logs
export RESTART_AGENT_EVAL_GOLD_ROOT=/abs/path/to/restart_agent_gold
export RESTART_AGENT_EVAL_RUN_ROOT=/abs/path/to/restart_agent_runs
export NVRX_RESTART_AGENT_PRODUCT_REPO=/abs/path/to/nvidia-resiliency-ext
```

Model-backed runs use environment-only credential slots:

```bash
export LLM_API_KEY_FILE=/secure/path/to/llm_api_key
export LLM_API_KEY_OLD_FILE=/secure/path/to/llm_api_key_old
export NVRX_LLM_BASE_URL=https://inference-api.nvidia.com/v1
```

Export-controlled workloads must use an authorized ECCN-compliant route on the
Regulated Inference Hub. Secrets and key-file paths are not stored in profiles,
gold, or generated artifacts.

## Review One Log

```bash
./examples/review_one_log.sh \
  "$RESTART_AGENT_EVAL_LOG_ROOT/checkpoint_logs/input.log" \
  models
```

Reuse an L0 bundle from an earlier run when the source log is unchanged:

```bash
./examples/review_one_log.sh \
  "$RESTART_AGENT_EVAL_LOG_ROOT/checkpoint_logs/input.log" \
  --l0-bundle-json-in /path/to/prior/run/l0_bundle.json \
  all
```

Replay is explicit. The product rejects a bundle when its schema, exact source
path, byte size, or source-file `mtime` no longer matches.

Targets include `deterministic`, `configured`, `qwen`, `qwen235b`, `qwen397b`,
`nemotron`, `gpt`, `claude`, `gemini`, `models`, and `all`. Multi-model review
uses product `collect_all`: L0 is built once and replayed identically to every
route. Shared L0 artifacts appear when L0 completes; each route result, trace,
and review appears when that route completes; panel artifacts appear after the
batch closes.

Open `review_index.md` first. It is a navigation page rather than another KPI
report. Use `panel_summary.md` for comparison and a model's `review.md` for one
route. The first substantive section of that route review is the complete
parsed L1 model answer. The adjacent `result.json` is the final composed product
result, while `trace.json` is reserved for raw responses, tool interaction, and
stage diagnostics.

Inspect evidence or rebuild a panel without model calls:

```bash
python3 src/inspect_trace.py /path/to/model.trace.json --view comparison
python3 src/summarize_review_panel.py /path/to/completed/run
```

## Score The Corpus

```bash
python3 src/eval_harness.py \
  --target deterministic \
  --log-root "$RESTART_AGENT_EVAL_LOG_ROOT" \
  --gold-root "$RESTART_AGENT_EVAL_GOLD_ROOT" \
  --run-root "$RESTART_AGENT_EVAL_RUN_ROOT"
```

The only scored corpus format is mirrored `gold.json`. There is no embedded
oracle or alternate YAML case path. A scored label must have an explicit
human-review status and a `source_sha256` matching the source log bytes.

## Check Deterministic Behavior

Behavior fixtures characterize product L0, DecisionEvidence, L0B, fallback
result, and deterministic trace for every gold case:

```bash
python3 src/capture_behavior_fixtures.py \
  --log-root "$RESTART_AGENT_EVAL_LOG_ROOT" \
  --gold-root "$RESTART_AGENT_EVAL_GOLD_ROOT" \
  --product-repo "$NVRX_RESTART_AGENT_PRODUCT_REPO" \
  --check
```

Regenerate fixtures only after reviewing an intended product behavior change.

## Measure Stability

```bash
./examples/summarize_decision_stability.sh \
  --runs-root "$RESTART_AGENT_EVAL_RUN_ROOT/path/to/input.log" \
  --route qwen397b \
  --minimum-samples 10
```

Stability is measured separately from gold-scored correctness. A consistently
wrong answer is stable, not accurate.

## Run The Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The module ownership and placement rules are documented in
[`tests/README.md`](tests/README.md).
