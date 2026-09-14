# NVRx Attribution Examples

These examples cover the Restart Agent CLI configuration, packaged
restart-agent/FR MCP integration, and legacy source-checkout LogSage notes.

The Restart Agent CLI and attrsvc paths are shipped in the main wheel:

```bash
pip install nvidia-resiliency-ext
```

Packaged MCP support for restart-agent log analysis and Flight Recorder analysis
is available from the attribution extra:

```bash
pip install 'nvidia-resiliency-ext[attribution]'
python -m nvidia_resiliency_ext.attribution.mcp_integration.server_launcher \
  --modules restart_agent fr_analyzer
```

Legacy LogSage implementations and their MCP tools are not shipped in the main
wheel, and there is no wheel extra for them. Source-checkout workflows that use
the legacy LogSage-backed MCP tools (`log_analyzer`, `log_fr_analyzer`) can
install the source-only dependencies with:

```bash
python -m pip install -r src/nvidia_resiliency_ext/attribution/legacy_logsage/requirements.txt
```

Then launch the server with `--enable-legacy-logsage`.

| File | Description |
|------|-------------|
| `single_server_example.py` | Source-checkout single MCP server with packaged restart-agent plus FR tools (run from repo root). |
| `restart_agent.json` | Minimal `restart_agent_config.v1` configuration with one Nemotron model route. |

The Restart Agent configuration contains no credential path or key material. It
omits `base_url` and therefore uses the product's default provider endpoint.
Its `credential_ref` names `LLM_API_KEY_FILE`; set that environment variable to
an authorized, readable API-key file, then run:

```bash
python3 -m nvidia_resiliency_ext.attribution.restart_agent.cli \
  /absolute/path/to/cycle.log \
  --config examples/attribution/restart_agent.json
```

See the complete
[Restart Agent configuration reference](../../docs/design/attribution/restart_agent/CONFIGURATION.md)
for every field, default, and constraint.

For the runnable HTTP services (**nvrx-attrsvc**, **nvrx-smonsvc**), install and
run from [services/](../../services/); see
[services/README.md](../../services/README.md).
