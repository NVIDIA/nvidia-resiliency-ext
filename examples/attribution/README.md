# NVRx Attribution Examples

These examples cover the existing attribution MCP integration and the Restart
Agent CLI configuration.

Install NVIDIA Resiliency Extension with the optional attribution dependencies
when running the MCP example, then run the examples from the repository root:

```bash
pip install 'nvidia-resiliency-ext[attribution]'
```

| File | Description |
|------|-------------|
| `single_server_example.py` | Single MCP server with multiple attribution modules (run from repo root). |
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
