# NVRX Attribution Examples

Minimal examples using the attribution library and MCP integration.

Install the optional attribution dependencies first:

```bash
pip install 'nvidia-resiliency-ext[attribution]'
```

| File | Description |
|------|-------------|
| `single_server_example.py` | Single MCP server with multiple attribution modules (run from repo root). |
| `restart_agent.json` | Restart-agent configuration with eight model routes: Qwen 235B with/without tools, Qwen 397B with tools, Qwen 3.5 without tools, plus Nemotron, GPT, Claude, and Gemini. |

The restart-agent config contains no credentials. Set each `credential_ref`
environment variable to its authorized key-file path, then run:

```bash
PYTHONPATH=src python3 -m nvidia_resiliency_ext.attribution.restart_agent.cli \
  /absolute/path/to/cycle.log \
  --config examples/attribution/restart_agent.json
```

For the runnable HTTP services (**nvrx-attrsvc**, **nvrx-smonsvc**), install and run from [services/](../../services/) — see [services/README.md](../../services/README.md).
