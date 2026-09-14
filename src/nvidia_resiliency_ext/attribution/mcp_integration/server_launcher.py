#!/usr/bin/env python3
"""Launcher for the NVRX Attribution MCP server.

Packaged installs register ``restart_agent`` and ``fr_analyzer`` by default via
``nvidia-resiliency-ext[attribution]``. Legacy LogSage-backed MCP tools are
source-checkout-only; launch with ``--enable-legacy-logsage`` after installing
``src/nvidia_resiliency_ext/attribution/legacy_logsage/requirements.txt``.
The process title is set to ``nvrx-mcp-analysis`` via ``setproctitle`` when available.

Usage:
    # Launch packaged restart-agent and FR MCP
    python -m nvidia_resiliency_ext.attribution.mcp_integration.server_launcher

    # Launch packaged restart-agent and FR MCP explicitly
    python -m nvidia_resiliency_ext.attribution.mcp_integration.server_launcher \
        --modules restart_agent fr_analyzer

    # Launch source-checkout-only legacy LogSage MCP tools too
    python -m nvidia_resiliency_ext.attribution.mcp_integration.server_launcher \
        --enable-legacy-logsage --modules restart_agent log_analyzer fr_analyzer

    # Launch with custom server name
    python -m nvidia_resiliency_ext.attribution.mcp_integration.server_launcher \
        --server-name my-attribution-server
"""

import argparse
import logging
import sys

_PROC_TITLE = "nvrx-mcp-analysis"


def _set_process_title(title: str) -> None:
    """Set argv-style process name for ps/top (Linux/macOS; no-op if setproctitle unavailable)."""
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle(title)
    except Exception as exc:
        logging.getLogger(__name__).debug("Process title was not set: %s", exc)


logger = logging.getLogger(__name__)

_LEGACY_LOGSAGE_MODULES = {
    "log_analyzer",
    "log_analyzer_progressive_start",
    "log_fr_analyzer",
}

_MCP_EXTRA_MESSAGE = (
    "Attribution MCP dependencies are not installed. "
    "Install with: pip install 'nvidia-resiliency-ext[attribution]'"
)

_LEGACY_LOGSAGE_MESSAGE = (
    "Legacy LogSage MCP tools are source-checkout-only. "
    "Run from a source checkout with --enable-legacy-logsage and install "
    "src/nvidia_resiliency_ext/attribution/legacy_logsage/requirements.txt."
)


def _is_mcp_dependency(exc: ModuleNotFoundError) -> bool:
    return bool(exc.name and (exc.name == "mcp" or exc.name.startswith("mcp.")))


def _is_legacy_logsage_dependency(exc: ModuleNotFoundError) -> bool:
    return bool(
        exc.name
        and (
            exc.name in {"langchain_core", "langchain_openai", "logsage"}
            or exc.name.startswith("logsage.")
            or exc.name.startswith("nvidia_resiliency_ext.attribution.legacy_logsage")
        )
    )


def _validate_requested_modules(
    requested_modules: list[str] | None, available_modules: list[str]
) -> None:
    if not requested_modules:
        return

    missing = sorted(set(requested_modules) - set(available_modules))
    if not missing:
        return

    message = f"Requested module(s) not registered: {', '.join(missing)}."
    if set(missing) & _LEGACY_LOGSAGE_MODULES:
        message += " Use --enable-legacy-logsage from a source checkout for LogSage tools."
    raise SystemExit(message)


def main():
    """Main entry point for the MCP server."""
    _set_process_title(_PROC_TITLE)

    try:
        from nvidia_resiliency_ext.attribution.mcp_integration.mcp_server import NVRxMCPServer
    except ModuleNotFoundError as exc:
        if not _is_mcp_dependency(exc):
            raise
        raise SystemExit(_MCP_EXTRA_MESSAGE) from exc

    from nvidia_resiliency_ext.attribution.mcp_integration.module_definitions import (
        register_all_modules,
    )
    from nvidia_resiliency_ext.attribution.mcp_integration.registry import global_registry

    parser = argparse.ArgumentParser(description='Launch NVRX Attribution MCP Server')
    parser.add_argument(
        '--server-name', default='nvidia-resiliency-attribution', help='Name of the MCP server'
    )
    parser.add_argument('--modules', nargs='*', help='Specific modules to enable (default: all)')
    parser.add_argument(
        '--enable-legacy-logsage',
        action='store_true',
        help='Register source-checkout-only LogSage-backed tools',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level',
    )

    args = parser.parse_args()

    _level = getattr(logging, args.log_level)
    if not logging.root.handlers:
        logging.basicConfig(
            level=_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
    else:
        logging.getLogger("nvidia_resiliency_ext").setLevel(_level)

    # Register all modules
    logger.info("Registering attribution modules...")
    register_all_modules()
    if args.enable_legacy_logsage:
        try:
            from nvidia_resiliency_ext.attribution.legacy_logsage.log_analyzer.mcp_module_definitions import (
                register_legacy_logsage_mcp_modules,
            )
        except ModuleNotFoundError as exc:
            if not _is_legacy_logsage_dependency(exc):
                raise
            raise SystemExit(_LEGACY_LOGSAGE_MESSAGE) from exc

        register_legacy_logsage_mcp_modules()

    # Filter modules if specified
    if args.modules:
        all_modules = global_registry.list_modules()
        _validate_requested_modules(args.modules, all_modules)
        for module in list(all_modules):
            if module not in args.modules:
                global_registry.unregister(module)
                logger.info(f"Unregistered module: {module}")
        logger.info(f"Enabled modules: {args.modules}")
    else:
        logger.info(f"Enabled modules: {global_registry.list_modules()}")

    # Create and run server
    server = NVRxMCPServer(registry=global_registry, server_name=args.server_name)

    logger.info(f"Starting server: {args.server_name}")

    try:
        server.run_sync()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
