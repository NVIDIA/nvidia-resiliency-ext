# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
import types
import unittest
from contextlib import asynccontextmanager
from dataclasses import dataclass

PY310_PLUS = sys.version_info >= (3, 10)


def _install_mcp_stub() -> None:
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    stdio_mod = types.ModuleType("mcp.server.stdio")
    types_mod = types.ModuleType("mcp.types")

    class Server:
        def __init__(self, _server_name):
            pass

        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            return lambda fn: fn

        def list_resources(self):
            return lambda fn: fn

        def read_resource(self):
            return lambda fn: fn

        def create_initialization_options(self):
            return {}

        async def run(self, *_args, **_kwargs):
            return None

    @asynccontextmanager
    async def stdio_server():
        yield None, None

    @dataclass
    class Resource:
        uri: str
        name: str
        mimeType: str
        description: str

    @dataclass
    class TextContent:
        type: str
        text: str

    @dataclass
    class Tool:
        name: str
        description: str
        inputSchema: dict

    server_mod.Server = Server
    stdio_mod.stdio_server = stdio_server
    types_mod.Resource = Resource
    types_mod.TextContent = TextContent
    types_mod.Tool = Tool
    sys.modules.setdefault("mcp", mcp_mod)
    sys.modules.setdefault("mcp.server", server_mod)
    sys.modules.setdefault("mcp.server.stdio", stdio_mod)
    sys.modules.setdefault("mcp.types", types_mod)


if PY310_PLUS:
    try:
        from nvidia_resiliency_ext.attribution.base import AttributionState
        from nvidia_resiliency_ext.attribution.mcp_integration.mcp_server import NVRxMCPServer
        from nvidia_resiliency_ext.attribution.mcp_integration.registry import (
            AttributionModuleRegistry,
        )
        from nvidia_resiliency_ext.attribution.orchestration.progressive import (
            MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            PROGRESSIVE_STATUS_STARTED,
            PROGRESSIVE_STATUS_UNSUPPORTED,
            ProgressiveLogAnalysisStartTool,
            ProgressiveStartResult,
        )
        from nvidia_resiliency_ext.attribution.orchestration.types import (
            AttributionRecommendation,
            LogSageAnalysisResult,
            RawAnalysisResultItem,
        )

    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            raise
        _install_mcp_stub()
        from nvidia_resiliency_ext.attribution.base import AttributionState
        from nvidia_resiliency_ext.attribution.mcp_integration.mcp_server import NVRxMCPServer
        from nvidia_resiliency_ext.attribution.mcp_integration.registry import (
            AttributionModuleRegistry,
        )
        from nvidia_resiliency_ext.attribution.orchestration.progressive import (
            MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            PROGRESSIVE_STATUS_STARTED,
            PROGRESSIVE_STATUS_UNSUPPORTED,
            ProgressiveLogAnalysisStartTool,
            ProgressiveStartResult,
        )
        from nvidia_resiliency_ext.attribution.orchestration.types import (
            AttributionRecommendation,
            LogSageAnalysisResult,
            RawAnalysisResultItem,
        )


class FakeLogAnalyzer:
    def __init__(self, _args):
        pass

    async def run(self, _arguments):
        item = RawAnalysisResultItem(
            raw_text="cycle summary",
            auto_resume="STOP",
            auto_resume_explanation="transient issue",
            attribution_text="transient issue",
            checkpoint_saved_flag=1,
            action="STOP",
        )
        return (
            LogSageAnalysisResult(
                [item],
                AttributionRecommendation(action="STOP", source="log_analyzer"),
            ),
            AttributionState.STOP,
        )


class SharedFakeLogAnalyzer(FakeLogAnalyzer):
    instances = []

    def __init__(self, args):
        super().__init__(args)
        self.init_args = dict(args)
        self.progressive_args = []
        self.run_args = []
        self.progressive_started = asyncio.Event()
        self.__class__.instances.append(self)

    async def analyze_logs_rt_start(self, args=None):
        self.progressive_args.append(dict(args or {}))
        self.progressive_started.set()
        return ProgressiveStartResult(status=PROGRESSIVE_STATUS_STARTED).as_payload()

    async def run(self, arguments):
        self.run_args.append(dict(arguments))
        return await super().run(arguments)


class FakeFrAnalyzer:
    def __init__(self, _args):
        pass

    async def run(self, _arguments):
        return (
            {
                "analysis_text": "PGID | Process Group Desc | Missing Ranks\n0 | dp | 1,2",
                "hanging_ranks": "hanging ranks: [1, 2]",
            },
            AttributionState.CONTINUE,
        )


@unittest.skipUnless(
    PY310_PLUS,
    "MCP server cache tests require Python 3.10+",
)
class TestMCPServerCache(unittest.IsolatedAsyncioTestCase):
    def _server(self):
        registry = AttributionModuleRegistry()
        registry.register(
            name="log_analyzer",
            module_class=FakeLogAnalyzer,
            description="fake log analyzer",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object"},
        )
        registry.register(
            name="fr_analyzer",
            module_class=FakeFrAnalyzer,
            description="fake fr analyzer",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object"},
        )
        return NVRxMCPServer(registry=registry)

    def _server_with_log_default(self):
        registry = AttributionModuleRegistry()
        registry.register(
            name="log_analyzer",
            module_class=FakeLogAnalyzer,
            description="fake log analyzer",
            input_schema={
                "type": "object",
                "properties": {
                    "log_path": {"type": "string"},
                    "is_per_cycle": {"type": "boolean", "default": False},
                },
                "required": ["log_path"],
            },
            output_schema={"type": "object"},
        )
        return NVRxMCPServer(registry=registry)

    def _server_with_progressive_start(self):
        registry = AttributionModuleRegistry()
        registry.register(
            name=MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            module_class=ProgressiveLogAnalysisStartTool,
            description="progressive start",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object"},
        )
        return NVRxMCPServer(registry=registry)

    def _server_with_shared_progressive_log_analyzer(self):
        SharedFakeLogAnalyzer.instances = []
        registry = AttributionModuleRegistry()
        registry.register(
            name="log_analyzer",
            module_class=SharedFakeLogAnalyzer,
            description="shared fake log analyzer",
            input_schema={
                "type": "object",
                "properties": {"log_path": {"type": "string"}},
                "required": ["log_path"],
            },
            output_schema={"type": "object"},
        )
        registry.register(
            name=MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            module_class=ProgressiveLogAnalysisStartTool,
            description="progressive start",
            input_schema={
                "type": "object",
                "properties": {"log_path": {"type": "string"}},
                "required": ["log_path"],
            },
            output_schema={"type": "object"},
        )
        return NVRxMCPServer(registry=registry)

    async def test_cached_resource_matches_normalized_module_response(self):
        server = self._server()

        content = await server._handle_module_execution(
            "log_analyzer", {"log_path": "/tmp/job.out"}
        )
        response = json.loads(content[0].text)
        cached = server.registry.get_cached_result_by_uri("log_analyzer", response["result_id"])
        fetched = await server._handle_get_result({"result_id": response["result_id"]})

        self.assertEqual(cached, response)
        self.assertEqual(json.loads(fetched[0].text), response)
        self.assertEqual(response["module"], "log_analyzer")
        self.assertEqual(response["recommendation"], {"action": "STOP", "source": "log_analyzer"})
        self.assertNotIsInstance(cached, tuple)

    async def test_get_result_requires_exact_result_id(self):
        server = self._server()
        server.registry.cache_result_by_id(
            "log_analyzer",
            "abcdef",
            {"result_id": "abcdef", "result": [], "recommendation": {"action": "UNKNOWN"}},
        )

        fetched = await server._handle_get_result({"result_id": "cde"})

        self.assertEqual(json.loads(fetched[0].text), {"error": "Result not found: cde"})

    async def test_result_id_uses_arguments_after_schema_defaults(self):
        server = self._server_with_log_default()

        omitted = await server._handle_module_execution(
            "log_analyzer", {"log_path": "/tmp/job.out"}
        )
        explicit = await server._handle_module_execution(
            "log_analyzer",
            {"log_path": "/tmp/job.out", "is_per_cycle": False},
        )
        omitted_response = json.loads(omitted[0].text)
        explicit_response = json.loads(explicit[0].text)

        self.assertEqual(omitted_response["result_id"], explicit_response["result_id"])
        self.assertEqual(
            omitted_response["resource_uri"],
            explicit_response["resource_uri"],
        )
        self.assertEqual(server.registry.count_results_cache_entries(), 1)

    async def test_direct_fr_response_uses_recommendation_envelope_without_state(self):
        server = self._server()

        content = await server._handle_module_execution("fr_analyzer", {"fr_path": "/tmp/fr_dump"})
        response = json.loads(content[0].text)

        self.assertEqual(response["module"], "fr_analyzer")
        self.assertEqual(response["recommendation"], {"action": "UNKNOWN", "source": "fr_analyzer"})
        self.assertEqual(response["result"]["hanging_ranks"], "hanging ranks: [1, 2]")
        self.assertNotIn("state", response)

    async def test_progressive_start_response_is_not_cached_result(self):
        server = self._server_with_progressive_start()

        content = await server._handle_module_execution(
            MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            {"log_path": "/tmp/job_cycle0.log", "is_per_cycle": True},
        )
        response = json.loads(content[0].text)

        self.assertEqual(response["module"], MODULE_LOG_ANALYZER_PROGRESSIVE_START)
        self.assertEqual(response["status"], PROGRESSIVE_STATUS_UNSUPPORTED)
        self.assertNotIn("result", response)
        self.assertNotIn("recommendation", response)
        self.assertNotIn("result_id", response)
        self.assertEqual(server.registry.count_results_cache_entries(), 0)

    async def test_progressive_start_reuses_terminal_log_analyzer_instance(self):
        server = self._server_with_shared_progressive_log_analyzer()

        # Direct MCP callers may still send obsolete orchestration metadata.
        # Progressive start should ignore it rather than leaking it into LogSage.
        content = await server._handle_module_execution(
            MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            {
                "log_path": "/tmp/job_cycle0.log",
                "user": "alice",
                "job_id": "123",
            },
        )
        response = json.loads(content[0].text)
        analyzer = server.module_instances["log_analyzer"]

        if not analyzer.progressive_args:
            await asyncio.wait_for(analyzer.progressive_started.wait(), timeout=0.5)

        self.assertEqual(response["module"], MODULE_LOG_ANALYZER_PROGRESSIVE_START)
        self.assertEqual(response["status"], PROGRESSIVE_STATUS_STARTED)
        self.assertIs(
            server.module_instances[MODULE_LOG_ANALYZER_PROGRESSIVE_START]._analyzer,
            analyzer,
        )
        self.assertEqual(
            analyzer.progressive_args,
            [{"log_path": "/tmp/job_cycle0.log"}],
        )
        self.assertEqual(analyzer.init_args, {"log_path": "/tmp/job_cycle0.log"})

        await server._handle_module_execution(
            "log_analyzer",
            {"log_path": "/tmp/job_cycle0.log"},
        )

        self.assertEqual(len(SharedFakeLogAnalyzer.instances), 1)
        self.assertEqual(
            analyzer.run_args,
            [{"log_path": "/tmp/job_cycle0.log"}],
        )


if __name__ == "__main__":
    unittest.main()
