# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from torch.distributed import TCPStore
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint

from nvidia_resiliency_ext.attribution.fact.client import (
    FactAttributionResult,
    FactAttributionService,
    collect_recent_dmesg_text,
)
from nvidia_resiliency_ext.attribution.fact.rpc import (
    DEFAULT_MAX_RPC_BYTES,
    default_socket_path,
    json_dumps,
    recv_frame,
    send_frame,
)
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig, setup_logger

logger = logging.getLogger(LogConfig.name)

DEFAULT_DMESG_WINDOW_S = 12.0 * 60.0
DEFAULT_OBSERVATION_DEADLINE_S = 30.0
DEFAULT_STORE_TIMEOUT_S = 60.0
_ATTRIBUTOR_FAILURE_PREFIX = "__nvrx_fact_attributor_failed__:"


@dataclass(frozen=True)
class FactAgentRequest:
    run_id: str
    cycle: int
    rdzv_endpoint: str
    local_node: str
    is_store_host: bool = False
    store_timeout_s: float = DEFAULT_STORE_TIMEOUT_S
    job_id: Optional[str] = None
    role: str = "default"
    expected_nodes: tuple[str, ...] = ()
    ranks_per_node: int = 1
    nranks: int = 1
    dmesg_path: Optional[str] = None
    result_path: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FactAgentRequest":
        if payload.get("event") != "cycle_failed":
            raise ValueError("unsupported FACT agent event")
        run_id = str(payload.get("run_id") or "").strip()
        rdzv_endpoint = str(payload.get("rdzv_endpoint") or "").strip()
        local_node = str(payload.get("local_node") or socket.getfqdn(socket.gethostname()))
        if not run_id:
            raise ValueError("cycle_failed requires run_id")
        if not rdzv_endpoint:
            raise ValueError("cycle_failed requires rdzv_endpoint")
        cycle = int(payload["cycle"])
        expected_nodes_raw = payload.get("expected_nodes") or []
        if not isinstance(expected_nodes_raw, list):
            raise ValueError("expected_nodes must be a list when provided")
        expected_nodes = tuple(str(node) for node in expected_nodes_raw if str(node))
        return cls(
            run_id=run_id,
            cycle=cycle,
            rdzv_endpoint=rdzv_endpoint,
            local_node=local_node,
            is_store_host=bool(payload.get("is_store_host", False)),
            store_timeout_s=float(payload.get("store_timeout_s", DEFAULT_STORE_TIMEOUT_S)),
            job_id=str(payload.get("job_id") or run_id),
            role=str(payload.get("role") or "default"),
            expected_nodes=expected_nodes,
            ranks_per_node=max(1, int(payload.get("ranks_per_node") or 1)),
            nranks=max(1, int(payload.get("nranks") or 1)),
            dmesg_path=str(payload["dmesg_path"]) if payload.get("dmesg_path") else None,
            result_path=str(payload["result_path"]) if payload.get("result_path") else None,
        )


class FactAgentKeys:
    def __init__(self, run_id: str, cycle: int) -> None:
        self.prefix = f"fact_agent:{run_id}:cycle{cycle}"

    @staticmethod
    def node_suffix(node: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in node)

    @property
    def attributor_id(self) -> str:
        return f"{self.prefix}:attributor_id"

    @property
    def done_count(self) -> str:
        return f"{self.prefix}:done_count"

    @property
    def result_status(self) -> str:
        return f"{self.prefix}:result_status"

    @property
    def result_path(self) -> str:
        return f"{self.prefix}:result_path"

    @property
    def faulty_nodes(self) -> str:
        return f"{self.prefix}:faulty_nodes"

    def status(self, node: str) -> str:
        return f"{self.prefix}:status:{self.node_suffix(node)}"

    def observation(self, node: str) -> str:
        return f"{self.prefix}:observation:{self.node_suffix(node)}"

    def done(self, index: int) -> str:
        return f"{self.prefix}:done:{index}"


StoreFactory = Callable[[FactAgentRequest], Any]
FactClientFactory = Callable[[], FactAttributionService]
DmesgCollector = Callable[[float, str], str]


class FactAgent:
    def __init__(
        self,
        *,
        fact_url: str,
        socket_path: Optional[str] = None,
        dmesg_window_s: float = DEFAULT_DMESG_WINDOW_S,
        observation_deadline_s: float = DEFAULT_OBSERVATION_DEADLINE_S,
        fact_timeout_s: float = 60.0,
        dmesg_prefilter: bool = True,
        store_factory: Optional[StoreFactory] = None,
        fact_client_factory: Optional[FactClientFactory] = None,
        dmesg_collector: Optional[DmesgCollector] = None,
    ) -> None:
        self.fact_url = fact_url
        self.socket_path = socket_path or default_socket_path()
        self.dmesg_window_s = dmesg_window_s
        self.observation_deadline_s = observation_deadline_s
        self.fact_timeout_s = fact_timeout_s
        self.dmesg_prefilter = dmesg_prefilter
        self._store_factory = store_factory or self._connect_tcp_store
        self._fact_client_factory = fact_client_factory or self._new_fact_client
        self._dmesg_collector = dmesg_collector or self._collect_dmesg
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="nvrx-fact-agent")
        self._stop_event = threading.Event()

    def _new_fact_client(self) -> FactAttributionService:
        return FactAttributionService(
            url=self.fact_url,
            timeout_s=self.fact_timeout_s,
            dmesg_prefilter=self.dmesg_prefilter,
        )

    def _connect_tcp_store(self, request: FactAgentRequest) -> TCPStore:
        host, port = parse_rendezvous_endpoint(request.rdzv_endpoint, default_port=-1)
        if not host or port == -1:
            raise ValueError(f"invalid rendezvous endpoint: {request.rdzv_endpoint!r}")
        return TCPStore(
            host,
            port,
            is_master=False,
            timeout=timedelta(seconds=max(1.0, request.store_timeout_s)),
            multi_tenant=True,
        )

    @staticmethod
    def _collect_dmesg(window_s: float, local_node: str) -> str:
        return collect_recent_dmesg_text(window_s=window_s, hostname=local_node)

    def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("event") == "ping":
            return {"accepted": True}
        request = FactAgentRequest.from_payload(payload)
        self._executor.submit(self.process_cycle_failed, request)
        return {"accepted": True}

    def process_cycle_failed(self, request: FactAgentRequest) -> None:
        try:
            store = self._store_factory(request)
        except Exception as exc:
            logger.warning("FACT agent failed to connect to TCPStore: %s", exc)
            return
        keys = FactAgentKeys(request.run_id, request.cycle)
        if request.is_store_host:
            self._process_store_host(request, store, keys)
        else:
            self._submit_local_evidence(request, store, keys)

    def _process_store_host(
        self, request: FactAgentRequest, store: Any, keys: FactAgentKeys
    ) -> None:
        expected_nodes = list(dict.fromkeys(request.expected_nodes or (request.local_node,)))
        nranks = max(request.nranks, request.ranks_per_node * max(1, len(expected_nodes)))
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=max(0.0, self.dmesg_window_s))
        self._store_set_json(store, keys.result_status, {"status": "pending"})
        try:
            service = self._fact_client_factory()
            attributor_id = service.create_failure_attributor(
                job_id=request.job_id or request.run_id,
                cycle_index=request.cycle,
                role=request.role or "default",
                nodes=expected_nodes,
                ranks_per_node=request.ranks_per_node,
                nranks=nranks,
                start_time=start_time,
                end_time=end_time,
            )
            store.set(keys.attributor_id, str(attributor_id).encode("utf-8"))
        except Exception as exc:
            logger.warning("FACT agent failed to create FACT attributor: %s", exc)
            self._publish_attributor_failure(store, keys, exc)
            self._store_set_json(store, keys.result_status, {"status": "failed", "error": str(exc)})
            return

        self._submit_local_evidence(request, store, keys)
        statuses = self._collect_statuses(store, keys, expected_nodes)
        observation_ids = [
            status["observation_id"]
            for status in statuses.values()
            if status.get("status") == "submitted" and status.get("observation_id") is not None
        ]
        try:
            result = service.get_attribution_result(
                attributor_id=str(attributor_id),
                observation_ids=observation_ids,
            )
        except Exception as exc:
            logger.warning("FACT agent attribution GET failed: %s", exc)
            self._store_set_json(store, keys.result_status, {"status": "failed", "error": str(exc)})
            return

        payload = self._result_payload(result, statuses)
        if request.result_path:
            try:
                self._write_json_file(request.result_path, payload)
                store.set(keys.result_path, request.result_path.encode("utf-8"))
            except Exception as exc:
                logger.warning(
                    "FACT agent failed to write result artifact %s: %s", request.result_path, exc
                )
        self._store_set_json(store, keys.faulty_nodes, result.faulty_nodes)
        self._store_set_json(
            store,
            keys.result_status,
            {
                "status": "complete",
                "faulty_nodes": result.faulty_nodes,
                "observation_count": len(observation_ids),
                "result_path": request.result_path or "",
            },
        )

    def _submit_local_evidence(
        self, request: FactAgentRequest, store: Any, keys: FactAgentKeys
    ) -> None:
        node = request.local_node
        status: dict[str, Any] = {
            "node": node,
            "cycle": request.cycle,
            "source": "dmesg",
            "status": "skipped",
            "observation_id": None,
            "lines_collected": 0,
            "bytes_collected": 0,
            "dmesg_path": request.dmesg_path or "",
            "dmesg_write_error": "",
            "error": "",
        }

        try:
            dmesg_text = self._dmesg_collector(self.dmesg_window_s, node)
            collection_end_time = datetime.now(timezone.utc)
            status["lines_collected"] = len(dmesg_text.splitlines())
            status["bytes_collected"] = len(dmesg_text.encode("utf-8", errors="replace"))
        except Exception as exc:
            status.update(status="collect_failed", error=str(exc))
            self._write_terminal_status(store, keys, node, status)
            return

        if request.dmesg_path:
            try:
                self._write_text_file(request.dmesg_path, dmesg_text)
            except Exception as exc:
                status["dmesg_write_error"] = str(exc)
                logger.warning(
                    "FACT agent failed to write dmesg evidence %s: %s",
                    request.dmesg_path,
                    exc,
                )

        try:
            raw_attributor_id = self._store_get_bytes_with_deadline(
                store, keys.attributor_id, request.store_timeout_s
            )
            if not raw_attributor_id:
                raise RuntimeError("timed out waiting for attributor_id")
            attributor_id = raw_attributor_id.decode("utf-8")
            if attributor_id.startswith(_ATTRIBUTOR_FAILURE_PREFIX):
                error = attributor_id[len(_ATTRIBUTOR_FAILURE_PREFIX) :]
                raise RuntimeError(f"FACT attributor creation failed on store host: {error}")
        except Exception as exc:
            status.update(status="post_failed", error=str(exc))
            self._write_terminal_status(store, keys, node, status)
            return

        end_time = collection_end_time
        start_time = end_time - timedelta(seconds=max(0.0, self.dmesg_window_s))
        try:
            observation_id = self._fact_client_factory().submit_dmesg_text_observation(
                attributor_id=attributor_id,
                dmesg_text=dmesg_text,
                start_time=start_time,
                end_time=end_time,
                default_hostname=node,
            )
            if observation_id is None:
                status.update(status="empty")
            else:
                status.update(status="submitted", observation_id=observation_id)
                store.set(keys.observation(node), str(observation_id).encode("utf-8"))
        except Exception as exc:
            status.update(status="post_failed", error=str(exc))
        self._write_terminal_status(store, keys, node, status)

    def _write_terminal_status(
        self,
        store: Any,
        keys: FactAgentKeys,
        node: str,
        status: dict[str, Any],
    ) -> None:
        self._store_set_json(store, keys.status(node), status)
        try:
            index = int(store.add(keys.done_count, 1))
            store.set(keys.done(index), node.encode("utf-8"))
        except Exception as exc:
            logger.warning("FACT agent failed to publish completion for %s: %s", node, exc)

    def _publish_attributor_failure(
        self,
        store: Any,
        keys: FactAgentKeys,
        exc: Exception,
    ) -> None:
        try:
            store.set(keys.attributor_id, f"{_ATTRIBUTOR_FAILURE_PREFIX}{exc}".encode("utf-8"))
        except Exception as store_exc:
            logger.warning(
                "FACT agent failed to publish attributor failure sentinel: %s", store_exc
            )

    def _collect_statuses(
        self,
        store: Any,
        keys: FactAgentKeys,
        expected_nodes: list[str],
    ) -> dict[str, dict[str, Any]]:
        expected = set(expected_nodes)
        statuses: dict[str, dict[str, Any]] = {}
        pending_nodes: set[str] = set()
        next_index = 1
        deadline = time.monotonic() + max(0.0, self.observation_deadline_s)
        while time.monotonic() < deadline and len(expected & set(statuses)) < len(expected):
            try:
                done_count = int(store.add(keys.done_count, 0))
            except Exception:
                done_count = next_index - 1
            progressed = False
            while next_index <= done_count:
                raw_node = self._store_get_bytes_with_deadline(store, keys.done(next_index), 0.2)
                if not raw_node:
                    break
                next_index += 1
                node = raw_node.decode("utf-8")
                pending_nodes.add(node)
                progressed = True

            for node in list(pending_nodes):
                raw_status = self._store_get_bytes_with_deadline(store, keys.status(node), 0.2)
                if not raw_status:
                    continue
                try:
                    status = json.loads(raw_status.decode("utf-8"))
                    if isinstance(status, dict):
                        statuses[str(status.get("node") or node)] = status
                    else:
                        logger.warning("FACT agent found non-object status payload for %s", node)
                    progressed = True
                    pending_nodes.discard(node)
                except json.JSONDecodeError:
                    logger.warning("FACT agent found invalid status payload for %s", node)
                    progressed = True
                    pending_nodes.discard(node)
            if not progressed:
                time.sleep(0.1)

        for node in sorted(expected - set(statuses)):
            statuses[node] = {
                "node": node,
                "cycle": None,
                "source": "dmesg",
                "status": "timeout",
                "observation_id": None,
                "lines_collected": 0,
                "bytes_collected": 0,
                "dmesg_path": "",
                "dmesg_write_error": "",
                "error": "no terminal status before deadline",
            }
        return statuses

    @staticmethod
    def _store_get_bytes_with_deadline(store: Any, key: str, timeout_s: float) -> Optional[bytes]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                if store.check([key]):
                    return store.get(key)
            except Exception:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.1, remaining))

    @staticmethod
    def _store_set_json(store: Any, key: str, payload: Any) -> None:
        store.set(key, json_dumps(payload))

    @staticmethod
    def _write_json_file(path: str, payload: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _write_text_file(path: str, payload: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", errors="replace") as handle:
            handle.write(payload)
            if payload and not payload.endswith("\n"):
                handle.write("\n")

    @staticmethod
    def _result_payload(
        result: FactAttributionResult,
        statuses: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "attributor_id": result.attributor_id,
            "observation_ids": result.observation_ids,
            "faulty_nodes": result.faulty_nodes,
            "attribution": result.attribution,
            "submission_statuses": statuses,
        }

    def stop(self) -> None:
        self._stop_event.set()

    def serve_forever(self, *, max_rpc_bytes: int = DEFAULT_MAX_RPC_BYTES) -> None:
        socket_path = Path(self.socket_path)
        if socket_path.exists():
            socket_path.unlink()
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            try:
                server.bind(str(socket_path))
                os.chmod(socket_path, 0o600)
                server.listen(128)
                server.settimeout(1.0)
                logger.info("nvrx-fact-agent listening on %s", socket_path)
                while not self._stop_event.is_set():
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn, max_rpc_bytes),
                        daemon=True,
                    ).start()
            finally:
                with contextlib.suppress(FileNotFoundError):
                    socket_path.unlink()

    def _handle_connection(self, conn: socket.socket, max_rpc_bytes: int) -> None:
        with conn:
            try:
                payload = recv_frame(conn, max_bytes=max_rpc_bytes)
                ack = self.handle_payload(payload)
                send_frame(conn, ack)
            except Exception as exc:
                logger.warning("nvrx-fact-agent rejected RPC: %s", exc)
                with contextlib.suppress(Exception):
                    send_frame(conn, {"accepted": False, "error": str(exc)})


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nvrx-fact-agent")
    parser.add_argument(
        "--fact-url", required=True, help="FACT API URL, e.g. http://host:8001/latest"
    )
    parser.add_argument("--socket-path", default=default_socket_path())
    parser.add_argument("--dmesg-window", type=float, default=DEFAULT_DMESG_WINDOW_S)
    parser.add_argument(
        "--observation-deadline", type=float, default=DEFAULT_OBSERVATION_DEADLINE_S
    )
    parser.add_argument("--fact-timeout", type=float, default=60.0)
    parser.add_argument(
        "--dmesg-prefilter",
        type=lambda x: str(x).lower() not in ("false", "0", "no"),
        default=True,
    )
    parser.add_argument("--max-rpc-bytes", type=int, default=DEFAULT_MAX_RPC_BYTES)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = get_arg_parser().parse_args(argv)
    setup_logger(node_local_tmp_prefix="nvrxfactagent")
    service = FactAgent(
        fact_url=args.fact_url,
        socket_path=args.socket_path,
        dmesg_window_s=args.dmesg_window,
        observation_deadline_s=args.observation_deadline,
        fact_timeout_s=args.fact_timeout,
        dmesg_prefilter=args.dmesg_prefilter,
    )
    service.serve_forever(max_rpc_bytes=args.max_rpc_bytes)


if __name__ == "__main__":
    main()
