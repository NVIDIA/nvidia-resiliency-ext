# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous persistence for complete shared L0 products."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping

from ..execution import L0Artifacts
from ..l0.codec import write_l0_bundle
from ..runtime import THREAD_EXECUTOR_FACTORY, ExecutorFactory
from .artifact_io import write_json_atomic

L0PublishedCallback = Callable[[L0Artifacts, Mapping[str, str]], None]


class L0ArtifactPublisher:
    """Persist one L0 snapshot without delaying model-route fanout."""

    def __init__(
        self,
        *,
        bundle_path: str | Path | None = None,
        decision_evidence_path: str | Path | None = None,
        model_view_path: str | Path | None = None,
        on_published: L0PublishedCallback | None = None,
        executor_factory: ExecutorFactory = THREAD_EXECUTOR_FACTORY,
    ) -> None:
        self._bundle_path = Path(bundle_path) if bundle_path else None
        self._decision_evidence_path = (
            Path(decision_evidence_path) if decision_evidence_path else None
        )
        self._model_view_path = Path(model_view_path) if model_view_path else None
        self._on_published = on_published
        self._executor: Executor = executor_factory(
            max_workers=1,
            thread_name_prefix="restart-agent-l0-artifacts",
        )
        self._future: Future[Mapping[str, str]] | None = None
        self._lock = Lock()

    def publish(self, artifacts: L0Artifacts) -> None:
        """Enqueue one complete L0 snapshot and return immediately."""

        with self._lock:
            if self._future is not None:
                raise RuntimeError("L0 artifacts were already submitted for publication")
            self._future = self._executor.submit(self._write, artifacts)

    def wait(self) -> Mapping[str, str]:
        """Wait for an enqueued snapshot and propagate persistence failures."""

        with self._lock:
            future = self._future
        return future.result() if future is not None else {}

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _write(self, artifacts: L0Artifacts) -> Mapping[str, str]:
        published: dict[str, str] = {}
        if self._bundle_path is not None:
            write_l0_bundle(self._bundle_path, artifacts.bundle)
            published["l0_bundle"] = str(self._bundle_path)
        if self._decision_evidence_path is not None:
            write_json_atomic(
                self._decision_evidence_path,
                artifacts.decision_evidence.to_payload(),
            )
            published["decision_evidence"] = str(self._decision_evidence_path)
        if self._model_view_path is not None and artifacts.model_view is not None:
            write_json_atomic(
                self._model_view_path,
                artifacts.model_view.to_payload(),
            )
            published["l0_model_view"] = str(self._model_view_path)
        if self._on_published is not None:
            self._on_published(artifacts, published)
        return published
