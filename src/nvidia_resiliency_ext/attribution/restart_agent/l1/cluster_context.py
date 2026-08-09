# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable cluster execution semantics supplied to L1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClusterExecutionContext:
    """Product guarantees that define the environment of the next attempt."""

    allocation_model: str = "homogeneous_node_pool"
    workload_isolation: str = "exclusive"
    replacement_hardware_bom: str = "equivalent"
    replacement_software_bom: str = "equivalent"
    replacement_resource_capacity: str = "equivalent"
    replacement_resource_limits: str = "unchanged"
    replacement_storage_access: str = "equivalent"
    dependency_paths: tuple[str, ...] = (
        "compute_node",
        "scale_up_fabric",
        "scale_out_fabric",
        "distributed_storage",
        "service_control",
    )
    faulty_resource_handling: str = "independent_detection_and_quarantine"

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical documentation and test representation."""

        return {
            "allocation_model": self.allocation_model,
            "workload_isolation": self.workload_isolation,
            "replacement_hardware_bom": self.replacement_hardware_bom,
            "replacement_software_bom": self.replacement_software_bom,
            "replacement_resource_capacity": self.replacement_resource_capacity,
            "replacement_resource_limits": self.replacement_resource_limits,
            "replacement_storage_access": self.replacement_storage_access,
            "dependency_paths": list(self.dependency_paths),
            "faulty_resource_handling": self.faulty_resource_handling,
        }


DEFAULT_CLUSTER_EXECUTION_CONTEXT = ClusterExecutionContext()


def render_cluster_execution_context(context: ClusterExecutionContext) -> str:
    """Render the immutable context once for the static L1 system prompt."""

    # This renderer encodes product guarantees, not a user-configurable extension point.
    if context != DEFAULT_CLUSTER_EXECUTION_CONTEXT:
        raise ValueError("unsupported cluster execution context")

    return """\
- Evaluate the next attempt under this declared cluster execution context. The workload
  runs on an exclusive allocation drawn from a homogeneous node pool; unrelated user
  workloads do not share its allocated nodes. Eligible nodes and devices are fungible
  for restart analysis.
- Workload code, data, configuration, and workload-selected software remain unchanged.
  Workload identity, credentials, environment, configured paths, and persistent filesystem
  permissions also remain unchanged across restart. Operator or administrator remediation
  is not part of the restart transition. Failed process state is recreated, and normal
  restart delay applies. Physical nodes or devices may be replaced, but replacement
  preserves the hardware and software BOM, resource capacity and limits, and storage
  access. The allocation resource envelope is invariant across the homogeneous pool.
- The workload depends on layered compute-node, scale-up-fabric, scale-out-fabric,
  distributed-storage, and service/control paths. A competing cause is relevant only when
  the failed operation depends on that path and the proposed cause can produce the observed
  failure mechanism. The exact failed physical component need not be identified. The mere
  presence of a component in the cluster, or the generic possibility that hardware,
  services, or workload code can fail, is not material evidence.
- A separate health mechanism may detect and quarantine malfunctioning nodes or devices, but
  this guarantee does not prove that a malfunction occurred. Process recreation, normal
  delay, reset, reconnection, replacement, quarantine, or mutable node-local/external state
  supports may_recover only when it can address the observed failure mechanism. Node
  replacement does not by itself repair a persistent fabric, storage, or service fault.
- When supported workload and infrastructure alternatives both remain, report the failure
  domain as unknown. Assess retry outlook independently: failure_domain=unknown may pair
  with retry_outlook_without_workload_change=may_recover. Unrelated speculation in either
  direction must not dilute stronger positive evidence or force either claim to unknown.
- Do not infer changed capacity, resource limits, workload demand, data, configuration, or
  software behavior from physical replacement. These guarantees describe the restart
  transition and are not current-log evidence of failure cause, ownership, or successful
  recovery. A possible state transition does not prove recovery, while a fixed request,
  workload call stack, or missing cleanup message does not prove that mutable failure
  state survives restart."""
