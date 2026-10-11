# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Host-side placement and admission for complete, single-GPU pipelines."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from vllm_omni.errors import OmniClientError


class PipelineUnavailableError(OmniClientError):
    """No complete pipeline replica can accept the request."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            error_type="PipelineUnavailableError",
        )


@dataclass(frozen=True)
class PipelineReplicaRoute:
    pipeline_replica_id: int
    gpu_uuid: str
    stage_replica_ids: tuple[int, ...]


def build_pipeline_routes(
    plans: Sequence[Any],
    *,
    devices: tuple[str, ...],
    visible_devices: str | None,
    uuid_for_device: Callable[[str], str],
) -> tuple[PipelineReplicaRoute, ...]:
    """Validate actual placement, including visibility remapping, before spawn.

    Unlike the legacy device resolver, this path never truncates a placement
    or interprets an out-of-range logical index as a physical ordinal.
    """
    visible = None if visible_devices is None else [d.strip() for d in visible_devices.split(",")]
    cache: dict[str, str] = {}

    def device_uuid(device: str) -> str:
        if not isinstance(device, str) or not device.isdigit():
            raise ValueError("Pipeline replicas require exactly one explicit logical GPU per stage replica")
        if visible is not None:
            index = int(device)
            if index >= len(visible) or not visible[index] or visible[index] == "-1":
                raise ValueError(f"Pipeline device {device} is outside the visible GPU list")
            physical = visible[index]
        else:
            physical = device
        if physical not in cache:
            cache[physical] = uuid_for_device(physical)
        return cache[physical]

    uuids = [device_uuid(device) for device in devices]
    if not uuids or len(set(uuids)) != len(uuids):
        raise ValueError("Pipeline replicas must use distinct physical GPUs")
    if not plans:
        raise ValueError("Pipeline deployment requires at least one stage")
    placements: dict[str, list[int]] = {uuid: [] for uuid in uuids}
    for stage_id, plan in enumerate(plans):
        if plan.stage_id != stage_id or len(plan.replicas) != len(uuids):
            raise ValueError("Every pipeline GPU must contain exactly one replica of every stage")
        stage_placements: dict[str, int] = {}
        for replica in plan.replicas:
            if replica.launch_mode != "local":
                raise ValueError("Pipeline deployment supports local replicas only")
            runtime = replica.metadata.runtime_cfg
            device = runtime.get("devices") if isinstance(runtime, Mapping) else getattr(runtime, "devices", None)
            uuid = device_uuid(device)
            if uuid not in placements or uuid in stage_placements:
                raise ValueError("Each pipeline GPU must contain exactly one replica of each stage")
            stage_placements[uuid] = replica.replica_id
        for uuid in uuids:
            placements[uuid].append(stage_placements[uuid])
    return tuple(PipelineReplicaRoute(i, uuid, tuple(placements[uuid])) for i, uuid in enumerate(uuids))


class PipelineReplicaRouter:
    """Select a whole route once; existing requests never migrate on failure."""

    def __init__(
        self,
        routes: Sequence[PipelineReplicaRoute],
        *,
        is_available: Callable[[int, int], bool] | None = None,
    ) -> None:
        self.routes = tuple(routes)
        if not self.routes or any(route.pipeline_replica_id != i for i, route in enumerate(self.routes)):
            raise ValueError("Pipeline route IDs must be contiguous and zero-based")
        self._is_available = is_available
        self._unavailable: set[int] = set()
        self._next = 0
        self._lock = threading.Lock()

    def get_route(self, pipeline_replica_id: int) -> PipelineReplicaRoute:
        if not 0 <= pipeline_replica_id < len(self.routes):
            raise ValueError(f"Unknown pipeline replica {pipeline_replica_id}")
        return self.routes[pipeline_replica_id]

    def is_route_available(self, route: PipelineReplicaRoute) -> bool:
        with self._lock:
            return self._route_available(route)

    def _route_available(self, route: PipelineReplicaRoute) -> bool:
        return route.pipeline_replica_id not in self._unavailable and (
            self._is_available is None
            or all(self._is_available(stage, replica) for stage, replica in enumerate(route.stage_replica_ids))
        )

    def select(self) -> PipelineReplicaRoute:
        with self._lock:
            for offset in range(len(self.routes)):
                index = (self._next + offset) % len(self.routes)
                route = self.routes[index]
                if not self._route_available(route):
                    continue
                self._next = (index + 1) % len(self.routes)
                return route
        raise PipelineUnavailableError("No complete pipeline replica is available")

    def mark_unavailable(self, stage_id: int, replica_id: int) -> None:
        with self._lock:
            for route in self.routes:
                if route.stage_replica_ids[stage_id] == replica_id:
                    self._unavailable.add(route.pipeline_replica_id)
