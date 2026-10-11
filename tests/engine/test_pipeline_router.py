# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm_omni.engine.pipeline_router import (
    PipelineReplicaRoute,
    PipelineReplicaRouter,
    PipelineUnavailableError,
    build_pipeline_routes,
)
from vllm_omni.errors import OmniClientError, raise_client_error_or

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def routes():
    return (
        PipelineReplicaRoute(0, "GPU-A", (0, 1, 0)),
        PipelineReplicaRoute(1, "GPU-B", (1, 0, 1)),
    )


def test_round_robin_selects_complete_routes():
    router = PipelineReplicaRouter(routes())
    assert [router.select().stage_replica_ids for _ in range(4)] == [(0, 1, 0), (1, 0, 1), (0, 1, 0), (1, 0, 1)]


def test_downstream_failure_disables_whole_group():
    router = PipelineReplicaRouter(routes())
    selected = router.select()
    router.mark_unavailable(stage_id=1, replica_id=1)
    assert router.select().pipeline_replica_id == 1
    assert selected.stage_replica_ids == (0, 1, 0)
    assert router.get_route(0) == selected
    router.mark_unavailable(stage_id=2, replica_id=1)
    with pytest.raises(PipelineUnavailableError):
        router.select()


def test_selection_checks_entire_group_availability():
    router = PipelineReplicaRouter(routes(), is_available=lambda stage, replica: (stage, replica) != (2, 0))
    assert router.select().pipeline_replica_id == 1


def test_unavailable_pipeline_preserves_service_unavailable_status():
    router = PipelineReplicaRouter(routes(), is_available=lambda stage, replica: False)
    with pytest.raises(OmniClientError) as direct:
        router.select()
    assert direct.value.status_code == 503
    with pytest.raises(OmniClientError) as queued:
        raise_client_error_or(
            "Selected pipeline replica became unavailable",
            status_code=503,
            error_type="PipelineUnavailableError",
            fallback=RuntimeError,
        )
    assert queued.value.status_code == 503


def test_concurrent_selection_does_not_lose_round_robin_updates():
    router = PipelineReplicaRouter(routes())
    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(lambda _: router.select().pipeline_replica_id, range(100)))
    assert ids.count(0) == ids.count(1) == 50


def plans(placements):
    return [
        SimpleNamespace(
            stage_id=s,
            replicas=[
                SimpleNamespace(
                    replica_id=r, launch_mode="local", metadata=SimpleNamespace(runtime_cfg={"devices": device})
                )
                for r, device in enumerate(devices)
            ],
        )
        for s, devices in enumerate(placements)
    ]


def test_layout_groups_by_actual_gpu_not_replica_number():
    result = build_pipeline_routes(
        plans([["0", "1"], ["1", "0"]]),
        devices=("0", "1"),
        visible_devices="3,2",
        uuid_for_device=lambda d: {"3": "GPU-A", "2": "GPU-B"}[d],
    )
    assert result == (PipelineReplicaRoute(0, "GPU-A", (0, 1)), PipelineReplicaRoute(1, "GPU-B", (1, 0)))


@pytest.mark.parametrize(
    "placements",
    [
        [["0", "1"], ["0"]],
        [["0", "1"], ["0", "0"]],
        [["0", "1"], ["0,1", "1"]],
        [["0", "1"], ["0", "2"]],
    ],
)
def test_incomplete_or_misplaced_pipeline_rejected(placements):
    with pytest.raises(ValueError):
        build_pipeline_routes(
            plans(placements), devices=("0", "1"), visible_devices="3,2", uuid_for_device=lambda d: "GPU-" + d
        )


def test_duplicate_physical_gpu_rejected():
    with pytest.raises(ValueError, match="distinct"):
        build_pipeline_routes(
            plans([["0", "1"]]), devices=("0", "1"), visible_devices="GPU-A,GPU-A", uuid_for_device=lambda d: d
        )


def test_out_of_range_visibility_does_not_fall_back_to_physical_id():
    with pytest.raises(ValueError, match="visible"):
        build_pipeline_routes(plans([["3"]]), devices=("3",), visible_devices="0,1", uuid_for_device=lambda d: d)
