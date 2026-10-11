# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError

from tests.engine.test_async_omni_engine_input import _FakeStageClient, _make_engine_core_request
from tests.engine.test_orchestrator import FakeStageClient
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine, StageRuntimeInfo
from vllm_omni.engine.messages import StageSubmissionMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.pipeline_router import PipelineReplicaRoute, PipelineReplicaRouter
from vllm_omni.engine.stage_pool import StagePool, StageUnavailableError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_pool(stage_id):
    clients = [FakeStageClient(stage_type="diffusion", final_output=stage_id == 2) for _ in range(2)]
    for i, client in enumerate(clients):
        client.stage_id = stage_id
        client.replica_id = i
    return StagePool(stage_id, clients)


def state(request_id, route):
    return OrchestratorRequestState(
        request_id=request_id,
        prompt={"prompt": request_id},
        final_stage_id=2,
        sampling_params_list=[OmniDiffusionSamplingParams() for _ in range(3)],
        pipeline_route=route,
    )


@pytest.mark.asyncio
async def test_route_overrides_each_stages_round_robin_even_when_completion_reorders():
    pools = [make_pool(i) for i in range(3)]
    a = state("A", PipelineReplicaRoute(0, "GPU-A", (0, 1, 0)))
    b = state("B", PipelineReplicaRoute(1, "GPU-B", (1, 0, 1)))
    for stage, order in enumerate(((a, b), (b, a), (a, b))):
        for req in order:
            await pools[stage].submit_initial(req.request_id, req, req.prompt)
    assert [p.get_bound_replica_id("A") for p in pools] == [0, 1, 0]
    assert [p.get_bound_replica_id("B") for p in pools] == [1, 0, 1]
    assert pools[1].clients[1].add_request_calls[0][0] == "A"


@pytest.mark.asyncio
async def test_dead_target_never_falls_back_to_other_replica():
    pool = make_pool(0)
    req = state("A", PipelineReplicaRoute(0, "GPU-A", (0, 1, 0)))
    pool.mark_replica_unavailable(0)
    with pytest.raises(StageUnavailableError):
        await pool.submit_initial("A", req, req.prompt)
    assert pool.clients[1].add_request_calls == []


@pytest.mark.asyncio
async def test_failed_diffusion_submission_releases_pipeline_binding():
    pool = make_pool(0)
    req = state("A", PipelineReplicaRoute(0, "GPU-A", (0, 1, 0)))

    async def fail(*args, **kwargs):
        raise RuntimeError("submission failed")

    pool.clients[0].add_request_async = fail
    with pytest.raises(RuntimeError, match="submission failed"):
        await pool.submit_initial("A", req, req.prompt)
    assert pool.get_bound_replica_id("A") is None


def test_conflicting_binding_is_rejected():
    pool = make_pool(0)
    pool.bind_replica("A", 1)
    with pytest.raises(StageUnavailableError):
        pool.bind_replica("A", 0)
    assert pool.get_bound_replica_id("A") == 1


@pytest.mark.asyncio
async def test_explicit_target_and_route_must_agree():
    pool = make_pool(0)
    req = state("A", PipelineReplicaRoute(0, "GPU-A", (0, 1, 0)))
    with pytest.raises(StageUnavailableError):
        await pool.submit_initial("A", req, req.prompt, target_replica_id=1)


@pytest.mark.asyncio
async def test_future_stage_death_cleans_waiting_requests_but_preserves_finished_stage():
    pools = [make_pool(i) for i in range(3)]
    routes = (PipelineReplicaRoute(0, "GPU-A", (0, 1, 0)), PipelineReplicaRoute(1, "GPU-B", (1, 0, 1)))
    router = PipelineReplicaRouter(routes)
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        pipeline_router=router,
    )
    pending = state("pending", routes[0])
    finished = state("finished", routes[0])
    finished.finished_stage_ids.add(1)
    healthy = state("healthy", routes[1])
    orch.request_states.update({r.request_id: r for r in (pending, finished, healthy)})
    await orch._handle_dead_replica(1, 1, EngineDeadError("failed"))
    assert set(orch.request_states) == {"finished", "healthy"}
    assert orch.output_async_queue.get_nowait().request_id == "pending"
    assert router.select().pipeline_replica_id == 1


@pytest.mark.asyncio
async def test_queued_request_rechecks_route_and_cleans_preprocessing_resources(tmp_path):
    pools = [make_pool(i) for i in range(3)]
    router = PipelineReplicaRouter((PipelineReplicaRoute(0, "GPU-A", (0, 1, 0)),))
    router.mark_unavailable(1, 1)
    pools[0].bind_replica("queued", 0)
    artifact = tmp_path / "input"
    artifact.mkdir()
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        pipeline_router=router,
    )
    msg = StageSubmissionMessage(
        type="add_request",
        request_id="queued",
        prompt={"prompt": "x"},
        original_prompt={"prompt": "x"},
        output_prompt_text=None,
        sampling_params_list=[OmniDiffusionSamplingParams() for _ in range(3)],
        final_stage_id=2,
        preprocess_ms=0,
        request_timestamp=0,
        enqueue_ts=0,
        pipeline_replica_id=0,
        request_artifact_dirs=[str(artifact)],
    )
    await orch._handle_add_request(msg)
    assert not pools[0].clients[0].add_request_calls
    assert pools[0].get_bound_replica_id("queued") is None
    assert "queued" not in orch.request_states
    assert not artifact.exists()
    assert orch.output_async_queue.get_nowait().status_code == 503


def input_engine():
    engine = object.__new__(AsyncOmniEngine)
    engine.model = "test"
    engine.default_sampling_params_list = [SamplingParams(max_tokens=1)]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)
    engine.stage_pools = [StagePool(0, [_FakeStageClient(), _FakeStageClient()])]
    # Route zero deliberately starts at replica one: numeric identity is not assumed.
    engine._pipeline_router = PipelineReplicaRouter(
        (PipelineReplicaRoute(0, "GPU-A", (1,)), PipelineReplicaRoute(1, "GPU-B", (0,)))
    )
    engine.prompt_expand_func = None
    return engine


def test_image_cache_and_message_use_same_pipeline_selection():
    engine = input_engine()
    seen = []

    def process_inputs(**kwargs):
        seen.append(kwargs["prompt"]["multi_modal_uuids"]["image"][0])
        return _make_engine_core_request(kwargs["request_id"])

    engine.input_processor = SimpleNamespace(process_inputs=process_inputs)
    messages = [
        engine._build_add_request_message(
            request_id=f"req-{i}", prompt={"prompt": "describe", "multi_modal_data": {"image": "same"}}
        )
        for i in range(2)
    ]
    assert [m.pipeline_replica_id for m in messages] == [0, 1]
    assert seen[0].startswith("stage0:rep1:")
    assert seen[1].startswith("stage0:rep0:")
    assert engine.stage_pools[0].get_bound_replica_id("req-0") == 1


@pytest.mark.parametrize("failure", ["preprocess", "enqueue"])
def test_input_failure_releases_pipeline_binding(failure):
    engine = input_engine()

    def process_inputs(**kwargs):
        if failure == "preprocess":
            raise ValueError("preprocess failed")
        return _make_engine_core_request(kwargs["request_id"])

    def put(msg):
        raise ValueError("enqueue failed")

    engine.input_processor = SimpleNamespace(process_inputs=process_inputs)
    engine.request_queue = SimpleNamespace(sync_q=SimpleNamespace(put=put))
    with pytest.raises(ValueError, match="failed"):
        engine.add_request("req", {"prompt": "describe", "multi_modal_data": {"image": "same"}})
    assert engine.stage_pools[0].get_bound_replica_id("req") is None


def test_pipeline_mode_rejects_resumable_before_preprocessing():
    engine = input_engine()
    with pytest.raises(ValueError, match="resumable"):
        engine._build_add_request_message("req", {"prompt": "x"}, resumable=True)
