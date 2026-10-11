# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from copy import deepcopy

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    PipelineDeploymentConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    load_deploy_config,
    merge_pipeline_deploy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def pipeline():
    return PipelineConfig(
        model_type="test_pipeline_deployment",
        model_arch="TestModel",
        stages=(
            StagePipelineConfig(stage_id=0, model_stage="first", execution_type=StageExecutionType.LLM_AR),
            StagePipelineConfig(
                stage_id=1,
                model_stage="second",
                execution_type=StageExecutionType.LLM_AR,
                input_sources=[0],
                final_output=True,
            ),
        ),
    )


def test_colocated_replicas_expand_all_registered_stages_without_mutating_deploy(tmp_path, pipeline):
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "pipeline_deployment:\n"
        "  mode: colocated_replicas\n"
        '  devices: "0, 2"\n'
        "stages:\n"
        "  - stage_id: 0\n"
        "    max_num_seqs: 8\n"
    )
    deploy = load_deploy_config(path)
    before = deepcopy(deploy)

    stages = merge_pipeline_deploy(pipeline, deploy)

    assert [(stage.yaml_runtime["devices"], stage.yaml_runtime["num_replicas"]) for stage in stages] == [
        ("0,2", 2),
        ("0,2", 2),
    ]
    assert stages[0].yaml_engine_args["max_num_seqs"] == 8
    assert all(stage.session_mode == "turn" for stage in stages)
    assert all(stage.yaml_engine_args["async_chunk"] is False for stage in stages)
    assert deploy == before


@pytest.mark.parametrize("devices", ["", " ", "0,", ",0", "0,,1", "-1", "foo", "0,00"])
def test_pipeline_deployment_rejects_invalid_device_lists(devices):
    with pytest.raises(ValueError, match="pipeline_deployment.devices"):
        PipelineDeploymentConfig(mode="colocated_replicas", devices=devices)


@pytest.mark.parametrize("replicas", [0, -1, 1, True])
def test_pipeline_deployment_rejects_invalid_replica_count(replicas):
    with pytest.raises(ValueError, match="pipeline_deployment.num_replicas"):
        PipelineDeploymentConfig(mode="colocated_replicas", devices="0,1", num_replicas=replicas)


def test_pipeline_deployment_rejects_unknown_mode():
    with pytest.raises(ValueError, match="pipeline_deployment.mode"):
        PipelineDeploymentConfig(mode="other", devices="0")


@pytest.mark.parametrize(
    "stage_yaml",
    [
        "    devices: '0'\n",
        "    num_replicas: 1\n",
        "    runtime:\n      devices: '0'\n",
        "    runtime:\n      num_replicas: 1\n",
        "    env:\n      CUDA_VISIBLE_DEVICES: '0'\n",
    ],
)
def test_pipeline_deployment_rejects_explicit_stage_placement_even_if_equal(tmp_path, pipeline, stage_yaml):
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "pipeline_deployment:\n  mode: colocated_replicas\n  devices: '0'\nstages:\n  - stage_id: 0\n" + stage_yaml
    )
    with pytest.raises(ValueError, match="pipeline_deployment conflicts with explicit stage 0 placement"):
        merge_pipeline_deploy(pipeline, load_deploy_config(path))


def test_programmatic_explicit_replica_count_conflicts_even_when_one(pipeline):
    deploy = DeployConfig(
        pipeline_deployment=PipelineDeploymentConfig(mode="colocated_replicas", devices="0"),
        stages=[StageDeployConfig(stage_id=0, num_replicas=1)],
    )
    with pytest.raises(ValueError, match="explicit stage 0 placement: num_replicas"):
        merge_pipeline_deploy(pipeline, deploy)


@pytest.mark.parametrize(
    "platform_stage",
    [
        {"stage_id": 1, "devices": "0"},
        {"stage_id": 1, "num_replicas": 1},
        {"stage_id": 1, "runtime": {"devices": "0"}},
        {"stage_id": 1, "runtime": {"num_replicas": 1}},
        {"stage_id": 1, "runtime": {"env": {"CUDA_VISIBLE_DEVICES": "0"}}},
    ],
)
def test_pipeline_deployment_rejects_platform_placement(pipeline, platform_stage):
    deploy = DeployConfig(
        pipeline_deployment=PipelineDeploymentConfig(mode="colocated_replicas", devices="0"),
        platforms={"cuda": {"stages": [platform_stage]}},
    )
    with pytest.raises(ValueError, match="pipeline_deployment conflicts with explicit stage 1 placement"):
        merge_pipeline_deploy(pipeline, deploy)


@pytest.mark.parametrize(
    "key,value", [("devices", "0"), ("num_replicas", 1), ("stage_1_devices", "0"), ("stage_1_num_replicas", 1)]
)
def test_pipeline_deployment_rejects_cli_placement(pipeline, key, value):
    deploy = DeployConfig(pipeline_deployment=PipelineDeploymentConfig(mode="colocated_replicas", devices="0"))
    with pytest.raises(ValueError, match="pipeline_deployment conflicts with explicit CLI placement"):
        merge_pipeline_deploy(pipeline, deploy, cli_overrides={key: value})


def test_pipeline_deployment_rejects_strategy_and_streaming(pipeline):
    placement = PipelineDeploymentConfig(mode="colocated_replicas", devices="0")
    with pytest.raises(ValueError, match="strategy_specs"):
        merge_pipeline_deploy(pipeline, DeployConfig(pipeline_deployment=placement), strategy_specs={})
    with pytest.raises(ValueError, match="session_mode='turn'"):
        merge_pipeline_deploy(pipeline, DeployConfig(pipeline_deployment=placement, session_mode="duplex"))
    with pytest.raises(ValueError, match="async_chunk=False"):
        merge_pipeline_deploy(pipeline, DeployConfig(pipeline_deployment=placement, async_chunk=True))
    with pytest.raises(ValueError, match="async_chunk=False"):
        VllmOmniConfig.from_pipeline_config(
            pipeline,
            user_deploy_config=DeployConfig(pipeline_deployment=placement),
            cli_overrides={"async_chunk": True},
        )
    with pytest.raises(ValueError, match="session_mode='turn'"):
        merge_pipeline_deploy(
            pipeline, DeployConfig(pipeline_deployment=placement), cli_overrides={"session_mode": "duplex"}
        )


def test_pipeline_deployment_rejects_inherited_placement(tmp_path, pipeline):
    base = tmp_path / "base.yaml"
    base.write_text("stages:\n  - stage_id: 0\n    num_replicas: 1\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("base_config: base.yaml\npipeline_deployment:\n  mode: colocated_replicas\n  devices: '0'\n")
    with pytest.raises(ValueError, match="explicit stage 0 placement"):
        merge_pipeline_deploy(pipeline, load_deploy_config(overlay))


def test_structured_config_exposes_expanded_pipeline_deployment(pipeline):
    placement = PipelineDeploymentConfig(mode="colocated_replicas", devices="0,2")
    deploy = DeployConfig(pipeline_deployment=placement)

    config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)

    assert config.orchestrator_config.pipeline_deployment == placement
    assert [(stage.runtime_config.devices, stage.runtime_config.num_replicas) for stage in config.stage_configs] == [
        ("0,2", 2),
        ("0,2", 2),
    ]
    assert all(stage.connector_config.async_chunk is False for stage in config.stage_configs)
    assert deploy.stages == []


@pytest.mark.parametrize(
    "override",
    [
        {"tensor_parallel_size": 2},
        {"data_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"stage_1_tensor_parallel_size": 2},
    ],
)
def test_structured_config_rejects_multigpu_stage_after_cli_overrides(pipeline, override):
    deploy = DeployConfig(pipeline_deployment=PipelineDeploymentConfig(mode="colocated_replicas", devices="0,1"))
    with pytest.raises(ValueError, match="pipeline_deployment requires one GPU per stage replica"):
        VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy, cli_overrides=override)


def test_structured_config_rejects_inherited_parallel_size(tmp_path, pipeline):
    base = tmp_path / "base.yaml"
    base.write_text("data_parallel_size: 2\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("base_config: base.yaml\npipeline_deployment:\n  mode: colocated_replicas\n  devices: '0,1'\n")
    with pytest.raises(ValueError, match="pipeline_deployment requires one GPU per stage replica"):
        VllmOmniConfig.from_pipeline_config(pipeline, deploy_config_path=str(overlay))
