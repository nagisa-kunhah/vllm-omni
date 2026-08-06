# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_video_patchify_round_trip_preserves_values():
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
        minimax_h3_unpatchify_video_tokens,
    )

    latent = torch.arange(2 * 3 * 2 * 4 * 6).reshape(2, 3, 2, 4, 6)
    rows = minimax_h3_patchify_video_latent(
        latent,
        patch_size=(1, 2, 2),
    )
    restored = minimax_h3_unpatchify_video_tokens(
        rows,
        latent_shape=(2, 2, 3, 3),
        patch_size=(1, 2, 2),
    )

    torch.testing.assert_close(restored, latent)


def test_audio_pack_round_trip_preserves_channel_major_order():
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_pack_audio_latent,
        minimax_h3_unpack_audio_tokens,
    )

    latent = torch.arange(2 * 4 * 5).reshape(2, 4, 5)
    rows = minimax_h3_pack_audio_latent(latent)
    restored = minimax_h3_unpack_audio_tokens(
        rows,
        audio_t=10,
        audio_channel=2,
    )

    torch.testing.assert_close(restored, latent)


def test_t2va_and_fl2va_packing_keep_update_rows_separate():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    common = dict(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
    )
    t2va = minimax_h3_packed_sequence(
        **common,
        include_keyframe_cond=False,
    )
    fl2va = minimax_h3_packed_sequence(
        **common,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0],
        frame_count=5,
    )

    assert int(t2va["seq_len"]) == 64
    assert t2va["img_pos"].numel() == 12
    assert t2va["update_mask"].all()
    assert fl2va["img_pos"].numel() == 18
    assert fl2va["update_mask"].sum().item() == 12
    assert (~fl2va["update_mask"]).sum().item() == 6
    assert t2va["audio_pos"].numel() == fl2va["audio_pos"].numel() == 6


def test_ref2va_packing_tracks_video_and_audio_update_masks():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        ref_blocks=[
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "audio", "ref_audio_t": 2},
        ],
    )

    assert int(packed["seq_len"]) == 64
    assert packed["img_pos"].numel() == 16
    assert packed["update_mask"].sum().item() == 12
    assert packed["audio_pos"].numel() == 10
    assert packed["audio_update_mask"].sum().item() == 6
    assert packed["cu_seqlens"].tolist() == [0, 30, 64]


def test_condition_noise_is_seeded_and_keeps_clean_anchor_at_timestep_one():
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )

    image_rows = torch.randn(4, 96)
    audio_rows = torch.randn(6, 32)

    torch.testing.assert_close(
        minimax_h3_imgvid_cond_noise_aug_rows(
            image_rows,
            condition_shapes=[(1, 4, 4)],
            target_latent_t=2,
            imgvid_cond_num_frames=1,
            seed=42,
            noise_aug=1.0,
        ),
        image_rows,
    )
    torch.testing.assert_close(
        minimax_h3_audio_cond_noise_aug_rows(
            audio_rows,
            condition_audio_t=[3],
            seed=42,
            noise_aug=1.0,
        ),
        audio_rows,
    )

    first = minimax_h3_audio_cond_noise_aug_rows(
        audio_rows,
        condition_audio_t=[3],
        seed=42,
        noise_aug=0.25,
    )
    second = minimax_h3_audio_cond_noise_aug_rows(
        audio_rows,
        condition_audio_t=[3],
        seed=42,
        noise_aug=0.25,
    )
    torch.testing.assert_close(first, second)


def test_condition_noise_accepts_a_reference_video_longer_than_the_target():
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
        minimax_h3_imgvid_cond_noise_aug_rows,
    )

    rows = torch.zeros(4 * 2 * 2, 96)
    result = minimax_h3_imgvid_cond_noise_aug_rows(
        rows,
        condition_shapes=[(4, 4, 4)],
        target_latent_t=2,
        imgvid_cond_num_frames=1,
        seed=7,
        noise_aug=0.5,
    )
    assert result.shape == rows.shape


def test_denoise_branch_reuses_packed_input_workspaces():
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_ROW_WIDTH,
        MINIMAX_H3_VIDEO_ROW_WIDTH,
        MiniMaxH3DenoiseBranch,
    )
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    packed = minimax_h3_packed_sequence(
        text_len=2,
        latent_t=1,
        latent_h=2,
        latent_w=2,
        audio_t=1,
        include_keyframe_cond=False,
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(2, 3),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    video_first = torch.full((branch.img_pos.numel(), MINIMAX_H3_VIDEO_ROW_WIDTH), 1.0)
    audio_first = torch.full((branch.audio_pos.numel(), MINIMAX_H3_AUDIO_ROW_WIDTH), 2.0)
    first = branch.forward_kwargs(
        video_rows=video_first,
        audio_rows=audio_first,
        t_video=0.2,
        t_audio=0.3,
        imgvid_cond_timestep=0.999,
        audio_ref_cond_timestep=1.0,
    )

    video_second = torch.full((branch.img_pos.numel(), MINIMAX_H3_VIDEO_ROW_WIDTH), 3.0)
    audio_second = torch.full((branch.audio_pos.numel(), MINIMAX_H3_AUDIO_ROW_WIDTH), 4.0)
    second = branch.forward_kwargs(
        video_rows=video_second,
        audio_rows=audio_second,
        t_video=0.4,
        t_audio=0.5,
        imgvid_cond_timestep=0.999,
        audio_ref_cond_timestep=1.0,
    )

    assert second["x"] is first["x"]
    assert second["audio_x"] is first["audio_x"]
    torch.testing.assert_close(second["x"][0, branch.img_pos_dev], video_second)
    torch.testing.assert_close(second["audio_x"][0, branch.audio_pos_dev], audio_second)


def test_denoise_loop_reuses_state_storage_across_steps():
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_ROW_WIDTH,
        MINIMAX_H3_VIDEO_ROW_WIDTH,
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    packed = minimax_h3_packed_sequence(
        text_len=2,
        latent_t=1,
        latent_h=2,
        latent_w=2,
        audio_t=1,
        include_keyframe_cond=False,
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(2, 3),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )

    def model(**kwargs):
        return (
            torch.zeros(kwargs["update_mask"].numel(), MINIMAX_H3_VIDEO_ROW_WIDTH),
            torch.zeros(kwargs["audio_pos_info"]["position_ids"].numel(), MINIMAX_H3_AUDIO_ROW_WIDTH),
        )

    state_ptrs: list[tuple[int, int]] = []
    minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=torch.ones(branch.img_pos.numel(), MINIMAX_H3_VIDEO_ROW_WIDTH),
        initial_audio_rows=torch.ones(branch.audio_pos.numel(), MINIMAX_H3_AUDIO_ROW_WIDTH),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
        on_step=lambda _step, video, audio: state_ptrs.append((video.data_ptr(), audio.data_ptr())),
    )

    assert len(set(state_ptrs)) == 1


@pytest.mark.parametrize(
    ("layout", "t_video", "t_audio", "imgvid_condition_timestep", "audio_condition_timestep"),
    [
        ("t2va", 0.2, 0.3, 0.999, 1.0),
        ("fl2va", 0.2, 0.3, 0.999, 1.0),
        ("ref2va", 0.2, 0.3, 0.999, 1.0),
        ("ref2va", 0.5, 0.5, 0.5, 0.5),
    ],
)
def test_denoise_branch_reuses_exact_timestep_inverse_metadata(
    layout,
    t_video,
    t_audio,
    imgvid_condition_timestep,
    audio_condition_timestep,
):
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_ROW_WIDTH,
        MINIMAX_H3_VIDEO_ROW_WIDTH,
        MiniMaxH3DenoiseBranch,
    )
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    common = dict(text_len=2, latent_t=1, latent_h=2, latent_w=2, audio_t=1)
    if layout == "ref2va":
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 2, "latent_w": 2},
                {"kind": "audio", "ref_audio_t": 1},
            ],
        )
    else:
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=layout == "fl2va",
            keyframe_frame_indices=[0] if layout == "fl2va" else None,
            frame_count=1 if layout == "fl2va" else None,
        )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(2, 3),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    video_rows = torch.zeros(branch.img_pos.numel(), MINIMAX_H3_VIDEO_ROW_WIDTH)
    audio_rows = torch.zeros(branch.audio_pos.numel(), MINIMAX_H3_AUDIO_ROW_WIDTH)

    expected_timesteps = torch.full((branch.seq_len,), t_video, dtype=torch.float32)
    expected_timesteps[branch.img_pos_dev[branch.update_mask_dev]] = t_video
    expected_timesteps[branch.img_pos_dev[~branch.update_mask_dev]] = imgvid_condition_timestep
    expected_timesteps[branch.audio_pos_dev[branch.audio_update_mask_dev]] = t_audio
    expected_timesteps[branch.audio_pos_dev[~branch.audio_update_mask_dev]] = audio_condition_timestep
    expected_unique, expected_inverse = torch.unique(expected_timesteps, sorted=True, return_inverse=True)

    first = branch.forward_kwargs(
        video_rows=video_rows,
        audio_rows=audio_rows,
        t_video=t_video,
        t_audio=t_audio,
        imgvid_cond_timestep=imgvid_condition_timestep,
        audio_ref_cond_timestep=audio_condition_timestep,
    )
    torch.testing.assert_close(first["unique_timesteps"], expected_unique)
    torch.testing.assert_close(first["inverse_indices"], expected_inverse)

    second = branch.forward_kwargs(
        video_rows=video_rows,
        audio_rows=audio_rows,
        t_video=t_video,
        t_audio=t_audio,
        imgvid_cond_timestep=imgvid_condition_timestep,
        audio_ref_cond_timestep=audio_condition_timestep,
    )
    assert second["inverse_indices"] is first["inverse_indices"]
