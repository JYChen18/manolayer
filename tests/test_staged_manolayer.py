import os
from pathlib import Path

import pytest
import torch

from manolayer import LegacyManoLayer, ManoLayer


def _mano_assets_root():
    candidates = [
        os.environ.get("MANO_ASSETS_ROOT"),
        "/Users/jiayi/Desktop/GRAB/models/mano",
        "/Users/jiayi/Desktop/kinder/assets/mano/models",
        "assets/mano/models",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if (root / "MANO_RIGHT.pkl").is_file() and (root / "MANO_LEFT.pkl").is_file():
            return str(root)
    pytest.skip("MANO_RIGHT.pkl and MANO_LEFT.pkl are not available locally.")


def _assert_mano_output_close(actual, expected, atol=1e-5):
    assert torch.allclose(actual.verts, expected.verts, atol=atol, rtol=atol)
    assert torch.allclose(actual.joints, expected.joints, atol=atol, rtol=atol)


@pytest.mark.parametrize("center_idx", [None, 0])
def test_axisang_forward_matches_legacy(center_idx):
    torch.manual_seed(0)
    assets_root = _mano_assets_root()
    legacy = LegacyManoLayer(
        mano_assets_root=assets_root,
        side="right",
        rot_mode="axisang",
        center_idx=center_idx,
    )
    staged = ManoLayer(mano_assets_root=assets_root, side="right")

    pose_coeffs = torch.randn(2, 16 * 3) * 0.2
    betas = torch.randn(2, legacy.th_betas.shape[-1]) * 0.03

    expected = legacy(pose_coeffs, betas=betas)
    actual = staged.forward(
        pose_coeffs,
        betas=betas,
        rot_mode="axisang",
        center_idx=center_idx,
    )

    _assert_mano_output_close(actual, expected)


def test_unsupported_center_idx_raises():
    assets_root = _mano_assets_root()
    staged = ManoLayer(mano_assets_root=assets_root, side="right")
    pose_coeffs = torch.zeros(1, 16 * 3)

    with pytest.raises(ValueError, match="center_idx=None or center_idx=0"):
        staged.pose_to_qpos(pose_coeffs, center_idx=8)


def test_pca_forward_matches_legacy():
    torch.manual_seed(1)
    assets_root = _mano_assets_root()
    ncomps = 6
    legacy = LegacyManoLayer(
        mano_assets_root=assets_root,
        side="left",
        rot_mode="axisang",
        center_idx=0,
        use_pca=True,
        flat_hand_mean=False,
        ncomps=ncomps,
    )
    staged = ManoLayer(mano_assets_root=assets_root, side="left")

    pose_coeffs = torch.randn(2, 3 + ncomps) * 0.2
    betas = torch.randn(2, legacy.th_betas.shape[-1]) * 0.03

    expected = legacy(pose_coeffs, betas=betas)
    actual = staged.forward(
        pose_coeffs,
        betas=betas,
        rot_mode="axisang",
        center_idx=0,
        use_pca=True,
        flat_hand_mean=False,
        ncomps=ncomps,
    )

    _assert_mano_output_close(actual, expected)


def test_quat_forward_matches_legacy():
    torch.manual_seed(2)
    assets_root = _mano_assets_root()
    legacy = LegacyManoLayer(
        mano_assets_root=assets_root,
        side="right",
        rot_mode="quat",
        center_idx=0,
    )
    staged = ManoLayer(mano_assets_root=assets_root, side="right")

    pose_coeffs = torch.nn.functional.normalize(torch.randn(2, 16, 4), dim=-1).view(
        2, -1
    )
    betas = torch.randn(2, legacy.th_betas.shape[-1]) * 0.03

    expected = legacy(pose_coeffs, betas=betas)
    actual = staged.forward(
        pose_coeffs,
        betas=betas,
        rot_mode="quat",
        center_idx=0,
    )

    _assert_mano_output_close(actual, expected)


def test_staged_api_matches_forward_for_root_centering():
    torch.manual_seed(3)
    assets_root = _mano_assets_root()
    layer = ManoLayer(mano_assets_root=assets_root, side="right")

    pose_coeffs = torch.randn(2, 16 * 3) * 0.2
    betas = torch.randn(2, layer.th_betas.shape[-1]) * 0.03

    expected = layer.forward(
        pose_coeffs,
        betas=betas,
        rot_mode="axisang",
        center_idx=0,
    )

    layer.update_beta(betas)
    qpos = layer.pose_to_qpos(pose_coeffs, rot_mode="axisang", center_idx=0)
    link_poses = layer.forward_kinematics(qpos)
    actual = layer.get_verts_joints(link_poses)

    assert qpos.shape == (2, 67)
    assert link_poses.shape == (2, 16, 7)
    assert torch.allclose(actual.verts, expected.verts, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual.joints, expected.joints, atol=1e-5, rtol=1e-5)
