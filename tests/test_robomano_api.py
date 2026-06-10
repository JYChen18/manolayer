import pytest
import torch

from manolayer import RoboManoLayer
from manolayer.manolayer import ManoLayer


def _assert_robo_output_close(actual, expected, atol=1e-5):
    assert torch.allclose(actual.verts, expected.verts, atol=atol, rtol=atol)
    assert torch.allclose(actual.joints, expected.joints, atol=atol, rtol=atol)


def test_robomano_requires_init_beta(mano_assets_root):
    with pytest.raises(TypeError, match="betas"):
        RoboManoLayer(mano_assets_root=mano_assets_root, side="right")

    with pytest.raises(ValueError, match="RoboManoLayer betas must have shape"):
        RoboManoLayer(
            mano_assets_root=mano_assets_root,
            side="right",
            betas=torch.zeros(2, 10),
        )


def test_robomano_staged_api_roundtrips_xml_link_frames(mano_assets_root):
    torch.manual_seed(0)
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.randn(10) * 0.03,
    )

    pose_coeffs = torch.randn(3, 16 * 3) * 0.2
    expected = layer.forward(pose_coeffs, center_idx=0)

    qpos = layer.pose_to_qpos(pose_coeffs, center_idx=0)
    link_poses = layer.forward_kinematics(qpos)
    actual = layer.get_verts_joints(link_poses)

    assert qpos.shape == (3, 67)
    assert link_poses.shape == (3, 16, 7)
    assert actual.verts.shape == (3, 778, 3)
    assert actual.joints.shape == (3, 21, 3)
    _assert_robo_output_close(actual, expected)


def test_robomano_matches_manolayer_for_nonzero_beta_and_pose(mano_assets_root):
    torch.manual_seed(1)
    betas = torch.randn(10) * 0.03
    pose_coeffs = torch.randn(4, 16 * 3) * 0.2

    mano_layer = ManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        rot_mode="axisang",
        center_idx=0,
    )
    robo_layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=betas,
    )

    expected = mano_layer.forward(pose_coeffs, betas=betas.unsqueeze(0).repeat(4, 1))
    actual = robo_layer.forward(pose_coeffs, center_idx=0)

    _assert_robo_output_close(actual, expected)


def test_left_robomano_matches_manolayer_for_nonzero_beta_and_pose(mano_assets_root):
    torch.manual_seed(4)
    betas = torch.randn(10) * 0.03
    pose_coeffs = torch.randn(3, 16 * 3) * 0.2

    mano_layer = ManoLayer(
        mano_assets_root=mano_assets_root,
        side="left",
        rot_mode="axisang",
        center_idx=0,
    )
    robo_layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="left",
        betas=betas,
    )

    expected = mano_layer.forward(pose_coeffs, betas=betas.unsqueeze(0).repeat(3, 1))
    actual = robo_layer.forward(pose_coeffs, center_idx=0)

    _assert_robo_output_close(actual, expected)


def test_robomano_zero_pose_matches_internal_bind_pose(mano_assets_root):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )

    zero_pose = torch.zeros(2, 16 * 3)
    qpos = layer.pose_to_qpos(zero_pose, center_idx=0)
    link_poses = layer.forward_kinematics(qpos)

    assert torch.allclose(
        link_poses,
        layer._link_bind_poses.repeat(2, 1, 1),
        atol=1e-5,
        rtol=1e-5,
    )

    qpos_reduced = layer.pose_to_qpos_reduced(zero_pose, center_idx=0)
    reduced_link_poses = layer.forward_kinematics_reduced(qpos_reduced)
    assert qpos_reduced.shape == (2, 27)
    assert torch.allclose(
        reduced_link_poses,
        layer._link_bind_poses.repeat(2, 1, 1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_robomano_bind_pose_uses_fixed_rotations_and_beta_translations(
    mano_assets_root,
):
    beta_zero = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )
    beta_changed = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.ones(10) * 0.03,
    )

    assert torch.allclose(
        beta_zero._link_bind_mats[:, :3, :3],
        beta_changed._link_bind_mats[:, :3, :3],
        atol=1e-7,
        rtol=1e-7,
    )
    assert not torch.allclose(
        beta_zero._link_bind_mats[:, :3, 3],
        beta_changed._link_bind_mats[:, :3, 3],
        atol=1e-7,
        rtol=1e-7,
    )
