import pytest
import roma
import torch

from manolayer import RoboManoLayer


def _body_names_for_ball_xml():
    return [
        "palm",
        "index1",
        "index2",
        "index3",
        "middle1",
        "middle2",
        "middle3",
        "pinky1",
        "pinky2",
        "pinky3",
        "ring1",
        "ring2",
        "ring3",
        "thumb1",
        "thumb2",
        "thumb3",
    ]


def _body_names_for_reduced_xml():
    return [
        "palm",
        "index1x",
        "index2",
        "index3",
        "middle1x",
        "middle2",
        "middle3",
        "pinky1x",
        "pinky2",
        "pinky3",
        "ring1x",
        "ring2",
        "ring3",
        "thumb1z",
        "thumb2",
        "thumb3",
    ]


def _assert_mujoco_tracks_robomano(model, data, body_names, qpos, link_poses, output):
    mujoco = pytest.importorskip("mujoco")
    root_rot = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(qpos[0, 3:7]))
    root_pos = qpos[0, :3]

    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    ]
    body_pos = torch.from_numpy(data.xpos[body_ids].copy()).to(link_poses.dtype)
    body_pos = body_pos @ root_rot.cpu().T + root_pos.cpu()
    assert torch.allclose(body_pos, link_poses[0, :, :3].cpu(), atol=1e-5, rtol=1e-5)

    mesh_vertices = []
    for mesh_id in range(model.nmesh):
        geom_id = next(
            geom_idx
            for geom_idx in range(model.ngeom)
            if model.geom_dataid[geom_idx] == mesh_id
        )
        start = model.mesh_vertadr[mesh_id]
        end = start + model.mesh_vertnum[mesh_id]
        local_verts = model.mesh_vert[start:end]
        geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
        world_verts = local_verts @ geom_rot.T + data.geom_xpos[geom_id]
        world_verts = torch.from_numpy(world_verts.copy()).to(output.verts.dtype)
        world_verts = world_verts @ root_rot.cpu().T + root_pos.cpu()
        mesh_vertices.append(world_verts)

    xml_verts = torch.cat(mesh_vertices, dim=0)
    mano_verts = output.verts[0].detach().cpu()
    dists = torch.cdist(xml_verts, mano_verts)

    assert dists.min(dim=1).values.mean() < 0.005
    assert dists.min(dim=0).values.mean() < 0.005


def test_exported_ball_xml_tracks_robomano_articulation(
    mano_assets_root,
    tmp_path,
):
    mujoco = pytest.importorskip("mujoco")
    torch.manual_seed(2)
    betas = torch.randn(10) * 0.03
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=betas,
    )
    paths = layer.export_xml(tmp_path)

    pose_coeffs = torch.randn(1, 16 * 3) * 0.2
    global_translation = torch.tensor([[0.05, -0.02, 0.03]])
    qpos = layer.pose_to_qpos(
        pose_coeffs,
        global_translation=global_translation,
        center_idx=0,
    )
    link_poses = layer.forward_kinematics(qpos)
    output = layer.get_verts_joints(link_poses)

    model = mujoco.MjModel.from_xml_path(str(paths["ball_xml"]))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0, 7:].detach().cpu().numpy()
    mujoco.mj_forward(model, data)

    _assert_mujoco_tracks_robomano(
        model,
        data,
        _body_names_for_ball_xml(),
        qpos,
        link_poses,
        output,
    )


def test_exported_reduced_xml_tracks_reduced_forward_kinematics(
    mano_assets_root,
    tmp_path,
):
    mujoco = pytest.importorskip("mujoco")
    torch.manual_seed(3)
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.randn(10) * 0.03,
    )
    paths = layer.export_xml(tmp_path)

    qpos = torch.zeros(1, 27)
    qpos[:, :3] = torch.tensor([[0.03, -0.01, 0.04]])
    qpos[:, 3] = 1
    qpos[:, 7:] = torch.randn(1, 20) * 0.2
    link_poses = layer.forward_kinematics_reduced(qpos)
    output = layer.get_verts_joints(link_poses)

    model = mujoco.MjModel.from_xml_path(str(paths["reduced_xml"]))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0, 7:].detach().cpu().numpy()
    mujoco.mj_forward(model, data)

    _assert_mujoco_tracks_robomano(
        model,
        data,
        _body_names_for_reduced_xml(),
        qpos,
        link_poses,
        output,
    )
