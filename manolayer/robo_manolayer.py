import os
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Optional

import torch
import roma

from .helper import ready_arguments
from .manolayer import DEFAULT_MANO_ASSETS_ROOT, ManoLayer
from .robo_mano_utils import (
    MANO_LINK_PARENT_IDXS,
    MANO_OUTPUT_REORDER_IDXS,
    MANO_REDUCED_QPOS_LINKS,
    RoboManoXmlBuilder,
    axis_rotation_transform,
    build_link_bind_matrices,
    build_skinning_bind_matrices,
    build_xml_bind_data_from_tensors,
    export_link_meshes,
    forward_kinematics_from_xml_qpos,
    hand_rotations_from_matrices,
    link_poses_to_matrices,
    matrices_to_link_poses,
    reduced_angles_from_xml_rots,
    xml_quats_from_mano_hand_rots,
)

RoboManoOutput = namedtuple("RoboManoOutput", ["verts", "joints"])


class RoboManoLayer(torch.nn.Module):
    def __init__(
        self,
        side: str = "right",
        mano_assets_root: str = DEFAULT_MANO_ASSETS_ROOT,
        *,
        betas: torch.Tensor,
    ):
        super().__init__()
        self.side = side
        self.mano_assets_root = os.path.expanduser(mano_assets_root)
        mano_assets_path = os.path.join(
            self.mano_assets_root, f"MANO_{side.upper()}.pkl"
        )
        assert os.path.isfile(
            mano_assets_path
        ), f"Can not find MANO assets {mano_assets_path}, please follow steps in README.md"

        smpl_data = ready_arguments(mano_assets_path)
        self.register_buffer(
            "th_betas", torch.from_numpy(smpl_data["betas"]).float().unsqueeze(0)
        )
        self.register_buffer(
            "th_shapedirs", torch.from_numpy(smpl_data["shapedirs"]).float()
        )
        self.register_buffer(
            "th_posedirs", torch.from_numpy(smpl_data["posedirs"]).float()
        )
        self.register_buffer(
            "th_v_template",
            torch.from_numpy(smpl_data["v_template"]).float().unsqueeze(0),
        )
        self.register_buffer(
            "th_J_regressor",
            torch.from_numpy(smpl_data["J_regressor"].toarray()).float(),
        )
        self.register_buffer(
            "th_weights", torch.from_numpy(smpl_data["weights"]).float()
        )
        self.register_buffer("th_faces", torch.from_numpy(smpl_data["f"]).long())

        kintree_table = smpl_data["kintree_table"]
        self.kintree_parents = list(kintree_table[0].tolist())
        self.register_buffer(
            "th_hands_mean",
            torch.from_numpy(smpl_data["hands_mean"].copy()).float().unsqueeze(0),
        )
        self.register_buffer(
            "th_hands_components",
            torch.from_numpy(smpl_data["hands_components"]).float(),
        )
        self.register_buffer("_shape_betas", None)
        self.register_buffer("_shape_v", None)
        self.register_buffer("_shape_J", None)
        self.register_buffer("_link_bind_poses", None)
        self.register_buffer("_link_bind_mats", None)
        self.register_buffer("_link_bind_inv_mats", None)
        self.register_buffer("_skinning_bind_mats", None)

        self._set_beta(betas)

    def _set_beta(self, betas: torch.Tensor):
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)
        if betas.ndim != 2 or betas.shape != self.th_betas.shape:
            raise ValueError(
                f"RoboManoLayer betas must have shape ({self.th_betas.shape[-1]},) "
                f"or {tuple(self.th_betas.shape)}, got {tuple(betas.shape)}."
            )
        betas = betas.to(dtype=self.th_betas.dtype, device=self.th_betas.device)
        shape_blend = torch.matmul(self.th_shapedirs, betas.transpose(1, 0)).permute(
            2, 0, 1
        )
        shaped_vertices = self.th_v_template + shape_blend
        joints = torch.matmul(self.th_J_regressor, shaped_vertices)
        self._shape_betas = betas
        self._shape_v = shaped_vertices
        self._shape_J = joints
        self._update_link_bind_poses()
        return self

    def _update_link_bind_poses(self):
        origins, frames = build_xml_bind_data_from_tensors(
            self._shape_v,
            self._shape_J,
            self.th_v_template,
            self.th_J_regressor,
            self.side,
        )
        bind_mats = build_link_bind_matrices(
            origins,
            frames,
            dtype=self._shape_J.dtype,
            device=self._shape_J.device,
        )
        self._link_bind_mats = bind_mats
        self._link_bind_poses = matrices_to_link_poses(bind_mats.unsqueeze(0))

        self._link_bind_inv_mats = torch.linalg.inv(bind_mats)
        self._skinning_bind_mats = build_skinning_bind_matrices(self._shape_J)

    def pose_to_qpos(
        self,
        pose_coeffs: torch.Tensor,
        global_translation: Optional[torch.Tensor] = None,
        rot_mode: str = "axisang",
        center_idx: Optional[int] = None,
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 15,
    ) -> torch.Tensor:
        if center_idx not in (None, 0):
            raise ValueError(
                "The staged ManoLayer only supports center_idx=None or center_idx=0."
            )
        if rot_mode == "axisang":
            hand_pose_coeffs = pose_coeffs[:, 3:]
            root_pose_coeffs = pose_coeffs[:, :3]
            if use_pca:
                selected_components = self.th_hands_components[:ncomps]
                full_hand_pose = hand_pose_coeffs.mm(selected_components)
            else:
                full_hand_pose = hand_pose_coeffs

            hands_mean = (
                torch.zeros_like(self.th_hands_mean)
                if flat_hand_mean
                else self.th_hands_mean
            )
            full_poses = torch.cat([root_pose_coeffs, hands_mean + full_hand_pose], 1)
            quat_xyzw = roma.rotvec_to_unitquat(full_poses.contiguous().view(-1, 3))
            full_quats = roma.quat_xyzw_to_wxyz(quat_xyzw).view(-1, 16, 4)
        elif rot_mode == "quat":
            full_quats = pose_coeffs.view(-1, 16, 4)
            if use_pca or not flat_hand_mean:
                warnings.warn(
                    "Quat mode doesn't support PCA pose or non flat_hand_mean !"
                )
        else:
            raise NotImplementedError(
                f"Unrecognized rotation mode, expect [axisang|quat], got {rot_mode}"
            )

        zero_trans = torch.zeros_like(pose_coeffs[:, :3])
        joint_trans = (
            zero_trans if center_idx == 0 else zero_trans + self._shape_J[:, 0]
        )
        root_pos = (
            joint_trans
            if global_translation is None
            else joint_trans + global_translation
        )
        hand_rots = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(full_quats[:, 1:]))
        xml_hand_quats = xml_quats_from_mano_hand_rots(hand_rots, self._link_bind_mats)
        return torch.cat(
            [
                root_pos,
                full_quats[:, 0],
                xml_hand_quats.reshape(-1, 15 * 4),
            ],
            dim=1,
        )

    def pose_to_qpos_reduced(
        self,
        pose_coeffs: torch.Tensor,
        global_translation: Optional[torch.Tensor] = None,
        rot_mode: str = "axisang",
        center_idx: Optional[int] = None,
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 15,
    ) -> torch.Tensor:
        full_qpos = self.pose_to_qpos(
            pose_coeffs,
            global_translation,
            rot_mode,
            center_idx,
            use_pca,
            flat_hand_mean,
            ncomps,
        )
        xml_quats = full_qpos[:, 7:].view(full_qpos.shape[0], 15, 4)
        xml_rots = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(xml_quats))
        reduced_angles = reduced_angles_from_xml_rots(xml_rots)
        return torch.cat([full_qpos[:, :7], reduced_angles], dim=1)

    def forward_kinematics(self, qpos: torch.Tensor) -> torch.Tensor:
        return forward_kinematics_from_xml_qpos(qpos, self._link_bind_mats)

    def forward_kinematics_reduced(self, qpos: torch.Tensor) -> torch.Tensor:
        if qpos.ndim != 2 or qpos.shape[-1] != 7 + 20:
            raise ValueError(
                f"RoboMano reduced qpos must have shape (B, 27), got {tuple(qpos.shape)}."
            )

        batch_size = qpos.shape[0]
        root_rot = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(qpos[:, 3:7]))
        link_mats = torch.eye(
            4,
            dtype=qpos.dtype,
            device=qpos.device,
        ).repeat(batch_size, 16, 1, 1)
        link_mats[:, 0, :3, :3] = root_rot
        link_mats[:, 0, :3, 3] = qpos[:, :3]

        bind_mats = self._link_bind_mats.to(dtype=qpos.dtype, device=qpos.device)
        bind_inv_mats = torch.linalg.inv(bind_mats)
        reduced_qpos = qpos[:, 7:]
        cursor = 0

        for finger_name, links in MANO_REDUCED_QPOS_LINKS:
            first_link, second_link, third_link = links
            first_parent = MANO_LINK_PARENT_IDXS[first_link - 1]
            first_rel = torch.matmul(bind_inv_mats[first_parent], bind_mats[first_link])
            first_y = axis_rotation_transform(reduced_qpos[:, cursor], "y")
            cursor += 1
            first_axis = "z" if finger_name == "thumb" else "x"
            first_axis_rot = axis_rotation_transform(
                reduced_qpos[:, cursor],
                first_axis,
            )
            cursor += 1
            link_mats[:, first_link] = torch.matmul(
                torch.matmul(link_mats[:, first_parent], first_rel),
                torch.matmul(first_y, first_axis_rot),
            )

            second_rel = torch.matmul(bind_inv_mats[first_link], bind_mats[second_link])
            second_rot = axis_rotation_transform(reduced_qpos[:, cursor], "x")
            cursor += 1
            link_mats[:, second_link] = torch.matmul(
                torch.matmul(link_mats[:, first_link], second_rel),
                second_rot,
            )

            third_rel = torch.matmul(bind_inv_mats[second_link], bind_mats[third_link])
            third_rot = axis_rotation_transform(reduced_qpos[:, cursor], "x")
            cursor += 1
            link_mats[:, third_link] = torch.matmul(
                torch.matmul(link_mats[:, second_link], third_rel),
                third_rot,
            )

        return matrices_to_link_poses(link_mats)

    def get_verts_joints(self, link_poses: torch.Tensor) -> RoboManoOutput:
        link_mats = link_poses_to_matrices(link_poses)
        if self._link_bind_poses is not None:
            relative_mats = torch.matmul(
                link_mats, self._link_bind_inv_mats.unsqueeze(0)
            )
            transforms_abs = torch.matmul(relative_mats, self._skinning_bind_mats)
        else:
            transforms_abs = link_mats
        batch_size = transforms_abs.shape[0]
        joints = self._shape_J
        shaped_vertices = self._shape_v
        if joints.shape[0] == 1 and batch_size != 1:
            joints = joints.repeat(batch_size, 1, 1)
            shaped_vertices = shaped_vertices.repeat(batch_size, 1, 1)
        elif joints.shape[0] != batch_size:
            raise ValueError(
                "MANO beta batch size must be 1 or match link pose batch size. "
                f"Got betas batch {joints.shape[0]} and link pose batch {batch_size}."
            )
        hand_rot = hand_rotations_from_matrices(transforms_abs)
        flat_rot = torch.eye(3, dtype=link_poses.dtype, device=link_poses.device).view(
            1, 1, 3, 3
        )
        rot_minus_mean_flat = (hand_rot - flat_rot).reshape(batch_size, 15 * 9)
        pose_blend = torch.matmul(
            self.th_posedirs,
            rot_minus_mean_flat.transpose(0, 1),
        ).permute(2, 0, 1)
        pose_blended_template = shaped_vertices + pose_blend

        joint_js = torch.cat([joints, joints.new_zeros(joints.shape[0], 16, 1)], 2)
        transformed_joints = torch.matmul(transforms_abs, joint_js.unsqueeze(3))
        skinning_transforms = (
            transforms_abs
            - torch.cat(
                [
                    transformed_joints.new_zeros(*transformed_joints.shape[:2], 4, 3),
                    transformed_joints,
                ],
                3,
            )
        ).permute(0, 2, 3, 1)

        T = torch.matmul(
            skinning_transforms,
            self.th_weights.transpose(0, 1),
        )
        template_homo = torch.cat(
            [
                pose_blended_template.transpose(2, 1),
                torch.ones(
                    (batch_size, 1, pose_blended_template.shape[1]),
                    dtype=T.dtype,
                    device=T.device,
                ),
            ],
            dim=1,
        ).unsqueeze(1)

        verts = (T * template_homo).sum(2).transpose(2, 1)[..., :3]
        joints = transforms_abs[:, :, :3, 3]

        if self.side == "right":
            tips = verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = verts[:, [745, 317, 445, 556, 673]]

        joints = torch.cat([joints, tips], 1)
        joints = joints[
            :,
            MANO_OUTPUT_REORDER_IDXS,
        ]

        return RoboManoOutput(verts=verts, joints=joints)

    def forward(
        self,
        pose_coeffs: torch.Tensor,
        global_translation: Optional[torch.Tensor] = None,
        rot_mode: str = "axisang",
        center_idx: Optional[int] = None,
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 15,
    ) -> RoboManoOutput:
        qpos = self.pose_to_qpos(
            pose_coeffs,
            global_translation,
            rot_mode,
            center_idx,
            use_pca,
            flat_hand_mean,
            ncomps,
        )
        link_poses = self.forward_kinematics(qpos)
        return self.get_verts_joints(link_poses)

    def export_xml(self, save_folder: str | os.PathLike):
        save_folder = Path(save_folder)
        mesh_folder = save_folder / "meshes"
        mesh_folder.mkdir(parents=True, exist_ok=True)

        origins, frames = build_xml_bind_data_from_tensors(
            self._shape_v,
            self._shape_J,
            self.th_v_template,
            self.th_J_regressor,
            self.side,
        )
        with torch.no_grad():
            vertices = self._shape_v[0].detach().cpu().numpy()
            joints = self._shape_J[0].detach().cpu().numpy()
            faces = self.get_mano_closed_faces().detach().cpu().numpy()
            weights = self.th_weights.detach().cpu().numpy()
        export_link_meshes(
            mesh_folder,
            vertices,
            joints,
            faces,
            weights,
            origins,
            frames,
        )

        reduced_xml = save_folder / f"{self.side}.xml"
        ball_xml = save_folder / f"{self.side}_ball.xml"
        xml_builder = RoboManoXmlBuilder(self.side, origins, frames)
        xml_builder.write(reduced_xml, ball_joints=False)
        xml_builder.write(ball_xml, ball_joints=True)
        return {
            "mesh_folder": mesh_folder,
            "reduced_xml": reduced_xml,
            "ball_xml": ball_xml,
        }

    def get_mano_closed_faces(self):
        return ManoLayer.get_mano_closed_faces(self)
