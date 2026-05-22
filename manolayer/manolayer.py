import os
import warnings
from collections import namedtuple
from typing import Optional

import numpy as np
import torch
import roma

from .helper import ready_arguments

LegacyManoOutput = namedtuple(
    "LegacyManoOutput",
    [
        "verts",
        "joints",
        "center_idx",
        "center_joint",
        "full_poses",
        "betas",
        "transforms_abs",
    ],
)
LegacyManoOutput.__new__.__defaults__ = (None,) * len(LegacyManoOutput._fields)

ManoOutput = namedtuple("ManoOutput", ["verts", "joints"])


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


class LegacyManoLayer(torch.nn.Module):

    def __init__(
        self,
        rot_mode: str = "axisang",
        side: str = "right",
        center_idx: Optional[int] = None,
        mano_assets_root: str = "assets/mano/models",
        use_pca: bool = False,
        flat_hand_mean: bool = True,  # Only used in pca mode
        ncomps: int = 15,  # Only used in pca mode
        **kargs,
    ):
        super().__init__()
        self.center_idx = center_idx
        self.rot_mode = rot_mode
        self.side = side
        self.use_pca = use_pca
        self.mano_assets_root = mano_assets_root
        self.flat_hand_mean = flat_hand_mean
        self.ncomps = ncomps if use_pca else -1

        if rot_mode == "axisang":
            self.rot_dim = 3
        elif rot_mode == "quat":
            self.rot_dim = 4
            if use_pca == True or flat_hand_mean == False:
                warnings.warn(
                    "Quat mode doesn't support PCA pose or non flat_hand_mean !"
                )
        else:
            raise NotImplementedError(
                f"Unrecognized rotation mode, expect [pca|axisang|quat], got {rot_mode}"
            )

        # load model according to side flag
        mano_assets_path = os.path.join(
            mano_assets_root, f"MANO_{side.upper()}.pkl"
        )  # eg.  MANO_RIGHT.pkl
        assert os.path.isfile(
            mano_assets_path
        ), f"Can not find MANO assets {mano_assets_path}, please follow steps in README.md"

        # parse and register stuff
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
        hands_components = smpl_data["hands_components"]

        if rot_mode == "axisang":
            hands_mean = (
                np.zeros(hands_components.shape[1])
                if flat_hand_mean
                else smpl_data["hands_mean"]
            )
            hands_mean = hands_mean.copy()
            hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
            self.register_buffer("th_hands_mean", hands_mean)

        if rot_mode == "axisang" and use_pca == True:
            selected_components = hands_components[:ncomps]
            selected_components = torch.Tensor(selected_components)
            self.register_buffer("th_selected_comps", selected_components)

        # End

    def rotation_by_axisang(self, pose_coeffs):
        batch_size = pose_coeffs.shape[0]
        hand_pose_coeffs = pose_coeffs[:, self.rot_dim :]
        root_pose_coeffs = pose_coeffs[:, : self.rot_dim]
        if self.use_pca:
            full_hand_pose = hand_pose_coeffs.mm(self.th_selected_comps)
        else:
            full_hand_pose = hand_pose_coeffs

        # Concatenate back global rot
        full_poses = torch.cat(
            [root_pose_coeffs, self.th_hands_mean + full_hand_pose], 1
        )

        pose_vec_reshaped = full_poses.contiguous().view(-1, 3)  # (B x N, 3)
        rot_mats = roma.rotvec_to_rotmat(pose_vec_reshaped)  # (B x N, 3, 3)
        # rot_mats = lietorch.SO3.exp(pose_vec_reshaped).matrix()[..., :3, :3]  # (B x N, 3, 3)
        full_rots = rot_mats.view(batch_size, 16, 3, 3)
        rotation_blob = {"full_rots": full_rots, "full_poses": full_poses}
        return rotation_blob

    def rotation_by_quaternion(self, pose_coeffs):
        batch_size = pose_coeffs.shape[0]
        full_quat_poses = roma.quat_wxyz_to_xyzw(
            pose_coeffs.view((batch_size, 16, 4))
        )  # [B. 16, 4]
        full_rots = roma.unitquat_to_rotmat(full_quat_poses)  # [B, 16, 3, 3]
        full_poses = roma.unitquat_to_rotvec(full_quat_poses).reshape(
            batch_size, -1
        )  # [B, 16 x 3]

        rotation_blob = {"full_rots": full_rots, "full_poses": full_poses}
        return rotation_blob

    def skinning_layer(self, full_rots: torch.Tensor, betas: Optional[torch.Tensor]):
        batch_size = full_rots.shape[0]
        n_rot = int(full_rots.shape[1])  # 16

        root_rot = full_rots[:, 0, :, :]  # (B, 3, 3)
        hand_rot = full_rots[:, 1:, :, :]  # (B, 15, 3, 3)
        # Full axis angle representation with root joint

        # ============== Shape Blend Shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # $ B_S = \sum_{n=1}^{|\arrow{\beta}|} \beta_n \mathbf{S}_n $  #Eq.4 in MANO
        _betas = self.th_betas if betas is None else betas
        B_S = torch.matmul(self.th_shapedirs, _betas.transpose(1, 0)).permute(
            2, 0, 1
        )  # (?, 778, 3), ? = 1, or B

        # $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
        J = torch.matmul(self.th_J_regressor, (self.th_v_template + B_S))  # (?, 16, 3)
        if betas is None:
            J = J.repeat(batch_size, 1, 1)  # (B, 16, 3)

        # ============== Pose Blender Shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        flat_rot = torch.eye(
            3, dtype=full_rots.dtype, device=full_rots.device
        )  # (3, 3)
        flat_rot = flat_rot.view(1, 1, 3, 3).repeat(
            batch_size, hand_rot.shape[1], 1, 1
        )  # (B, 15, 3, 3)

        # $ R_n (\arrow{\theta}) -  R_n (\arrow{\theta}^{*}) $
        rot_minus_mean_flat = (hand_rot - flat_rot).reshape(
            batch_size, hand_rot.shape[1] * 9
        )  # (B, 15 x 9)

        # $ B_P = \sum_{n=1}^{9K} (R_n (\arrow{\theta}) -  R_n (\arrow{\theta}^{*})) * \mathbf{P}_n $  #Eq.3 in MANO
        B_P = torch.matmul(
            self.th_posedirs, rot_minus_mean_flat.transpose(0, 1)
        ).permute(
            2, 0, 1
        )  # (B, 778, 3)

        # $ T_P =\bar{\mathbf{T}} + B_S + B_P $ # Eq.2 in MANO
        T_P = self.th_v_template + B_S + B_P

        # ============== Constructing $ G_{k} $ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Global rigid transformation
        root_j = J[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_transf = th_with_zeros(torch.cat([root_rot, root_j], 2))

        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = hand_rot[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = hand_rot[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = hand_rot[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = J[:, lev1_idxs]
        lev2_j = J[:, lev2_idxs]
        lev3_j = J[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_transf.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(
            torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        root_trans_flt = (
            root_transf.unsqueeze(1)
            .repeat(1, 5, 1, 1)
            .view(root_transf.shape[0] * 5, 4, 4)
        )
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(hand_rot.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(
            torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(hand_rot.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(
            torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(hand_rot.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]

        # Eq. 4 in SMPL
        G_k = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_transf_global = G_k

        # ============== Constructing $ G^{\prime}_{k} $ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        joint_js = torch.cat([J, J.new_zeros(batch_size, 16, 1)], 2)
        tmp2 = torch.matmul(G_k, joint_js.unsqueeze(3))
        G_prime_k = (
            G_k - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)
        ).permute(0, 2, 3, 1)

        # ============== Finally, blender skinning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # we define $ T = w_{k, i} * G^{\prime}_k $
        T = torch.matmul(G_prime_k, self.th_weights.transpose(0, 1))  # (B, 4, 4, 778)

        T_P_homo = torch.cat(
            [
                T_P.transpose(2, 1),
                torch.ones(
                    (batch_size, 1, B_P.shape[1]), dtype=T.dtype, device=T.device
                ),
            ],
            dim=1,
        )
        T_P_homo = T_P_homo.unsqueeze(1)  # (B, 1, 4, 778)

        # Eq. 7 in SMPL
        # Theorem: A \cdot B = (A * B^{T}).sum(1) # A is a matrix, B is a vector
        verts = (T * T_P_homo).sum(2).transpose(2, 1)  # (B, 778, 4)
        joints = th_transf_global[:, :, :3, 3]  # (B, 16, 3)
        verts = verts[:, :, :3]  # (B, 778, 3)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == "right":
            tips = verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = verts[:, [745, 317, 445, 556, 673]]

        joints = torch.cat([joints, tips], 1)

        # ** original MANO joint order (right hand)
        #                16-15-14-13-\
        #                             \
        #          17 --3 --2 --1------0
        #        18 --6 --5 --4-------/
        #        19 -12 -11 --10-----/
        #          20 --9 --8 --7---/

        # Reorder joints to match SNAP definition
        joints = joints[
            :,
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        ]

        if self.center_idx is not None:
            center_joint = joints[:, self.center_idx].unsqueeze(1)
        else:  # dummy center joint (B, 1, 3)
            center_joint = torch.zeros_like(joints[:, 0].unsqueeze(1))

        # apply center shift on verts and joints
        joints = joints - center_joint
        verts = verts - center_joint

        # apply center shift on global
        global_rot = th_transf_global[:, :, :3, :3]  # (B, 16, 3, 3)
        global_tsl = th_transf_global[:, :, :3, 3:]  # (B, 16, 3, 1)
        global_tsl = global_tsl - center_joint.unsqueeze(-1)  # (B, [16], 3, 1)
        global_transf = torch.cat([global_rot, global_tsl], dim=3)  # (B, 16, 3, 4)
        global_transf = th_with_zeros(global_transf.view(-1, 3, 4))
        global_transf = global_transf.view(batch_size, 16, 4, 4)

        skinning_blob = {
            "verts": verts,
            "joints": joints,
            "center_joint": center_joint,
            "transforms_abs": global_transf,
            "betas": _betas,
        }
        return skinning_blob

    def forward(
        self, pose_coeffs: torch.Tensor, betas: Optional[torch.Tensor] = None, **kwargs
    ) -> LegacyManoOutput:
        if self.rot_mode == "axisang":
            rot_blob = self.rotation_by_axisang(pose_coeffs)
        elif self.rot_mode == "quat":
            rot_blob = self.rotation_by_quaternion(pose_coeffs)

        full_rots = rot_blob["full_rots"]  # TENSOR
        skinning_blob = self.skinning_layer(full_rots, betas)
        output = LegacyManoOutput(
            verts=skinning_blob["verts"],
            joints=skinning_blob["joints"],
            center_idx=self.center_idx,
            center_joint=skinning_blob["center_joint"],
            full_poses=rot_blob["full_poses"],
            betas=skinning_blob["betas"],
            transforms_abs=skinning_blob["transforms_abs"],
        )
        return output

    def get_rotation_center(self, betas: Optional[torch.Tensor] = None):
        """

        V = MANO(theta, beta)

        Then we apply a rotation R on the vertices V
        V_1 = R @ V

        or, we can apply a rotation R on the global components of theta: first 3 elements of the theta
        theta' = CONCAT( SO3.log(R @ SO3.exp(theta[:3])), theta[3:] )

        V_2 = MANO(theta', beta)

        No doubt that, V_1 != V_2
        we found V_1 = V_2 + t, the t is an unknown translation offset

        Directly apply R on V would rotate V w.r.t the rotation center at V's [0,0,0] coordinate.
        However, apply R on the theta[:3] would cause the vertices rotate w.r.t to a rotation center at
        a non-zero, soley beta-sepcified center, C

        In other word, apply any disturb on the theta[:3] would not change the C's coordinates.
        the following code describe how we acquire the rotation center C

        This function will be called at artiboost/utils/refineunit.py in our upcoming work ArtiBoost
        """

        if betas is None:
            betas = self.th_betas

        batch_size = betas.shape[0]
        if self.center_idx is not None:
            return torch.zeros((batch_size, 3), device=betas.device)

        # $ B_S = \sum_{n=1}^{|\arrow{\beta}|} \beta_n \mathbf{S}_n $  #Eq.4 in MANO
        B_S = torch.matmul(self.th_shapedirs, betas.transpose(1, 0)).permute(2, 0, 1)

        # $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
        J = torch.matmul(self.th_J_regressor, (self.th_v_template + B_S))  # (B, 16, 3)

        root_rotation_center = J[:, 0, :].contiguous().view(-1, 3)
        return root_rotation_center

    def get_mano_closed_faces(self):
        """
        The default MANO mesh is "open" at the wrist. By adding additional faces, the hand mesh is closed,
        which looks much better.
        https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py
        """
        close_faces = torch.Tensor(
            [
                [92, 38, 122],
                [234, 92, 122],
                [239, 234, 122],
                [279, 239, 122],
                [215, 279, 122],
                [215, 122, 118],
                [215, 118, 117],
                [215, 117, 119],
                [215, 119, 120],
                [215, 120, 108],
                [215, 108, 79],
                [215, 79, 78],
                [215, 78, 121],
                [214, 215, 121],
            ]
        )
        if self.side == "left":
            close_faces = close_faces[:, [2, 1, 0]]
        th_closed_faces = torch.cat(
            [self.th_faces.clone().detach().cpu(), close_faces.long()]
        )
        # Indices of faces added during closing --> should be ignored as they match the wrist
        # part of the hand, which is not an external surface of the human

        # Valid because added closed faces are at the end
        hand_ignore_faces = [
            1538,
            1539,
            1540,
            1541,
            1542,
            1543,
            1544,
            1545,
            1546,
            1547,
            1548,
            1549,
            1550,
            1551,
        ]
        return th_closed_faces  # , hand_ignore_faces


class ManoLayer(torch.nn.Module):
    def __init__(
        self,
        mano_assets_root: str = "assets/mano/models",
        side: str = "right",
    ):
        super().__init__()
        self.side = side
        self.mano_assets_root = mano_assets_root
        self.center_idx: Optional[int] = None
        self._shape_betas: Optional[torch.Tensor] = None
        self._raw_shape_v: Optional[torch.Tensor] = None
        self._raw_shape_J: Optional[torch.Tensor] = None
        self._shape_v: Optional[torch.Tensor] = None
        self._shape_J: Optional[torch.Tensor] = None

        mano_assets_path = os.path.join(
            mano_assets_root, f"MANO_{side.upper()}.pkl"
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

        self.update_beta()

    @staticmethod
    def _normalize_qpos_shape(qpos: torch.Tensor) -> torch.Tensor:
        if qpos.ndim != 2 or qpos.shape[-1] != 3 + 16 * 4:
            raise ValueError(
                f"MANO qpos must have shape (B, 67), got {tuple(qpos.shape)}."
            )
        return qpos

    def _as_betas(self, betas: Optional[torch.Tensor]) -> torch.Tensor:
        if betas is None:
            return self.th_betas
        if not torch.is_tensor(betas):
            betas = torch.as_tensor(
                betas, dtype=self.th_betas.dtype, device=self.th_betas.device
            )
        else:
            betas = betas.to(dtype=self.th_betas.dtype, device=self.th_betas.device)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)
        if betas.ndim != 2 or betas.shape[-1] != self.th_betas.shape[-1]:
            raise ValueError(
                f"MANO betas must have shape (B, {self.th_betas.shape[-1]}), got {tuple(betas.shape)}."
            )
        return betas

    def update_beta(
        self,
        betas: Optional[torch.Tensor] = None,
    ):
        betas = self._as_betas(betas)
        shape_blend = torch.matmul(
            self.th_shapedirs, betas.transpose(1, 0)
        ).permute(2, 0, 1)
        shaped_vertices = self.th_v_template + shape_blend
        joints = torch.matmul(self.th_J_regressor, shaped_vertices)
        self._shape_betas = betas
        self._raw_shape_v = shaped_vertices
        self._raw_shape_J = joints
        self._apply_center_idx(self.center_idx)
        return self

    def _apply_center_idx(self, center_idx: Optional[int]):
        if center_idx not in (None, 0):
            raise ValueError(
                "The staged ManoLayer only supports center_idx=None or center_idx=0."
            )
        self.center_idx = center_idx
        if self._raw_shape_v is None or self._raw_shape_J is None:
            return
        if center_idx == 0:
            center_joint = self._raw_shape_J[:, 0:1]
            self._shape_v = self._raw_shape_v - center_joint
            self._shape_J = self._raw_shape_J - center_joint
        else:
            self._shape_v = self._raw_shape_v
            self._shape_J = self._raw_shape_J

    def _shape_state_for(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ):
        if self._shape_betas is None:
            self.update_beta()

        betas = self._shape_betas.to(dtype=dtype, device=device)
        shaped_vertices = self._shape_v.to(dtype=dtype, device=device)
        joints = self._shape_J.to(dtype=dtype, device=device)

        if shaped_vertices.shape[0] == 1 and batch_size != 1:
            shaped_vertices = shaped_vertices.repeat(batch_size, 1, 1)
            joints = joints.repeat(batch_size, 1, 1)
            betas = betas.repeat(batch_size, 1)
        elif shaped_vertices.shape[0] != batch_size:
            raise ValueError(
                "MANO beta batch size must be 1 or match qpos batch size. "
                f"Got betas batch {shaped_vertices.shape[0]} and qpos batch {batch_size}."
            )
        return betas, shaped_vertices, joints

    def _pose_to_qpos_and_full_poses(
        self,
        pose_coeffs: torch.Tensor,
        rot_mode: str = "axisang",
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 15,
    ):
        batch_size = pose_coeffs.shape[0]
        if rot_mode == "axisang":
            hand_pose_coeffs = pose_coeffs[:, 3:]
            root_pose_coeffs = pose_coeffs[:, :3]
            if use_pca:
                selected_components = self.th_hands_components[:ncomps].to(
                    dtype=pose_coeffs.dtype, device=pose_coeffs.device
                )
                full_hand_pose = hand_pose_coeffs.mm(selected_components)
            else:
                full_hand_pose = hand_pose_coeffs

            hands_mean = (
                torch.zeros_like(self.th_hands_mean)
                if flat_hand_mean
                else self.th_hands_mean
            ).to(dtype=pose_coeffs.dtype, device=pose_coeffs.device)
            full_poses = torch.cat([root_pose_coeffs, hands_mean + full_hand_pose], 1)
            quat_xyzw = roma.rotvec_to_unitquat(full_poses.contiguous().view(-1, 3))
            qpos = roma.quat_xyzw_to_wxyz(quat_xyzw).view(batch_size, 16, 4)
        elif rot_mode == "quat":
            if use_pca or not flat_hand_mean:
                warnings.warn("Quat mode doesn't support PCA pose or non flat_hand_mean !")
            qpos = pose_coeffs.view(batch_size, 16, 4)
            quat_xyzw = roma.quat_wxyz_to_xyzw(qpos)
            full_poses = roma.unitquat_to_rotvec(quat_xyzw).reshape(batch_size, -1)
        else:
            raise NotImplementedError(
                f"Unrecognized rotation mode, expect [axisang|quat], got {rot_mode}"
            )

        return qpos, full_poses

    def pose_to_qpos(
        self,
        pose_coeffs: torch.Tensor,
        rot_mode: str = "axisang",
        center_idx: Optional[int] = None,
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 15,
    ) -> torch.Tensor:
        self._apply_center_idx(center_idx)
        qpos, _ = self._pose_to_qpos_and_full_poses(
            pose_coeffs,
            rot_mode=rot_mode,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            ncomps=ncomps,
        )
        batch_size = qpos.shape[0]
        _, _, joints = self._shape_state_for(
            batch_size, dtype=qpos.dtype, device=qpos.device
        )
        root_pos = joints[:, 0]
        return torch.cat([root_pos, qpos.reshape(batch_size, 16 * 4)], dim=1)

    def _split_qpos(self, qpos: torch.Tensor):
        qpos = self._normalize_qpos_shape(qpos)
        root_pos = qpos[:, :3]
        full_quats = qpos[:, 3:].view(qpos.shape[0], 16, 4)
        return root_pos, full_quats

    def qpos_to_rotations(self, qpos: torch.Tensor):
        _, full_quats = self._split_qpos(qpos)
        batch_size = full_quats.shape[0]
        quat_xyzw = roma.quat_wxyz_to_xyzw(full_quats)
        full_rots = roma.unitquat_to_rotmat(quat_xyzw)
        full_poses = roma.unitquat_to_rotvec(quat_xyzw).reshape(batch_size, -1)
        return {"full_rots": full_rots, "full_poses": full_poses}

    def forward_kinematics(
        self,
        qpos: torch.Tensor,
    ) -> torch.Tensor:
        root_pos, _ = self._split_qpos(qpos)
        rot_blob = self.qpos_to_rotations(qpos)
        full_rots = rot_blob["full_rots"]
        batch_size = full_rots.shape[0]

        _, _, joints = self._shape_state_for(
            batch_size, dtype=full_rots.dtype, device=full_rots.device
        )

        root_rot = full_rots[:, 0]
        hand_rot = full_rots[:, 1:]
        root_transf = th_with_zeros(torch.cat([root_rot, root_pos.unsqueeze(2)], 2))

        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = hand_rot[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = hand_rot[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = hand_rot[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = joints[:, lev1_idxs]
        lev2_j = joints[:, lev2_idxs]
        lev3_j = joints[:, lev3_idxs]

        all_transforms = [root_transf.unsqueeze(1)]
        lev1_j_rel = lev1_j - joints[:, 0:1]
        lev1_rel_transform_flt = th_with_zeros(
            torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        root_trans_flt = (
            root_transf.unsqueeze(1)
            .repeat(1, 5, 1, 1)
            .view(root_transf.shape[0] * 5, 4, 4)
        )
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(batch_size, 5, 4, 4))

        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(
            torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(batch_size, 5, 4, 4))

        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(
            torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(batch_size, 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        transforms_abs = torch.cat(all_transforms, 1)[:, reorder_idxs]
        quat_wxyz = roma.quat_xyzw_to_wxyz(
            roma.rotmat_to_unitquat(transforms_abs[:, :, :3, :3])
        )
        return torch.cat([transforms_abs[:, :, :3, 3], quat_wxyz], dim=2)

    def _link_poses_to_matrices(self, link_poses: torch.Tensor) -> torch.Tensor:
        if link_poses.ndim != 3 or link_poses.shape[-2:] != (16, 7):
            raise ValueError(
                f"MANO link poses must have shape (B, 16, 7), got {tuple(link_poses.shape)}."
            )
        batch_size = link_poses.shape[0]
        rot = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(link_poses[:, :, 3:]))
        transforms = torch.zeros(
            batch_size,
            16,
            4,
            4,
            dtype=link_poses.dtype,
            device=link_poses.device,
        )
        transforms[:, :, :3, :3] = rot
        transforms[:, :, :3, 3] = link_poses[:, :, :3]
        transforms[:, :, 3, 3] = 1.0
        return transforms

    def _local_rotations_from_link_poses(self, link_poses: torch.Tensor) -> torch.Tensor:
        transforms = self._link_poses_to_matrices(link_poses)
        global_rots = transforms[:, :, :3, :3]
        parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        local_rots = []
        for idx, parent_idx in enumerate(parents):
            if parent_idx < 0:
                local_rots.append(global_rots[:, idx])
            else:
                local_rots.append(
                    torch.matmul(global_rots[:, parent_idx].transpose(1, 2), global_rots[:, idx])
                )
        return torch.stack(local_rots, dim=1)

    def get_verts_joints(self, link_poses: torch.Tensor) -> ManoOutput:
        transforms_abs = self._link_poses_to_matrices(link_poses)
        batch_size = link_poses.shape[0]
        _, shaped_vertices, joints = self._shape_state_for(
            batch_size, dtype=link_poses.dtype, device=link_poses.device
        )
        local_rots = self._local_rotations_from_link_poses(link_poses)
        hand_rot = local_rots[:, 1:]
        flat_rot = torch.eye(3, dtype=link_poses.dtype, device=link_poses.device)
        flat_rot = flat_rot.view(1, 1, 3, 3).repeat(batch_size, 15, 1, 1)
        rot_minus_mean_flat = (hand_rot - flat_rot).reshape(batch_size, 15 * 9)
        pose_blend = torch.matmul(
            self.th_posedirs.to(dtype=link_poses.dtype, device=link_poses.device),
            rot_minus_mean_flat.transpose(0, 1),
        ).permute(2, 0, 1)
        pose_blended_template = shaped_vertices + pose_blend

        joint_js = torch.cat([joints, joints.new_zeros(batch_size, 16, 1)], 2)
        transformed_joints = torch.matmul(transforms_abs, joint_js.unsqueeze(3))
        skinning_transforms = (
            transforms_abs
            - torch.cat(
                [
                    transformed_joints.new_zeros(
                        *transformed_joints.shape[:2], 4, 3
                    ),
                    transformed_joints,
                ],
                3,
            )
        ).permute(0, 2, 3, 1)

        T = torch.matmul(
            skinning_transforms,
            self.th_weights.to(
                dtype=pose_blended_template.dtype,
                device=pose_blended_template.device,
            ).transpose(0, 1),
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
        joints = link_poses[:, :, :3]

        if self.side == "right":
            tips = verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = verts[:, [745, 317, 445, 556, 673]]

        joints = torch.cat([joints, tips], 1)
        joints = joints[
            :,
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        ]

        return ManoOutput(
            verts=verts,
            joints=joints,
        )

    def forward(
        self,
        pose_coeffs: torch.Tensor,
        betas: Optional[torch.Tensor] = None,
        rot_mode: str = "axisang",
        center_idx: Optional[int] = None,
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 15,
        **kwargs,
    ) -> ManoOutput:
        self.update_beta(betas)
        qpos = self.pose_to_qpos(
            pose_coeffs,
            rot_mode=rot_mode,
            center_idx=center_idx,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            ncomps=ncomps,
        )
        link_poses = self.forward_kinematics(qpos)
        return self.get_verts_joints(link_poses)

    def rotation_by_axisang(
        self,
        pose_coeffs: torch.Tensor,
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 15,
    ):
        qpos, full_poses = self._pose_to_qpos_and_full_poses(
            pose_coeffs,
            rot_mode="axisang",
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            ncomps=ncomps,
        )
        quat_xyzw = roma.quat_wxyz_to_xyzw(qpos)
        return {
            "full_rots": roma.unitquat_to_rotmat(quat_xyzw),
            "full_poses": full_poses,
        }

    def rotation_by_quaternion(self, pose_coeffs: torch.Tensor):
        qpos, full_poses = self._pose_to_qpos_and_full_poses(
            pose_coeffs, rot_mode="quat"
        )
        quat_xyzw = roma.quat_wxyz_to_xyzw(qpos)
        return {
            "full_rots": roma.unitquat_to_rotmat(quat_xyzw),
            "full_poses": full_poses,
        }

    def skinning_layer(
        self,
        full_rots: torch.Tensor,
        betas: Optional[torch.Tensor] = None,
    ):
        self.update_beta(betas)
        batch_size = full_rots.shape[0]
        _, _, joints = self._shape_state_for(
            batch_size, dtype=full_rots.dtype, device=full_rots.device
        )
        qpos = torch.cat(
            [
                joints[:, 0],
                roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(full_rots)).reshape(
                    batch_size, 16 * 4
                ),
            ],
            dim=1,
        )
        link_poses = self.forward_kinematics(qpos)
        output = self.get_verts_joints(link_poses)
        return {
            "verts": output.verts,
            "joints": output.joints,
        }

    def get_rotation_center(self, betas: Optional[torch.Tensor] = None):
        betas = self._as_betas(betas)
        batch_size = betas.shape[0]
        if self.center_idx is not None:
            return torch.zeros((batch_size, 3), device=betas.device)

        shape_blend = torch.matmul(
            self.th_shapedirs, betas.transpose(1, 0)
        ).permute(2, 0, 1)
        joints = torch.matmul(self.th_J_regressor, self.th_v_template + shape_blend)
        return joints[:, 0, :].contiguous().view(-1, 3)

    def get_mano_closed_faces(self):
        return LegacyManoLayer.get_mano_closed_faces(self)
