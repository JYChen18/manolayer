import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import roma


MANO_LINK_CHILD_IDXS = list(range(1, 16))
MANO_LINK_PARENT_IDXS = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
MANO_OUTPUT_REORDER_IDXS = [
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
]

MANO_LINK_MESH_NAMES = {
    0: "palm",
    1: "index1x",
    2: "index2",
    3: "index3",
    4: "middle1x",
    5: "middle2",
    6: "middle3",
    7: "pinky1x",
    8: "pinky2",
    9: "pinky3",
    10: "ring1x",
    11: "ring2",
    12: "ring3",
    13: "thumb1z",
    14: "thumb2",
    15: "thumb3",
}
MANO_BALL_BODY_NAMES = {
    0: "palm",
    1: "index1",
    2: "index2",
    3: "index3",
    4: "middle1",
    5: "middle2",
    6: "middle3",
    7: "pinky1",
    8: "pinky2",
    9: "pinky3",
    10: "ring1",
    11: "ring2",
    12: "ring3",
    13: "thumb1",
    14: "thumb2",
    15: "thumb3",
}
MANO_FINGER_LINKS = [
    ("index", [1, 2, 3]),
    ("middle", [4, 5, 6]),
    ("pinky", [7, 8, 9]),
    ("ring", [10, 11, 12]),
    ("thumb", [13, 14, 15]),
]
MANO_REDUCED_FIRST_JOINTS = {
    "index": ("index1y", "j_index1y", "index1x", "j_index1x"),
    "middle": ("middle1y", "j_middle1y", "middle1x", "j_middle1x"),
    "pinky": ("pinky1y", "j_pinky1y", "pinky1x", "j_pinky1x"),
    "ring": ("ring1y", "j_ring1y", "ring1x", "j_ring1x"),
    "thumb": ("thumb1y", "j_thumb1y", "thumb1z", "j_thumb1z"),
}
MANO_REDUCED_RANGES = {
    "index1y": "-0.349066 0.349066",
    "finger1x": "-0.174533 1.5708",
    "finger2": "0 1.74533",
    "finger3": "0 1.74533",
    "middle1y": "-0.523599 0.349066",
    "pinky1y": "-0.698132 0.349066",
    "ring1y": "-0.523599 0.349066",
    "thumb1y": "-0.174533 2.61799",
    "thumb1z": "-0.698132 0.698132",
}
MANO_REDUCED_QPOS_LINKS = MANO_FINGER_LINKS
MANO_TERMINAL_TIP_VERTS_RIGHT = {3: 317, 6: 444, 9: 673, 12: 556, 15: 745}
MANO_TERMINAL_TIP_VERTS_LEFT = {3: 317, 6: 445, 9: 673, 12: 556, 15: 745}
MANO_LEFT_MIRROR = np.diag([-1.0, 1.0, 1.0])
MANO_RIGHT_XML_FRAME_QUATS = {
    1: "0.0207461 -0.704984 -0.0206397 -0.708619",
    2: "0.0207461 -0.704984 -0.0206397 -0.708619",
    3: "0.0207461 -0.704984 -0.0206397 -0.708619",
    4: "0.0280283 -0.644307 -0.0115664 -0.764166",
    5: "0.0280283 -0.644307 -0.0115664 -0.764166",
    6: "0.0280283 -0.644307 -0.0115664 -0.764166",
    7: "0.114604 0.469585 0.0826793 0.871505",
    8: "0.114604 0.469585 0.0826793 0.871505",
    9: "0.114604 0.469585 0.0826793 0.871505",
    10: "0.0233679 0.585251 0.0590453 0.808362",
    11: "0.0233679 0.585251 0.0590453 0.808362",
    12: "0.0233679 0.585251 0.0590453 0.808362",
    13: "0.457776 -0.494578 -0.459441 -0.578574",
    14: "-0.303832 0.79185 -0.375505 0.373706",
    15: "-0.303832 0.79185 -0.375505 0.373706",
}


def format_float(value: float) -> str:
    text = f"{float(value):.8g}"
    return "0" if text == "-0" else text


def format_vec(values) -> str:
    return " ".join(format_float(value) for value in values)


def normalize_np(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        vec = fallback
        norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=float)
    return vec / norm


def frame_from_direction(direction: np.ndarray) -> np.ndarray:
    x_axis = normalize_np(-direction, np.array([1.0, 0.0, 0.0]))
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(x_axis, ref))) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    y_axis = normalize_np(
        ref - np.dot(ref, x_axis) * x_axis,
        np.array([0.0, 1.0, 0.0]),
    )
    z_axis = normalize_np(np.cross(x_axis, y_axis), np.array([0.0, 0.0, 1.0]))
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix_tensor = torch.from_numpy(matrix).float()
    quat_xyzw = roma.rotmat_to_unitquat(matrix_tensor)
    return roma.quat_xyzw_to_wxyz(quat_xyzw).detach().cpu().numpy()


def quat_wxyz_to_matrix(quat: str) -> np.ndarray:
    quat_tensor = torch.tensor(
        [float(value) for value in quat.split()],
        dtype=torch.float32,
    )
    matrix = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(quat_tensor))
    return matrix.detach().cpu().numpy()


def mano_xml_frame_matrix(side: str, link_idx: int) -> np.ndarray:
    right_frame = quat_wxyz_to_matrix(MANO_RIGHT_XML_FRAME_QUATS[link_idx])
    if side == "right":
        return right_frame
    if side == "left":
        return MANO_LEFT_MIRROR @ right_frame @ MANO_LEFT_MIRROR
    raise ValueError(f"Unsupported MANO side {side}.")


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray):
    with path.open("wb") as stl_file:
        stl_file.write(b"RoboManoLayer export".ljust(80, b"\0"))
        stl_file.write(struct.pack("<I", int(faces.shape[0])))
        for face in faces:
            tri = vertices[face]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal = normalize_np(normal, np.zeros(3))
            stl_file.write(
                struct.pack(
                    "<12fH",
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    float(tri[0, 0]),
                    float(tri[0, 1]),
                    float(tri[0, 2]),
                    float(tri[1, 0]),
                    float(tri[1, 1]),
                    float(tri[1, 2]),
                    float(tri[2, 0]),
                    float(tri[2, 1]),
                    float(tri[2, 2]),
                    0,
                )
            )


def link_poses_to_matrices(link_poses: torch.Tensor) -> torch.Tensor:
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


def matrices_to_link_poses(transforms: torch.Tensor) -> torch.Tensor:
    quat_wxyz = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(transforms[..., :3, :3]))
    return torch.cat([transforms[..., :3, 3], quat_wxyz], dim=-1)


def hand_rotations_from_matrices(transforms: torch.Tensor) -> torch.Tensor:
    global_rots = transforms[:, :, :3, :3]
    parent_rots = global_rots[:, MANO_LINK_PARENT_IDXS]
    child_rots = global_rots[:, MANO_LINK_CHILD_IDXS]
    return torch.matmul(parent_rots.transpose(2, 3), child_rots)


def axis_rotation_transform(angles: torch.Tensor, axis: str) -> torch.Tensor:
    axis_vectors = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    if axis not in axis_vectors:
        raise ValueError(f"Unsupported axis {axis}.")

    axis_vector = angles.new_tensor(axis_vectors[axis])
    rotvecs = angles.unsqueeze(1) * axis_vector
    transforms = torch.eye(
        4,
        dtype=angles.dtype,
        device=angles.device,
    ).repeat(angles.shape[0], 1, 1)
    transforms[:, :3, :3] = roma.rotvec_to_rotmat(rotvecs)
    return transforms


def rotation_transform_from_matrix(rotations: torch.Tensor) -> torch.Tensor:
    transforms = torch.eye(
        4,
        dtype=rotations.dtype,
        device=rotations.device,
    ).repeat(rotations.shape[0], 1, 1)
    transforms[:, :3, :3] = rotations
    return transforms


def link_body_transform(
    parent_mats: torch.Tensor,
    bind_mats: torch.Tensor,
    bind_inv_mats: torch.Tensor,
    parent_idx: int,
    link_idx: int,
    joint_rotations: torch.Tensor,
) -> torch.Tensor:
    bind_rel = torch.matmul(bind_inv_mats[parent_idx], bind_mats[link_idx])
    joint_transform = rotation_transform_from_matrix(joint_rotations)
    return torch.matmul(
        torch.matmul(parent_mats, bind_rel),
        joint_transform,
    )


def forward_kinematics_from_xml_qpos(
    qpos: torch.Tensor,
    link_bind_mats: torch.Tensor,
) -> torch.Tensor:
    if qpos.ndim != 2 or qpos.shape[-1] != 3 + 16 * 4:
        raise ValueError(
            f"RoboMano qpos must have shape (B, 67), got {tuple(qpos.shape)}."
        )

    batch_size = qpos.shape[0]
    root_rot = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(qpos[:, 3:7]))
    xml_hand_quats = qpos[:, 7:].view(batch_size, 15, 4)
    xml_hand_rots = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(xml_hand_quats))
    link_mats = torch.eye(
        4,
        dtype=qpos.dtype,
        device=qpos.device,
    ).repeat(batch_size, 16, 1, 1)
    link_mats[:, 0, :3, :3] = root_rot
    link_mats[:, 0, :3, 3] = qpos[:, :3]

    bind_mats = link_bind_mats.to(dtype=qpos.dtype, device=qpos.device)
    bind_inv_mats = torch.linalg.inv(bind_mats)
    for link_idx, parent_idx in zip(MANO_LINK_CHILD_IDXS, MANO_LINK_PARENT_IDXS):
        link_mats[:, link_idx] = link_body_transform(
            link_mats[:, parent_idx],
            bind_mats,
            bind_inv_mats,
            parent_idx,
            link_idx,
            xml_hand_rots[:, link_idx - 1],
        )

    return matrices_to_link_poses(link_mats)


def reduced_angles_from_xml_rots(xml_rots: torch.Tensor) -> torch.Tensor:
    angles = []
    for finger_name, links in MANO_REDUCED_QPOS_LINKS:
        first_rot = xml_rots[:, links[0] - 1]
        if finger_name == "thumb":
            first_y = torch.atan2(first_rot[:, 0, 2], first_rot[:, 2, 2])
            first_z = torch.atan2(first_rot[:, 1, 0], first_rot[:, 1, 1])
            angles.extend([first_y, first_z])
        else:
            first_y = torch.atan2(-first_rot[:, 2, 0], first_rot[:, 0, 0])
            first_x = torch.atan2(-first_rot[:, 1, 2], first_rot[:, 1, 1])
            angles.extend([first_y, first_x])

        for link_idx in links[1:]:
            rot = xml_rots[:, link_idx - 1]
            angles.append(torch.atan2(-rot[:, 1, 2], rot[:, 1, 1]))
    return torch.stack(angles, dim=1)


def xml_quats_from_mano_hand_rots(
    hand_rots: torch.Tensor,
    link_bind_mats: torch.Tensor,
) -> torch.Tensor:
    link_frames = link_bind_mats[1:, :3, :3]
    xml_rots = torch.matmul(
        link_frames.transpose(1, 2).unsqueeze(0),
        torch.matmul(hand_rots, link_frames.unsqueeze(0)),
    )
    return roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(xml_rots))


def build_skinning_bind_matrices(joints: torch.Tensor) -> torch.Tensor:
    bind_mats = torch.eye(
        4,
        dtype=joints.dtype,
        device=joints.device,
    ).repeat(16, 1, 1)
    bind_mats[:, :3, 3] = joints[0] - joints[0, 0]
    return bind_mats


def build_link_bind_matrices(
    origins: np.ndarray,
    frames: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bind_mats = torch.eye(4, dtype=dtype, device=device).repeat(16, 1, 1)
    bind_mats[:, :3, :3] = torch.from_numpy(frames).to(dtype=dtype, device=device)
    bind_mats[:, :3, 3] = torch.from_numpy(origins).to(dtype=dtype, device=device)
    return bind_mats


def build_xml_link_frames(
    vertices: np.ndarray,
    origins: np.ndarray,
    side: str,
) -> np.ndarray:
    frames = np.repeat(np.eye(3)[None], 16, axis=0)
    if side in ("right", "left"):
        for link_idx in MANO_RIGHT_XML_FRAME_QUATS:
            frames[link_idx] = mano_xml_frame_matrix(side, link_idx)
        return frames

    child_map = {
        1: 2,
        2: 3,
        4: 5,
        5: 6,
        7: 8,
        8: 9,
        10: 11,
        11: 12,
        13: 14,
        14: 15,
    }
    tip_map = (
        MANO_TERMINAL_TIP_VERTS_RIGHT
        if side == "right"
        else MANO_TERMINAL_TIP_VERTS_LEFT
    )
    parent_map = {
        child: parent
        for child, parent in zip(MANO_LINK_CHILD_IDXS, MANO_LINK_PARENT_IDXS)
    }

    for link_idx in range(1, 16):
        if link_idx in child_map:
            target = origins[child_map[link_idx]]
        else:
            target = vertices[tip_map[link_idx]]
        direction = target - origins[link_idx]
        if np.linalg.norm(direction) < 1e-8:
            direction = origins[link_idx] - origins[parent_map[link_idx]]
        frames[link_idx] = frame_from_direction(direction)
    return frames


def build_xml_bind_data(
    vertices: np.ndarray,
    joints: np.ndarray,
    zero_vertices: np.ndarray,
    zero_joints: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    root = joints[0].copy()
    zero_root = zero_joints[0].copy()
    origins = joints - root
    zero_vertices = zero_vertices - zero_root
    zero_origins = zero_joints - zero_root
    frames = build_xml_link_frames(zero_vertices, zero_origins, side)
    return origins, frames


def build_xml_bind_data_from_tensors(
    shaped_vertices: torch.Tensor,
    shaped_joints: torch.Tensor,
    template_vertices: torch.Tensor,
    joint_regressor: torch.Tensor,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        vertices = shaped_vertices[0].detach().cpu().numpy()
        joints = shaped_joints[0].detach().cpu().numpy()
        zero_vertices = template_vertices[0].detach().cpu().numpy()
        zero_joints = (
            torch.matmul(joint_regressor, template_vertices).detach().cpu().numpy()[0]
        )
    return build_xml_bind_data(vertices, joints, zero_vertices, zero_joints, side)


def export_link_meshes(
    mesh_folder: Path,
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    origins: np.ndarray,
    frames: np.ndarray,
):
    root = joints[0].copy()
    vertices = vertices - root
    face_links = weights[faces].sum(axis=1).argmax(axis=1)

    for link_idx, mesh_name in MANO_LINK_MESH_NAMES.items():
        local_vertices = vertices - origins[link_idx]
        link_faces = faces[face_links == link_idx]
        write_binary_stl(
            mesh_folder / f"{mesh_name}.stl",
            local_vertices,
            link_faces,
        )


class RoboManoXmlBuilder:
    def __init__(self, side: str, origins: np.ndarray, frames: np.ndarray):
        self.side = side
        self.origins = origins
        self.frames = frames

    def _add_inertial(self, body: ET.Element):
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": "1e-3",
                "diaginertia": "1e-4 1e-4 1e-4",
            },
        )

    def write(self, xml_path: Path, ball_joints: bool):
        root = ET.Element(
            "mujoco",
            {"model": f"mano_{self.side}{'_ball' if ball_joints else ''}"},
        )
        ET.SubElement(
            root,
            "compiler",
            {"angle": "radian", "meshdir": "meshes", "autolimits": "true"},
        )
        option = ET.SubElement(
            root,
            "option",
            {
                "impratio": "10",
                "integrator": "implicitfast",
                "cone": "elliptic",
                "noslip_iterations": "2",
            },
        )
        ET.SubElement(option, "flag", {"gravity": "disable", "nativeccd": "enable"})
        self._add_defaults(root, ball_joints)
        self._add_assets(root)
        worldbody = ET.SubElement(root, "worldbody")
        palm = ET.SubElement(worldbody, "body", {"name": "palm"})
        ET.SubElement(palm, "geom", {"name": "c_palm", "type": "mesh", "mesh": "palm"})
        if ball_joints:
            self._add_ball_bodies(palm)
        else:
            self._add_reduced_bodies(palm)
        self._add_contacts(root, ball_joints)
        self._add_actuators(root, ball_joints)

        ET.indent(root, space="  ")
        ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=False)

    def _add_defaults(self, root: ET.Element, ball_joints: bool):
        class_name = f"{self.side}_hand"
        defaults = ET.SubElement(root, "default")
        hand_defaults = ET.SubElement(defaults, "default", {"class": class_name})
        ET.SubElement(
            hand_defaults,
            "joint",
            {"damping": "0.05", "armature": "0.0002", "frictionloss": "0.01"},
        )
        if ball_joints:
            ET.SubElement(
                hand_defaults,
                "position",
                {"kp": "5", "ctrlrange": "-3.14159 3.14159"},
            )
            return

        ET.SubElement(hand_defaults, "position", {"kp": "5"})
        for name, ctrlrange in MANO_REDUCED_RANGES.items():
            child_default = ET.SubElement(hand_defaults, "default", {"class": name})
            ET.SubElement(child_default, "position", {"ctrlrange": ctrlrange})

    def _add_assets(self, root: ET.Element):
        asset = ET.SubElement(root, "asset")
        for mesh_name in MANO_LINK_MESH_NAMES.values():
            ET.SubElement(
                asset,
                "mesh",
                {"name": mesh_name, "file": f"{mesh_name}.stl"},
            )

    def _body_transform_attrs(self, link_idx: int, parent_idx: int):
        rel_pos = (self.origins[link_idx] - self.origins[parent_idx]) @ self.frames[
            parent_idx
        ]
        rel_rot = self.frames[parent_idx].T @ self.frames[link_idx]
        attrs = {"pos": format_vec(rel_pos)}
        if not np.allclose(rel_rot, np.eye(3), atol=1e-7):
            attrs["quat"] = format_vec(matrix_to_quat_wxyz(rel_rot))
        return attrs

    def _geom_attrs(self, geom_name: str, mesh_name: str, link_idx: int):
        attrs = {"name": geom_name, "type": "mesh", "mesh": mesh_name}
        geom_rot = self.frames[link_idx].T
        if not np.allclose(geom_rot, np.eye(3), atol=1e-7):
            attrs["quat"] = format_vec(matrix_to_quat_wxyz(geom_rot))
        return attrs

    def _add_ball_bodies(self, palm: ET.Element):
        body_by_link = {0: palm}
        parent_map = {
            child: parent
            for child, parent in zip(MANO_LINK_CHILD_IDXS, MANO_LINK_PARENT_IDXS)
        }
        for _, links in MANO_FINGER_LINKS:
            for link_idx in links:
                body_name = MANO_BALL_BODY_NAMES[link_idx]
                parent_idx = parent_map[link_idx]
                body = ET.SubElement(
                    body_by_link[parent_idx],
                    "body",
                    {
                        "name": body_name,
                        **self._body_transform_attrs(link_idx, parent_idx),
                    },
                )
                self._add_inertial(body)
                ET.SubElement(
                    body,
                    "joint",
                    {
                        "name": f"j_{body_name}",
                        "type": "ball",
                        "pos": "0 0 0",
                        "actuatorfrcrange": "-100 100",
                    },
                )
                mesh_name = MANO_LINK_MESH_NAMES[link_idx]
                ET.SubElement(
                    body,
                    "geom",
                    self._geom_attrs(f"c_{mesh_name}", mesh_name, link_idx),
                )
                body_by_link[link_idx] = body

    def _add_reduced_bodies(self, palm: ET.Element):
        for finger_name, links in MANO_FINGER_LINKS:
            first_link, second_link, third_link = links
            y_body_name, y_joint_name, x_body_name, x_joint_name = (
                MANO_REDUCED_FIRST_JOINTS[finger_name]
            )
            y_body = ET.SubElement(
                palm,
                "body",
                {
                    "name": y_body_name,
                    **self._body_transform_attrs(first_link, 0),
                },
            )
            self._add_inertial(y_body)
            ET.SubElement(
                y_body,
                "joint",
                {
                    "name": y_joint_name,
                    "pos": "0 0 0",
                    "axis": "0 1 0",
                    "range": MANO_REDUCED_RANGES[y_body_name],
                    "actuatorfrcrange": "-100 100",
                },
            )

            x_body = ET.SubElement(y_body, "body", {"name": x_body_name})
            self._add_inertial(x_body)
            x_axis = "0 0 1" if finger_name == "thumb" else "1 0 0"
            x_class = "thumb1z" if finger_name == "thumb" else "finger1x"
            ET.SubElement(
                x_body,
                "joint",
                {
                    "name": x_joint_name,
                    "pos": "0 0 0",
                    "axis": x_axis,
                    "range": MANO_REDUCED_RANGES[x_class],
                    "actuatorfrcrange": "-100 100",
                },
            )
            first_mesh = MANO_LINK_MESH_NAMES[first_link]
            ET.SubElement(
                x_body,
                "geom",
                self._geom_attrs(f"c_{first_mesh}", first_mesh, first_link),
            )

            second_body = self._add_reduced_child_body(
                x_body,
                second_link,
                first_link,
                "finger2",
            )
            self._add_reduced_child_body(
                second_body,
                third_link,
                second_link,
                "finger3",
            )

    def _add_reduced_child_body(
        self,
        parent_body: ET.Element,
        link_idx: int,
        parent_idx: int,
        joint_class: str,
    ):
        body_name = MANO_LINK_MESH_NAMES[link_idx]
        body = ET.SubElement(
            parent_body,
            "body",
            {
                "name": body_name,
                **self._body_transform_attrs(link_idx, parent_idx),
            },
        )
        self._add_inertial(body)
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"j_{body_name}",
                "pos": "0 0 0",
                "axis": "1 0 0",
                "range": MANO_REDUCED_RANGES[joint_class],
                "actuatorfrcrange": "-100 100",
            },
        )
        ET.SubElement(
            body,
            "geom",
            self._geom_attrs(f"c_{body_name}", body_name, link_idx),
        )
        return body

    def _add_contacts(self, root: ET.Element, ball_joints: bool):
        contact = ET.SubElement(root, "contact")
        first_bodies = (
            ["index1", "middle1", "pinky1", "ring1", "thumb1"]
            if ball_joints
            else ["index1x", "middle1x", "pinky1x", "ring1x", "thumb1z"]
        )
        for body_name in first_bodies:
            ET.SubElement(contact, "exclude", {"body1": "palm", "body2": body_name})
        ET.SubElement(contact, "exclude", {"body1": "palm", "body2": "thumb2"})

    def _add_actuators(self, root: ET.Element, ball_joints: bool):
        actuator = ET.SubElement(root, "actuator")
        side_prefix = "rh" if self.side == "right" else "lh"
        class_name = f"{self.side}_hand"
        if ball_joints:
            gears = {
                "x": "0 0 0 1 0 0",
                "y": "0 0 0 0 1 0",
                "z": "0 0 0 0 0 1",
            }
            for _, links in MANO_FINGER_LINKS:
                for link_idx in links:
                    body_name = MANO_BALL_BODY_NAMES[link_idx]
                    for axis_name, gear in gears.items():
                        ET.SubElement(
                            actuator,
                            "position",
                            {
                                "name": f"{side_prefix}_A_{body_name}_{axis_name}",
                                "joint": f"j_{body_name}",
                                "gear": gear,
                                "class": class_name,
                            },
                        )
            return

        reduced_actuators = [
            ("index1y", "j_index1y", "index1y"),
            ("index1x", "j_index1x", "finger1x"),
            ("index2", "j_index2", "finger2"),
            ("index3", "j_index3", "finger3"),
            ("middle1y", "j_middle1y", "middle1y"),
            ("middle1x", "j_middle1x", "finger1x"),
            ("middle2", "j_middle2", "finger2"),
            ("middle3", "j_middle3", "finger3"),
            ("pinky1y", "j_pinky1y", "pinky1y"),
            ("pinky1x", "j_pinky1x", "finger1x"),
            ("pinky2", "j_pinky2", "finger2"),
            ("pinky3", "j_pinky3", "finger3"),
            ("ring1y", "j_ring1y", "ring1y"),
            ("ring1x", "j_ring1x", "finger1x"),
            ("ring2", "j_ring2", "finger2"),
            ("ring3", "j_ring3", "finger3"),
            ("thumb1y", "j_thumb1y", "thumb1y"),
            ("thumb1z", "j_thumb1z", "thumb1z"),
            ("thumb2", "j_thumb2", "finger2"),
            ("thumb3", "j_thumb3", "finger3"),
        ]
        for actuator_name, joint_name, joint_class in reduced_actuators:
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": f"{side_prefix}_A_{actuator_name}",
                    "joint": joint_name,
                    "class": joint_class,
                },
            )
