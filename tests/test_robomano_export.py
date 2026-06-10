import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import roma
import torch

from manolayer import RoboManoLayer
from manolayer.robo_mano_utils import mano_xml_frame_matrix


def _stl_triangle_count(path: Path):
    with path.open("rb") as stl_file:
        stl_file.seek(80)
        return struct.unpack("<I", stl_file.read(4))[0]


def _vec(text: str):
    return torch.tensor([float(value) for value in text.split()])


def _assert_same_rotation(actual: str, expected: str):
    actual_matrix = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(_vec(actual)))
    expected_matrix = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(_vec(expected)))
    assert torch.allclose(actual_matrix, expected_matrix, atol=3e-6, rtol=3e-6)


def _assert_rotation_matrix(actual: str, expected: torch.Tensor):
    actual_matrix = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(_vec(actual)))
    assert torch.allclose(actual_matrix, expected, atol=3e-6, rtol=3e-6)


def test_robomano_export_xml_writes_meshes_and_both_joint_modes(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )

    paths = layer.export_xml(tmp_path)

    assert paths["mesh_folder"] == tmp_path / "meshes"
    assert paths["reduced_xml"] == tmp_path / "right.xml"
    assert paths["ball_xml"] == tmp_path / "right_ball.xml"

    reduced = ET.parse(paths["reduced_xml"]).getroot()
    ball = ET.parse(paths["ball_xml"]).getroot()
    mesh_files = sorted(paths["mesh_folder"].glob("*.stl"))

    assert reduced.attrib["model"] == "mano_right"
    assert ball.attrib["model"] == "mano_right_ball"
    assert len(mesh_files) == 16
    assert all(_stl_triangle_count(path) > 0 for path in mesh_files)
    assert len(reduced.findall("./actuator/position")) == 20
    assert len(ball.findall("./actuator/position")) == 45
    assert len(reduced.findall("./worldbody//joint")) == 20
    assert len(ball.findall("./worldbody//joint")) == 15


def test_robomano_zero_beta_reduced_xml_uses_robowrapper_joint_frames(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )

    paths = layer.export_xml(tmp_path)
    root = ET.parse(paths["reduced_xml"]).getroot()

    index1y = root.find(".//body[@name='index1y']")
    index1x = root.find(".//body[@name='index1x']")
    index2 = root.find(".//body[@name='index2']")
    index3 = root.find(".//body[@name='index3']")
    thumb1y = root.find(".//body[@name='thumb1y']")
    thumb2 = root.find(".//body[@name='thumb2']")
    thumb3 = root.find(".//body[@name='thumb3']")

    assert torch.allclose(
        _vec(index1y.attrib["pos"]),
        torch.tensor([-0.0880972, -0.00520036, 0.020686]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_same_rotation(
        index1y.attrib["quat"],
        "0.0207461 -0.704984 -0.0206397 -0.708619",
    )
    assert "quat" not in index1x.attrib
    assert torch.allclose(
        _vec(index2.attrib["pos"]),
        torch.tensor([0.00238357, -0.00591432, -0.0323765]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index2.attrib
    assert torch.allclose(
        _vec(index3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0221942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index3.attrib
    _assert_same_rotation(
        index2.find("geom").attrib["quat"],
        "0.0207461 0.704984 0.0206397 0.708619",
    )

    _assert_same_rotation(
        thumb1y.attrib["quat"],
        "0.457776 -0.494578 -0.459441 -0.578574",
    )
    assert torch.allclose(
        _vec(thumb2.attrib["pos"]),
        torch.tensor([0.0252649, -0.0175957, 0.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_same_rotation(
        thumb2.attrib["quat"],
        "0.574412 -0.601175 0.038173 0.554241",
    )
    assert torch.allclose(
        _vec(thumb3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0270942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in thumb3.attrib
    _assert_same_rotation(
        thumb2.find("geom").attrib["quat"],
        "0.303832 0.79185 -0.375505 0.373706",
    )


def test_robomano_left_zero_beta_reduced_xml_uses_mirrored_joint_frames(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="left",
        betas=torch.zeros(10),
    )

    paths = layer.export_xml(tmp_path)
    root = ET.parse(paths["reduced_xml"]).getroot()

    index1y = root.find(".//body[@name='index1y']")
    index1x = root.find(".//body[@name='index1x']")
    index2 = root.find(".//body[@name='index2']")
    index3 = root.find(".//body[@name='index3']")
    thumb1y = root.find(".//body[@name='thumb1y']")
    thumb2 = root.find(".//body[@name='thumb2']")
    thumb3 = root.find(".//body[@name='thumb3']")

    index_frame = torch.from_numpy(mano_xml_frame_matrix("left", 1)).float()
    thumb1_frame = torch.from_numpy(mano_xml_frame_matrix("left", 13)).float()
    thumb2_frame = torch.from_numpy(mano_xml_frame_matrix("left", 14)).float()

    assert torch.allclose(
        _vec(index1y.attrib["pos"]),
        torch.tensor([0.0880972, -0.00520036, 0.020686]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_rotation_matrix(index1y.attrib["quat"], index_frame)
    assert "quat" not in index1x.attrib
    assert torch.allclose(
        _vec(index2.attrib["pos"]),
        torch.tensor([-0.00238357, -0.00591432, -0.0323765]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index2.attrib
    assert torch.allclose(
        _vec(index3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0221942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index3.attrib
    _assert_rotation_matrix(index2.find("geom").attrib["quat"], index_frame.T)

    _assert_rotation_matrix(thumb1y.attrib["quat"], thumb1_frame)
    assert torch.allclose(
        _vec(thumb2.attrib["pos"]),
        torch.tensor([-0.0252649, -0.0175957, 0.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_rotation_matrix(thumb2.attrib["quat"], thumb1_frame.T @ thumb2_frame)
    assert torch.allclose(
        _vec(thumb3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0270942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in thumb3.attrib
    _assert_rotation_matrix(thumb2.find("geom").attrib["quat"], thumb2_frame.T)
