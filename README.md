# ManoLayer without chumpy

Why another manolayer? 
- I failed to install chumpy with uv, which is required by both [manotorch](https://github.com/lixiny/manotorch) and [manopth](https://github.com/hassony2/manopth).
- [smplx](https://github.com/vchoutas/smplx/tree/main) has a different input format, which results in a ghost translation.


# Installation 
```bash
uv add https://github.com/JYChen18/manolayer.git
```

# Example
Download mano models from [here](https://mano.is.tue.mpg.de/)
```python
from manolayer import ManoLayer

# If you place the models in assets/mano/models/MANO_RIGHT.pkl
mano_layer = ManoLayer(mano_assets_root="assets/mano/models", side="right")
```

# Staged API
The original implementation is still available as `LegacyManoLayer`. The public
`ManoLayer` now exposes the MANO forward pass as reusable stages:

```python
from manolayer import ManoLayer

mano_layer = ManoLayer(mano_assets_root="assets/mano/models", side="right")

mano_layer.update_beta(betas)
qpos = mano_layer.pose_to_qpos(
    pose_coeffs,
    rot_mode="axisang",
    center_idx=0,
)
link_poses = mano_layer.forward_kinematics(qpos)
output = mano_layer.get_verts_joints(link_poses)
```

The staged `qpos` has shape `(B, 67)`: root position `(3)`, root quaternion
`(4)`, then 15 joint quaternions. `center_idx` is only accepted by
`pose_to_qpos`; the staged API supports `center_idx=None` and `center_idx=0`.
The final `ManoOutput` only contains `verts` and `joints`.

# Acknowledgement
- [mano](https://mano.is.tue.mpg.de/)
- [manotorch](https://github.com/lixiny/manotorch)
- [manopth](https://github.com/hassony2/manopth)
- [smplx](https://github.com/vchoutas/smplx/tree/main)
