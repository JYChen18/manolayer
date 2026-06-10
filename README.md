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

# Place the models in ~/.manolayer/assets/mano/models/MANO_RIGHT.pkl
mano_layer = ManoLayer(side="right")
```

# Staged API
The original implementation is still available as `LegacyManoLayer`. The public
`ManoLayer` now exposes the MANO forward pass as reusable stages:

```python
from manolayer import ManoLayer

mano_layer = ManoLayer(
    side="right",
    rot_mode="axisang",
    center_idx=0,
)

qpos = mano_layer.pose_to_qpos(pose_coeffs)
link_poses = mano_layer.forward_kinematics(qpos)
output = mano_layer.get_verts_joints(link_poses)
```

# RoboMano export
`RoboManoLayer` exports a beta-specific MuJoCo model:

```python
from pathlib import Path

import torch

from manolayer import RoboManoLayer

betas = torch.zeros(10)
beta_name = "neutral"
robo_layer = RoboManoLayer(
    side="right",
    betas=betas,
)
paths = robo_layer.export_xml(Path("exports") / "robomano" / "right" / beta_name)
```

See `examples/save_robomano_model.py` for a simple beta-hashed save path.

The staged `qpos` has shape `(B, 67)`: root position `(3)`, root quaternion
`(4)`, then 15 joint quaternions. Pose options such as `rot_mode`,
`center_idx`, `use_pca`, `flat_hand_mean`, and `ncomps` are fixed when the
layer is created. `center_idx` only affects the root translation baked into
`pose_to_qpos`; `forward_kinematics` and `get_verts_joints` do not read
`center_idx`. The staged API supports `center_idx=None` and `center_idx=0`.
The final `ManoOutput` only contains `verts` and `joints`.

If incoming link poses use robot or simulator link frames instead of MANO link
frames, configure their zero-pose link frames once:

```python
mano_layer = ManoLayer(
    side="right",
    center_idx=0,
    bind_poses=bind_poses,
).to(device=robot_link_poses.device, dtype=robot_link_poses.dtype)
output = mano_layer.get_verts_joints(robot_link_poses)
```

# Acknowledgement
- [mano](https://mano.is.tue.mpg.de/)
- [manotorch](https://github.com/lixiny/manotorch)
- [manopth](https://github.com/hassony2/manopth)
- [smplx](https://github.com/vchoutas/smplx/tree/main)
