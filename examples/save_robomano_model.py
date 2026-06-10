from hashlib import sha1
from pathlib import Path

import torch

from manolayer import RoboManoLayer


def beta_tag(betas: torch.Tensor) -> str:
    betas = torch.as_tensor(betas, dtype=torch.float32).reshape(-1).contiguous()
    digest = sha1(betas.cpu().numpy().tobytes()).hexdigest()[:10]
    return f"beta_{digest}"


def main():
    side = "right"
    betas = torch.zeros(10)

    save_folder = Path("exports") / "robomano" / side / beta_tag(betas)
    layer = RoboManoLayer(
        side=side,
        betas=betas,
    )
    paths = layer.export_xml(save_folder)

    print(f"Saved RoboMano model to {save_folder}")
    print(f"Reduced-joint XML: {paths['reduced_xml']}")
    print(f"Ball-joint XML: {paths['ball_xml']}")
    print(f"Meshes: {paths['mesh_folder']}")


if __name__ == "__main__":
    main()
