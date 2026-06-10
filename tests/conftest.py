import os
from pathlib import Path

import pytest


@pytest.fixture
def mano_assets_root():
    candidates = [
        os.environ.get("MANO_ASSETS_ROOT"),
        "~/.manolayer/assets/mano/models",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if (root / "MANO_RIGHT.pkl").is_file():
            return str(root)
    pytest.skip("MANO_RIGHT.pkl is not available locally.")
