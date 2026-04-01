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

# Acknowledgement
- [mano](https://mano.is.tue.mpg.de/)
- [manotorch](https://github.com/lixiny/manotorch)
- [manopth](https://github.com/hassony2/manopth)
- [smplx](https://github.com/vchoutas/smplx/tree/main)