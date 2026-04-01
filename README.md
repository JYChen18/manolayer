# ManoLayer without chumpy

Why another manolayer? 
- I failed to install chumpy with uv, which is required by both [manotorch](https://github.com/lixiny/manotorch) and [manopth](https://github.com/hassony2/manopth).
- [smplx](https://github.com/vchoutas/smplx/tree/main) has a different input format, which results in a ghost translation.

# Prepare Mano Models
1. Download mano models from [here](https://mano.is.tue.mpg.de/)
and place them as 
```
assets/mano/models
|- MANO_LEFT.pkl
|_ MANO_RIGHT.pkl
```

2. Remove chumpy in the mano models using another environment with chumpy
```bash
conda create -n tmp python=3.10 
conda activate tmp
pip install chumpy==0.70 numpy==1.23.0
python clean_ch.py
```

# Installation 
```bash
uv add https://github.com/JYChen18/manolayer.git
```


# Acknowledgement
- [mano](https://mano.is.tue.mpg.de/)
- [manotorch](https://github.com/lixiny/manotorch)
- [manopth](https://github.com/hassony2/manopth)
- [smplx](https://github.com/vchoutas/smplx/tree/main)