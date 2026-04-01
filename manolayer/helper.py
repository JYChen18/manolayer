import pickle
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def _lrotmin(p):
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    else:
        p = np.asarray(p)

    if p.ndim == 1:
        p = p.reshape(-1, 3)
    elif p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape(-1, 3)

    rot_mats = R.from_rotvec(p[1:]).as_matrix()
    return (rot_mats - np.eye(3)).reshape(-1)


class ChumpyDummy:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __array__(self, dtype=None):
        # 1. Did chumpy leave slicing instructions? (e.g., chumpy.reordering.Select)
        if hasattr(self, "a") and hasattr(self, "idxs"):
            underlying_array = np.array(self.a, dtype=dtype)

            # Chumpy slicing operates on the FLATTENED array
            flat_sliced = underlying_array.flatten()[self.idxs]

            # Try to restore the intended shape
            if hasattr(self, "shape"):
                return flat_sliced.reshape(self.shape)
            elif hasattr(self, "_shape"):
                return flat_sliced.reshape(self._shape)
            else:
                # Fallback: keep the original dimensions but infer the sliced dimension
                # e.g., converts (778, 3, 20) -> (778, 3, -1) -> (778, 3, 10)
                new_shape = underlying_array.shape[:-1] + (-1,)
                return flat_sliced.reshape(new_shape)

        # 2. Is there a cached evaluated array?
        if hasattr(self, "r"):
            return np.array(self.r, dtype=dtype)
        if hasattr(self, "x"):
            return np.array(self.x, dtype=dtype)

        # 3. Fallback: just grab whatever raw data is inside
        return np.array(list(self.__dict__.values())[0], dtype=dtype)


class ChumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return ChumpyDummy
        return super().find_class(module, name)


# ADD THIS HELPER FUNCTION:
def _clean_chumpy_dict(data):
    """Recursively search for ChumpyDummy objects and convert them to numpy arrays."""
    if isinstance(data, dict):
        return {k: _clean_chumpy_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_clean_chumpy_dict(v) for v in data]
    elif isinstance(data, ChumpyDummy):
        # This triggers the __array__ method we wrote and returns a pure numpy array
        return np.array(data)
    else:
        return data


def ready_arguments(fname_or_dict, posekey4vposed="pose"):
    if not isinstance(fname_or_dict, dict):
        dd = ChumpyUnpickler(open(fname_or_dict, "rb"), encoding="latin1").load()
        dd = _clean_chumpy_dict(dd)
    else:
        dd = fname_or_dict

    want_shapemodel = "shapedirs" in dd
    nposeparms = dd["kintree_table"].shape[1] * 3

    if "trans" not in dd:
        dd["trans"] = np.zeros(3)
    if "pose" not in dd:
        dd["pose"] = np.zeros(nposeparms)
    if "shapedirs" in dd and "betas" not in dd:
        dd["betas"] = np.zeros(dd["shapedirs"].shape[-1])

    for s in [
        "v_template",
        "weights",
        "posedirs",
        "pose",
        "trans",
        "shapedirs",
        "betas",
        "J",
    ]:
        if (s in dd) and not hasattr(dd[s], "dterms"):
            dd[s] = np.array(dd[s])

    assert posekey4vposed in dd
    if want_shapemodel:
        dd["v_shaped"] = dd["shapedirs"].dot(dd["betas"]) + dd["v_template"]
        v_shaped = dd["v_shaped"]
        J_tmpx = dd["J_regressor"].dot(v_shaped[:, 0])
        J_tmpy = dd["J_regressor"].dot(v_shaped[:, 1])
        J_tmpz = dd["J_regressor"].dot(v_shaped[:, 2])
        dd["J"] = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        pose_map_res = _lrotmin(dd[posekey4vposed])
        dd["v_posed"] = v_shaped + dd["posedirs"].dot(pose_map_res)
    else:
        pose_map_res = _lrotmin(dd[posekey4vposed])
        dd_add = dd["posedirs"].dot(pose_map_res)
        dd["v_posed"] = dd["v_template"] + dd_add

    return dd
