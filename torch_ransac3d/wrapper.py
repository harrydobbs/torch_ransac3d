import numpy as np
import torch


def numpy_to_torch(func):
    def wrapper(pts, *args, **kwargs):
        is_numpy = isinstance(pts, np.ndarray)
        if is_numpy:
            pts = torch.from_numpy(pts.astype(np.float32))

        result = func(pts, *args, **kwargs)

        if is_numpy:
            if isinstance(result, tuple):
                return tuple(
                    r.numpy() if isinstance(r, torch.Tensor) else r for r in result
                )
            return result.numpy() if isinstance(result, torch.Tensor) else result
        return result

    return wrapper
