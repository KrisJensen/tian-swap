import numpy as np

def var_explained(true, pred):
    """trials by cells"""
    res = true - pred
    var_true, var_res = np.var(true, axis = 0), np.var(res, axis = 0)

    return 1 - var_res/var_true

