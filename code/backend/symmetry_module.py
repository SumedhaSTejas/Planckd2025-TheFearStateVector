import numpy as np

def reduce_hilbert_space(H_P, H_M, symmetry_fn):
    mask = symmetry_fn(np.arange(H_P.shape[0]))
    return H_P[np.ix_(mask, mask)], H_M[np.ix_(mask, mask)]
