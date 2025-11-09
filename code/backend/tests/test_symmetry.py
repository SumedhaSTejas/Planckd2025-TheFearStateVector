import numpy as np
from symmetry_module import reduce_hilbert_space

def test_symmetry_reduction_identity():
    H_P = np.diag(np.arange(8))
    H_reduced, _ = reduce_hilbert_space(H_P, H_P, lambda idx: idx % 2 == 0)
    assert H_reduced.shape[0] < H_P.shape[0]
