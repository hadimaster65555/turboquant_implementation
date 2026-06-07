import numpy as np

from turboquant.codebooks import precompute_codebook


def test_codebook_properties() -> None:
    codebook = precompute_codebook(64, 2, resolution=4097)
    assert np.all(np.diff(codebook.centroids) >= 0)
    assert codebook.boundaries[0] == -1.0
    assert codebook.boundaries[-1] == 1.0
    assert np.isclose(codebook.masses.sum(), 1.0, atol=1e-3)


def test_higher_bitwidth_reduces_cost() -> None:
    codebook_1 = precompute_codebook(128, 1, resolution=4097)
    codebook_3 = precompute_codebook(128, 3, resolution=4097)
    assert codebook_3.mse_cost < codebook_1.mse_cost
