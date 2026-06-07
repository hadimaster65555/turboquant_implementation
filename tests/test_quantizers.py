import numpy as np

from turboquant.quantizers import TurboQuantMSE, TurboQuantProd


def random_sphere(samples: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((samples, dim))
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data


def test_mse_improves_with_bitwidth() -> None:
    vectors = random_sphere(128, 64, 1)
    q1 = TurboQuantMSE(64, 1, 11, codebook_mode="compute")
    q3 = TurboQuantMSE(64, 3, 11, codebook_mode="compute")
    recon1 = q1.dequantize(q1.quantize(vectors))
    recon3 = q3.dequantize(q3.quantize(vectors))
    mse1 = np.mean(np.sum((vectors - recon1) ** 2, axis=1))
    mse3 = np.mean(np.sum((vectors - recon3) ** 2, axis=1))
    assert mse3 < mse1


def test_prod_is_nearly_unbiased_empirically() -> None:
    x = random_sphere(512, 32, 2)
    y = random_sphere(512, 32, 3)
    quantizer = TurboQuantProd(32, 3, 17, codebook_mode="compute")
    codes = quantizer.quantize(x)
    decoded = quantizer.dequantize(codes)
    bias = float(np.mean(np.sum(y * decoded, axis=1) - np.sum(y * x, axis=1)))
    assert abs(bias) < 0.08
