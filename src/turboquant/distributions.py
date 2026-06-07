from __future__ import annotations

import math

import numpy as np


def sphere_coordinate_pdf(dim: int, x: np.ndarray) -> np.ndarray:
    if dim < 2:
        raise ValueError("dim must be at least 2")
    log_coeff = math.lgamma(dim / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1) / 2.0)
    coeff = math.exp(log_coeff)
    values = np.clip(1.0 - np.square(x), 0.0, None) ** ((dim - 3) / 2.0)
    return coeff * values


def integration_grid(dim: int, resolution: int = 32769) -> tuple[np.ndarray, np.ndarray]:
    if resolution % 2 == 0:
        resolution += 1
    grid = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    pdf = sphere_coordinate_pdf(dim, grid)
    dx = grid[1] - grid[0]
    mass = np.trapezoid(pdf, dx=dx)
    pdf = pdf / mass
    return grid, pdf


def weighted_interval_stats(
    grid: np.ndarray,
    pdf: np.ndarray,
    left: float,
    right: float,
) -> tuple[float, float]:
    mask = (grid >= left) & (grid <= right)
    subgrid = grid[mask]
    subpdf = pdf[mask]
    if subgrid.size == 0:
        midpoint = (left + right) / 2.0
        return midpoint, 0.0
    mass = np.trapezoid(subpdf, subgrid)
    if mass <= 0:
        midpoint = (left + right) / 2.0
        return midpoint, 0.0
    mean = np.trapezoid(subgrid * subpdf, subgrid) / mass
    return float(mean), float(mass)
