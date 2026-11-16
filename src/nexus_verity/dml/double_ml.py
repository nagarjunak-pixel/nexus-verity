"""Chernozhukov et al. (Econometrica 2018) Cross-Fitted Double / Debiased Machine Learning (DML)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nexus_verity.types import DoubleMLReport


def _poly_features(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Nonlinear basis [1, x, x^2, sin(x_0), x_0*x_1] capturing nonlinear confounding."""
    n = x.shape[0]
    cols = [np.ones((n, 1), dtype=np.float64), x, x**2, np.sin(x[:, :1])]
    if x.shape[1] >= 2:
        cols.append(x[:, 0:1] * x[:, 1:2])
    return np.hstack(cols)


def _fit_ridge(
    phi: NDArray[np.float64], y: NDArray[np.float64], l2: float = 1e-2
) -> NDArray[np.float64]:
    d = phi.shape[1]
    reg = np.eye(d, dtype=np.float64) * l2
    reg[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + reg, phi.T @ y)


def run_double_ml_simpsons_benchmark(
    num_samples: int = 2000, num_folds: int = 5, true_ate: float = 2.50
) -> DoubleMLReport:
    """Prove K-Fold Cross-Fitted Neyman-Orthogonal DML reverses Simpson's Paradox confounding."""
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2 for out-of-fold cross-fitting")

    rng = np.random.default_rng(77)
    x = rng.normal(0.0, 1.0, size=(num_samples, 3))

    # Confounder X strongly increases propensity of Treatment T=1 while strongly DECREASING Outcome Y
    # (classic clinical/credit Simpson's Paradox: sicker patients get treated more -> naive ATE < 0!)
    g0 = 1.35 * x[:, 0] + 0.85 * (x[:, 1] ** 2 - 1.0) + 0.65 * np.sin(x[:, 0])
    prob_t = 1.0 / (1.0 + np.exp(-g0))
    t = (rng.random(num_samples) < prob_t).astype(np.float64)

    m0 = -3.40 * x[:, 0] - 1.95 * (x[:, 1] ** 2) - 1.20 * np.sin(x[:, 0]) + 0.50 * x[:, 2]
    y = true_ate * t + m0 + rng.normal(0.0, 0.85, size=num_samples)

    # 1. Naive Observational Difference-in-Means (suffers severe Simpson's sign reversal: negative!)
    naive_diff = float(np.mean(y[t == 1.0]) - np.mean(y[t == 0.0]))

    # 2. Chernozhukov K-Fold Cross-Fitted Double/Debiased ML (Neyman-Orthogonal Partialling-Out Score)
    phi = _poly_features(x)
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    folds = np.array_split(idx, num_folds)

    y_tilde = np.zeros(num_samples, dtype=np.float64)
    t_tilde = np.zeros(num_samples, dtype=np.float64)

    for k in range(num_folds):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(num_folds) if j != k])

        w_m = _fit_ridge(phi[train_idx], y[train_idx], l2=1e-2)
        w_g = _fit_ridge(phi[train_idx], t[train_idx], l2=1e-2)

        y_tilde[val_idx] = y[val_idx] - (phi[val_idx] @ w_m)
        t_tilde[val_idx] = t[val_idx] - (phi[val_idx] @ w_g)

    denom = float(np.mean(t_tilde * t_tilde))
    theta_dml = float(np.mean(t_tilde * y_tilde) / denom)

    # Neyman-orthogonal influence function psi_i = (t_tilde_i * (y_tilde_i - theta * t_tilde_i)) / E[t_tilde^2]
    psi = (t_tilde * (y_tilde - theta_dml * t_tilde)) / denom
    se_dml = float(np.std(psi, ddof=1) / math.sqrt(num_samples))
    ci_low = theta_dml - 1.95996 * se_dml
    ci_high = theta_dml + 1.95996 * se_dml

    naive_bias = abs(naive_diff - true_ate)
    dml_bias = max(1e-4, abs(theta_dml - true_ate))

    return DoubleMLReport(
        num_samples=num_samples,
        num_folds=num_folds,
        true_causal_ate=true_ate,
        naive_confounded_diff_in_means=round(naive_diff, 4),
        simpsons_sign_reversal_detected=(naive_diff < 0.0 < true_ate),
        dml_ate_estimate=round(theta_dml, 4),
        dml_robust_se=round(se_dml, 4),
        dml_ci95_low=round(ci_low, 4),
        dml_ci95_high=round(ci_high, 4),
        true_ate_covered_in_ci95=(ci_low <= true_ate <= ci_high),
        bias_reduction_x=round(naive_bias / dml_bias, 2),
    )
