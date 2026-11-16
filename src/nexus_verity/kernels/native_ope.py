"""Native C++20 bridge (`clang++ -O3`) for Doubly-Robust Off-Policy Evaluation & Split Conformal Quantiles."""

from __future__ import annotations

import ctypes
import hashlib
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nexus_verity.types import ConformalAndOPEReport


def _compile_native_lib() -> tuple[ctypes.CDLL | None, str]:
    src_path = Path(__file__).resolve().parents[3] / "csrc" / "nexus_kernels.cpp"
    clang = shutil.which("clang++") or shutil.which("g++")
    if not src_path.exists() or clang is None:
        return None, "python-numpy-fallback"

    digest = hashlib.sha256(src_path.read_bytes()).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "nexus_verity_native_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".dylib" if Path("/System/Library").exists() else ".so"
    lib_path = out_dir / f"libnexus_kernels_{digest}{ext}"

    if not lib_path.exists():
        cmd = [clang, "-O3", "-std=c++20", "-shared", "-fPIC", str(src_path), "-o", str(lib_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None, "python-numpy-fallback"

    lib = ctypes.CDLL(str(lib_path))
    lib.nexus_doubly_robust_ope_f64.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.nexus_doubly_robust_ope_f64.restype = None

    lib.nexus_conformal_quantile_f64.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_double,
    ]
    lib.nexus_conformal_quantile_f64.restype = ctypes.c_double
    return lib, "c++20-native (clang++ -O3)"


_NATIVE_LIB, _BACKEND_NAME = _compile_native_lib()


def active_backend_name() -> str:
    return _BACKEND_NAME


def numpy_doubly_robust_ope(
    logged_actions: NDArray[np.int32],
    target_actions: NDArray[np.int32],
    logged_propensities: NDArray[np.float64],
    observed_rewards: NDArray[np.float64],
    qhat_logged: NDArray[np.float64],
    qhat_target: NDArray[np.float64],
    clip_min_propensity: float = 1e-3,
) -> tuple[float, float, float, float]:
    n = int(logged_actions.shape[0])
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    eps = max(1e-6, clip_min_propensity)
    p = np.maximum(eps, logged_propensities)
    match = (logged_actions == target_actions).astype(np.float64)
    w = match / p
    ips_terms = w * observed_rewards
    dr_terms = qhat_target + w * (observed_rewards - qhat_logged)

    mean_dr = float(np.mean(dr_terms))
    mean_ips = float(np.mean(ips_terms))
    w_sum = float(np.sum(w))
    snips = float(np.sum(ips_terms) / w_sum) if w_sum > 1e-12 else 0.0
    se_dr = float(np.std(dr_terms, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean_dr, se_dr, mean_ips, snips


def doubly_robust_ope(
    logged_actions: NDArray[np.int32],
    target_actions: NDArray[np.int32],
    logged_propensities: NDArray[np.float64],
    observed_rewards: NDArray[np.float64],
    qhat_logged: NDArray[np.float64],
    qhat_target: NDArray[np.float64],
    *,
    clip_min_propensity: float = 1e-3,
    force_python: bool = False,
) -> tuple[float, float, float, float]:
    la = np.ascontiguousarray(logged_actions, dtype=np.int32)
    ta = np.ascontiguousarray(target_actions, dtype=np.int32)
    lp = np.ascontiguousarray(logged_propensities, dtype=np.float64)
    rw = np.ascontiguousarray(observed_rewards, dtype=np.float64)
    ql = np.ascontiguousarray(qhat_logged, dtype=np.float64)
    qt = np.ascontiguousarray(qhat_target, dtype=np.float64)
    n = int(la.shape[0])
    if not (ta.shape == lp.shape == rw.shape == ql.shape == qt.shape == (n,)):
        raise ValueError("All OPE input vectors must have identical 1D length")

    if _NATIVE_LIB is None or force_python:
        return numpy_doubly_robust_ope(la, ta, lp, rw, ql, qt, clip_min_propensity)

    c_dr = ctypes.c_double(0.0)
    c_se = ctypes.c_double(0.0)
    c_ips = ctypes.c_double(0.0)
    c_snips = ctypes.c_double(0.0)
    _NATIVE_LIB.nexus_doubly_robust_ope_f64(
        la.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ql.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        clip_min_propensity,
        ctypes.byref(c_dr),
        ctypes.byref(c_se),
        ctypes.byref(c_ips),
        ctypes.byref(c_snips),
    )
    return float(c_dr.value), float(c_se.value), float(c_ips.value), float(c_snips.value)


def split_conformal_quantile(
    scores: NDArray[np.float64], alpha: float = 0.10, *, force_python: bool = False
) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    arr = np.ascontiguousarray(scores, dtype=np.float64)
    n = int(arr.shape[0])
    if n == 0:
        return 0.0
    if _NATIVE_LIB is not None and not force_python:
        return float(
            _NATIVE_LIB.nexus_conformal_quantile_f64(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n, alpha
            )
        )
    s_sorted = np.sort(arr)
    k = max(1, min(n, int(math.ceil((n + 1) * (1.0 - alpha)))))
    return float(s_sorted[k - 1])


def run_conformal_and_dr_policy_benchmark(alpha: float = 0.10) -> ConformalAndOPEReport:
    """Benchmark Finite-Sample Split Conformal Coverage vs. Overfit Plug-In Gaussian + C++20 DR OPE."""
    rng = np.random.default_rng(108)
    n_train, n_cal, n_test, d_feat = 140, 600, 1200, 65
    x_all = rng.normal(0.0, 1.0, size=(n_train + n_cal + n_test, d_feat))

    # Sparse true weights + skewed heavy-tailed shock: high-d OLS overfits train residuals!
    w_true = np.zeros(d_feat, dtype=np.float64)
    w_true[:5] = np.array([2.2, -1.6, 1.3, -0.9, 1.1])
    shock = rng.exponential(scale=1.25, size=x_all.shape[0]) - 1.25
    y_all = (x_all @ w_true) + shock

    phi_all = np.column_stack([np.ones(x_all.shape[0]), x_all])
    phi_tr, y_tr = phi_all[:n_train], y_all[:n_train]
    beta, _, _, _ = np.linalg.lstsq(phi_tr, y_tr, rcond=None)

    # Naive plug-in Gaussian 90% interval from in-sample training residuals severely under-covers!
    resid_tr = y_tr - (phi_tr @ beta)
    sigma_hat = float(np.std(resid_tr, ddof=1))
    gauss_half = 1.6448536 * sigma_hat

    # Split Conformal calibration on held-out exchangeable calibration fold
    phi_cal, y_cal = phi_all[n_train : n_train + n_cal], y_all[n_train : n_train + n_cal]
    cal_scores = np.abs(y_cal - (phi_cal @ beta))
    q_cpp = split_conformal_quantile(cal_scores, alpha=alpha)
    q_py = split_conformal_quantile(cal_scores, alpha=alpha, force_python=True)

    phi_te, y_te = phi_all[n_train + n_cal :], y_all[n_train + n_cal :]
    pred_te = phi_te @ beta
    conf_cov = float(np.mean(np.abs(y_te - pred_te) <= q_cpp) * 100.0)
    gauss_cov = float(np.mean(np.abs(y_te - pred_te) <= gauss_half) * 100.0)

    # Off-Policy Counterfactual Bandit Evaluation (skewed propensity logging policy vs optimal target policy)
    n_ope = 1200
    ctx = rng.standard_normal((n_ope, 2))
    # True expected reward for action a in {0, 1}:
    r0_mean = 2.0 + 0.8 * ctx[:, 0] - 0.5 * ctx[:, 1]
    r1_mean = 2.0 + 2.1 * ctx[:, 0] + 0.9 * ctx[:, 1]

    # Target policy picks action 1 iff r1_mean > r0_mean
    target_act = (r1_mean > r0_mean).astype(np.int32)
    true_target_value = float(np.mean(np.where(target_act == 1, r1_mean, r0_mean)))

    # Logging policy was conservative (rarely explored action 1 when ctx[:,0] > 0 -> small propensities!)
    logits_log = -2.1 - 0.9 * ctx[:, 0] + 0.3 * ctx[:, 1]
    prob_a1 = np.clip(1.0 / (1.0 + np.exp(-logits_log)), 0.02, 0.98)
    logged_act = (rng.random(n_ope) < prob_a1).astype(np.int32)
    logged_prop = np.where(logged_act == 1, prob_a1, 1.0 - prob_a1)
    obs_reward = np.where(logged_act == 1, r1_mean, r0_mean) + rng.normal(0.0, 0.6, size=n_ope)

    # Reward model q_hat (slightly regularized)
    qhat_a0 = r0_mean + rng.normal(0.0, 0.08, size=n_ope)
    qhat_a1 = r1_mean + rng.normal(0.0, 0.08, size=n_ope)
    qhat_logged = np.where(logged_act == 1, qhat_a1, qhat_a0)
    qhat_target = np.where(target_act == 1, qhat_a1, qhat_a0)

    dr_v, dr_se, ips_v, snips_v = doubly_robust_ope(
        logged_act, target_act, logged_prop, obs_reward, qhat_logged, qhat_target
    )
    dr_py, se_py, ips_py, snips_py = numpy_doubly_robust_ope(
        logged_act, target_act, logged_prop, obs_reward, qhat_logged, qhat_target
    )
    max_delta = max(
        abs(q_cpp - q_py),
        abs(dr_v - dr_py),
        abs(dr_se - se_py),
        abs(ips_v - ips_py),
        abs(snips_v - snips_py),
    )

    ips_err = abs(ips_v - true_target_value)
    dr_err = max(1e-6, abs(dr_v - true_target_value))

    return ConformalAndOPEReport(
        backend=active_backend_name(),
        alpha=alpha,
        target_coverage_pct=round((1.0 - alpha) * 100.0, 2),
        empirical_conformal_coverage_pct=round(conf_cov, 2),
        naive_gaussian_coverage_pct=round(gauss_cov, 2),
        conformal_quantile_qhat=round(q_cpp, 4),
        true_counterfactual_policy_value=round(true_target_value, 4),
        ips_estimate=round(ips_v, 4),
        snips_estimate=round(snips_v, 4),
        dr_estimate=round(dr_v, 4),
        dr_stderr=round(dr_se, 4),
        dr_vs_ips_error_reduction_x=round(ips_err / dr_err, 2),
        cpp_vs_numpy_max_delta=max_delta,
    )
