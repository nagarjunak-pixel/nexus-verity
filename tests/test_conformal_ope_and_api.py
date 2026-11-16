"""Unit & Hypothesis tests for C++20 Doubly-Robust OPE, Split Conformal Coverage, CEGIS & FastAPI/CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from nexus_verity.api.app import create_app
from nexus_verity.audit.ollama_causal import (
    _check_collider_vars,
    _check_conformal_rank,
    _check_dr_single_obs,
    _check_ipw_ate,
    _check_simpsons_ate,
    _check_wald_late,
    build_causal_incident_suite,
    run_real_causal_ollama_audit,
)
from nexus_verity.cli import app as cli_app
from nexus_verity.kernels.native_ope import (
    active_backend_name,
    doubly_robust_ope,
    numpy_doubly_robust_ope,
    run_conformal_and_dr_policy_benchmark,
    split_conformal_quantile,
)


@settings(max_examples=25, deadline=None)
@given(
    n=st.integers(min_value=10, max_value=120),
    alpha=st.floats(min_value=0.05, max_value=0.45, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_cpp20_conformal_and_dr_ope_exact_parity(n: int, alpha: float) -> None:
    rng = np.random.default_rng(n)
    scores = np.abs(rng.standard_normal(n))
    q_cpp = split_conformal_quantile(scores, alpha=alpha)
    q_py = split_conformal_quantile(scores, alpha=alpha, force_python=True)
    assert abs(q_cpp - q_py) < 1e-12
    # Empirical in-sample rank property: at least (1 - alpha) fraction of cal scores <= q_cpp
    assert float(np.mean(scores <= q_cpp)) >= (1.0 - alpha) - 1e-12

    la = rng.integers(0, 2, size=n, dtype=np.int32)
    ta = rng.integers(0, 2, size=n, dtype=np.int32)
    lp = rng.uniform(0.05, 0.95, size=n)
    rw = rng.normal(1.0, 2.0, size=n)
    ql = rng.normal(1.0, 1.5, size=n)
    qt = rng.normal(1.2, 1.5, size=n)
    res_cpp = doubly_robust_ope(la, ta, lp, rw, ql, qt)
    res_py = doubly_robust_ope(la, ta, lp, rw, ql, qt, force_python=True)
    for x_c, x_p in zip(res_cpp, res_py, strict=True):
        assert abs(x_c - x_p) < 1e-11


def test_ope_and_conformal_edge_cases_and_benchmark() -> None:
    empty_i = np.zeros(0, dtype=np.int32)
    empty_f = np.zeros(0, dtype=np.float64)
    assert doubly_robust_ope(empty_i, empty_i, empty_f, empty_f, empty_f, empty_f) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert numpy_doubly_robust_ope(empty_i, empty_i, empty_f, empty_f, empty_f, empty_f) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert split_conformal_quantile(empty_f, alpha=0.1) == 0.0
    with pytest.raises(ValueError, match="alpha"):
        split_conformal_quantile(np.ones(5), alpha=1.2)
    with pytest.raises(ValueError, match="identical"):
        doubly_robust_ope(
            np.ones(3, dtype=np.int32),
            np.ones(2, dtype=np.int32),
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        )

    rep = run_conformal_and_dr_policy_benchmark(alpha=0.10)
    assert rep.empirical_conformal_coverage_pct >= 88.0
    assert rep.empirical_conformal_coverage_pct > rep.naive_gaussian_coverage_pct
    assert rep.cpp_vs_numpy_max_delta < 1e-11
    assert "c++20" in active_backend_name() or "python" in active_backend_name()


def test_causal_validators_and_cegis_flow(tmp_path: Path) -> None:
    suite = build_causal_incident_suite()
    assert len(suite) == 6
    for sp in suite:
        assert sp.validator_fn(sp.expected_value) is None

    assert _check_simpsons_ate(-7) is not None
    assert _check_simpsons_ate(12) is not None
    assert _check_conformal_rank(178, n_cal=198, alpha=0.10) is not None
    assert _check_conformal_rank(185, n_cal=198, alpha=0.10) is not None
    assert _check_wald_late(18) is not None
    assert _check_wald_late(45) is not None
    assert _check_collider_vars(2) is not None
    assert _check_dr_single_obs(56) is not None
    assert _check_dr_single_obs(10) is not None
    assert _check_ipw_ate(12) is not None

    fake_art = tmp_path / "causal_audit.json"
    seq = [-7, 15, 180, 60, 1, 26, 16]
    idx = {"k": 0}

    class DummyResp:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(url: str, **kwargs: object) -> DummyResp:
        v = seq[idx["k"]]
        idx["k"] += 1
        return DummyResp(200, {"response": json.dumps({"steps": ["eq"], "value": v})})

    with (
        patch("httpx.get", return_value=DummyResp(200, {"models": []})),
        patch("httpx.post", side_effect=fake_post),
    ):
        rep = run_real_causal_ollama_audit(save_path=fake_art)
        assert rep.raw_safe_count == 5
        assert rep.verified_safe_count == 6

    with patch("httpx.get", side_effect=httpx.ConnectError("offline")):
        replayed = run_real_causal_ollama_audit(save_path=fake_art, allow_offline_replay=True)
        assert replayed.verified_safe_count == 6
        with pytest.raises(RuntimeError, match="unreachable"):
            run_real_causal_ollama_audit(
                save_path=tmp_path / "missing.json", allow_offline_replay=False
            )


def test_fastapi_endpoints_and_cli_commands() -> None:
    app = create_app()
    client = TestClient(app)

    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/v1/causal/discover").status_code == 200
    assert (
        client.post("/api/v1/dml/estimate", json={"num_samples": 1000, "num_folds": 4}).status_code
        == 200
    )
    assert client.get("/api/v1/conformal/evaluate").status_code == 200
    assert client.get("/api/v1/audit/ollama").status_code == 200

    runner = CliRunner()
    assert runner.invoke(cli_app, ["discover-dag"]).exit_code == 0
    assert runner.invoke(cli_app, ["estimate-dml"]).exit_code == 0
    assert runner.invoke(cli_app, ["evaluate-policy"]).exit_code == 0
    assert runner.invoke(cli_app, ["eval-ollama"]).exit_code == 0
