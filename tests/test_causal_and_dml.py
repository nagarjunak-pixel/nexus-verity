"""Unit and Hypothesis property tests for Pearl Causal DAG, d-Separation, PC Discovery & Double ML."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nexus_verity.causal.dag_pc import CausalDAG, run_causal_discovery_and_backdoor_benchmark
from nexus_verity.dml.double_ml import run_double_ml_simpsons_benchmark


def test_causal_dag_validation_d_separation_and_collider_mbias() -> None:
    with pytest.raises(ValueError, match="Unknown node"):
        CausalDAG(["A", "B"], [("A", "C")])
    with pytest.raises(ValueError, match="Self-loop"):
        CausalDAG(["A", "B"], [("A", "A")])
    with pytest.raises(ValueError, match="cycle"):
        CausalDAG(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")])

    dag = CausalDAG(
        ["A", "B", "Z", "M", "C", "D"],
        [("Z", "A"), ("Z", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("Z", "M")],
    )
    assert len(dag.edges()) == 6
    assert "Z" in dag.ancestors_of({"A"})

    with pytest.raises(ValueError, match="distinct"):
        dag.is_d_separated("A", "A", set())

    assert dag.is_d_separated("A", "B", {"Z"}) is True
    assert dag.is_d_separated("A", "B", {"Z", "C"}) is False
    assert dag.is_d_separated("A", "B", {"Z", "D"}) is False

    bad_t, _ = dag.verify_backdoor_criterion("A", "B", {"A"})
    assert bad_t is False

    rep = run_causal_discovery_and_backdoor_benchmark(num_samples=2500)
    assert rep.skeleton_shd == 0
    assert rep.collider_bias_rejected is True
    assert len(rep.discovered_v_structures) >= 1


def test_double_ml_reverses_simpsons_paradox_and_covers_true_ate() -> None:
    with pytest.raises(ValueError, match="num_folds"):
        run_double_ml_simpsons_benchmark(num_samples=400, num_folds=1)

    rep = run_double_ml_simpsons_benchmark(num_samples=2000, num_folds=5, true_ate=2.50)
    assert rep.simpsons_sign_reversal_detected is True
    assert rep.naive_confounded_diff_in_means < 0.0
    assert abs(rep.dml_ate_estimate - 2.50) < 0.15
    assert rep.true_ate_covered_in_ci95 is True
    assert rep.bias_reduction_x > 15.0


@settings(max_examples=12, deadline=None)
@given(true_ate=st.floats(min_value=1.0, max_value=4.5, allow_nan=False, allow_infinity=False))
def test_hypothesis_dml_consistency_across_treatment_effects(true_ate: float) -> None:
    rep = run_double_ml_simpsons_benchmark(num_samples=900, num_folds=3, true_ate=true_ate)
    assert abs(rep.dml_ate_estimate - true_ate) < 0.25
