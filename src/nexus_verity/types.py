"""Strongly-typed Pydantic models for Causal Discovery, Cross-Fitted DML, Split Conformal & Doubly-Robust OPE."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AdjustmentSetCheck(BaseModel):
    candidate_set: list[str]
    valid_backdoor: bool
    reason: str


class CausalDiscoveryReport(BaseModel):
    num_samples: int
    variables: list[str]
    true_edges: list[tuple[str, str]]
    discovered_skeleton_edges: list[tuple[str, str]]
    discovered_v_structures: list[str]
    skeleton_shd: int
    collider_bias_rejected: bool
    minimal_valid_backdoor_set: list[str]
    candidate_checks: list[AdjustmentSetCheck]


class DoubleMLReport(BaseModel):
    num_samples: int
    num_folds: int
    true_causal_ate: float
    naive_confounded_diff_in_means: float
    simpsons_sign_reversal_detected: bool
    dml_ate_estimate: float
    dml_robust_se: float
    dml_ci95_low: float
    dml_ci95_high: float
    true_ate_covered_in_ci95: bool
    bias_reduction_x: float


class ConformalAndOPEReport(BaseModel):
    backend: str
    alpha: float
    target_coverage_pct: float
    empirical_conformal_coverage_pct: float
    naive_gaussian_coverage_pct: float
    conformal_quantile_qhat: float
    true_counterfactual_policy_value: float
    ips_estimate: float
    snips_estimate: float
    dr_estimate: float
    dr_stderr: float
    dr_vs_ips_error_reduction_x: float
    cpp_vs_numpy_max_delta: float


class CausalIncidentOutcome(BaseModel):
    incident_id: str
    title: str
    expected_safe_value: int
    raw_llm_value: int | None
    raw_llm_safe: bool
    verified_value: int | None
    verified_safe: bool
    turns_used: int
    latency_ms: float


class RealCausalAuditReport(BaseModel):
    model_name: str
    num_incidents: int
    raw_safe_count: int
    raw_safe_pct: float
    verified_safe_count: int
    verified_safe_pct: float
    safety_lift_pct: float
    zero_confounded_policy_leakage: bool = Field(default=True)
    outcomes: list[CausalIncidentOutcome]
