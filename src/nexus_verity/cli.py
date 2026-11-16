"""Rich CLI for NexusVerity: Causal Discovery, Cross-Fitted DML, Split Conformal/DR & Live Qwen CEGIS."""

from __future__ import annotations

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from nexus_verity.api.app import create_app
from nexus_verity.audit.ollama_causal import run_real_causal_ollama_audit
from nexus_verity.causal.dag_pc import run_causal_discovery_and_backdoor_benchmark
from nexus_verity.dml.double_ml import run_double_ml_simpsons_benchmark
from nexus_verity.kernels.native_ope import run_conformal_and_dr_policy_benchmark

app = typer.Typer(
    name="nexus",
    help="NexusVerity — Pearl Causal DAG/PC, Chernozhukov DML, Split Conformal & C++20 Doubly-Robust OPE.",
    no_args_is_help=True,
)
console = Console()


@app.command("discover-dag")
def cmd_discover_dag(num_samples: int = typer.Option(2500, "--samples")) -> None:
    """Run Spirtes-Glymour-Scheines PC Skeleton/v-Structure Discovery & Pearl Backdoor Criterion."""
    rep = run_causal_discovery_and_backdoor_benchmark(num_samples=num_samples)
    tbl = Table(title=f"Pearl Backdoor & PC Structural Causal Discovery (N={rep.num_samples})")
    tbl.add_column("Metric / Candidate Set", style="cyan")
    tbl.add_column("Verified Value", justify="right", style="green")
    tbl.add_row("Skeleton Structural Hamming Distance (SHD)", str(rep.skeleton_shd))
    tbl.add_row("Discovered v-Structures", ", ".join(rep.discovered_v_structures) or "none")
    tbl.add_row("Post-Treatment Collider Bias Rejected", str(rep.collider_bias_rejected))
    tbl.add_row("Minimal Valid Backdoor Set", "{" + ", ".join(rep.minimal_valid_backdoor_set) + "}")
    for chk in rep.candidate_checks:
        label = "{" + ", ".join(chk.candidate_set) + "}"
        tbl.add_row(f"Backdoor({label})", "VALID ✓" if chk.valid_backdoor else "REJECTED ✗")
    console.print(tbl)


@app.command("estimate-dml")
def cmd_estimate_dml(
    num_samples: int = typer.Option(2000, "--samples"),
    num_folds: int = typer.Option(5, "--folds"),
) -> None:
    """Run Chernozhukov K-Fold Cross-Fitted Neyman-Orthogonal Double/Debiased ML vs. Simpson's Bias."""
    rep = run_double_ml_simpsons_benchmark(num_samples=num_samples, num_folds=num_folds)
    tbl = Table(
        title=f"Chernozhukov {rep.num_folds}-Fold Cross-Fitted Double ML (N={rep.num_samples})"
    )
    tbl.add_column("Estimator / Invariant", style="cyan")
    tbl.add_column("Verified Value", justify="right", style="green")
    tbl.add_row("True Ground-Truth Causal ATE (theta_0)", f"+{rep.true_causal_ate:.4f}")
    tbl.add_row(
        "Naive Confounded Diff-in-Means (Simpson's Reversal!)",
        f"{rep.naive_confounded_diff_in_means:+.4f}",
    )
    tbl.add_row(
        "Cross-Fitted Neyman-Orthogonal DML ATE",
        f"{rep.dml_ate_estimate:+.4f} ± {rep.dml_robust_se:.4f}",
    )
    tbl.add_row(
        "95% Robust Confidence Interval", f"[{rep.dml_ci95_low:.4f}, {rep.dml_ci95_high:.4f}]"
    )
    tbl.add_row("Causal Bias Reduction vs. Naive", f"{rep.bias_reduction_x:.2f}x cleaner")
    console.print(tbl)


@app.command("evaluate-policy")
def cmd_evaluate_policy(alpha: float = typer.Option(0.10, "--alpha")) -> None:
    """Run C++20 Finite-Sample Split Conformal Prediction & Doubly-Robust Off-Policy Evaluator."""
    rep = run_conformal_and_dr_policy_benchmark(alpha=alpha)
    tbl = Table(title=f"C++20 Split Conformal & Doubly-Robust Off-Policy Engine ({rep.backend})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Verified Value", justify="right", style="green")
    tbl.add_row(
        "Target (1-alpha) vs. Conformal vs. Naive Gaussian",
        f"{rep.target_coverage_pct:.1f}% vs. {rep.empirical_conformal_coverage_pct:.2f}% vs. {rep.naive_gaussian_coverage_pct:.2f}%",
    )
    tbl.add_row(
        "True Counterfactual Policy Value V(pi)", f"{rep.true_counterfactual_policy_value:.4f}"
    )
    tbl.add_row(
        "Raw IPS vs. SNIPS vs. C++20 Doubly-Robust (DR)",
        f"{rep.ips_estimate:.4f} / {rep.snips_estimate:.4f} / {rep.dr_estimate:.4f} (±{rep.dr_stderr:.4f})",
    )
    tbl.add_row("DR Error Reduction vs. Raw IPS", f"{rep.dr_vs_ips_error_reduction_x:.2f}x")
    tbl.add_row("C++20 vs. NumPy Max |Δ|", f"{rep.cpp_vs_numpy_max_delta:.2e}")
    console.print(tbl)


@app.command("eval-ollama")
def cmd_eval_ollama(
    model: str = typer.Option("qwen3.5:4b", "--model"),
    live_only: bool = typer.Option(False, "--live-only"),
) -> None:
    """Run Zero-Mock Live Ollama Causal Backdoor, Simpson's Paradox & Conformal CEGIS Gate."""
    rep = run_real_causal_ollama_audit(model=model, allow_offline_replay=not live_only)
    tbl = Table(title=f"Zero-Mock Causal & Conformal CEGIS ({rep.model_name})")
    tbl.add_column("Incident ID", style="cyan")
    tbl.add_column("Safe Spec", justify="right")
    tbl.add_column("Raw Pass@1", justify="right")
    tbl.add_column("Nexus Verified", justify="right")
    for o in rep.outcomes:
        r_str = (
            f"[green]{o.raw_llm_value} ✓[/]" if o.raw_llm_safe else f"[red]{o.raw_llm_value} ✗[/]"
        )
        v_str = (
            f"[green]{o.verified_value} ✓ (t={o.turns_used})[/]"
            if o.verified_safe
            else "[yellow]BLOCKED[/]"
        )
        tbl.add_row(o.incident_id, str(o.expected_safe_value), r_str, v_str)
    console.print(tbl)
    console.print(
        f"Summary: Raw Pass@1 = {rep.raw_safe_count}/{rep.num_incidents} ({rep.raw_safe_pct:.1f}%) -> "
        f"Nexus Verified = {rep.verified_safe_count}/{rep.num_incidents} ({rep.verified_safe_pct:.1f}%, "
        f"Lift: +{rep.safety_lift_pct:.1f}%)"
    )


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8090, "--port"),
) -> None:
    """Launch the NexusVerity FastAPI & React Workbench server."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
