"""FastAPI backend serving Pearl Causal DAG/PC, Cross-Fitted DML, Split Conformal & C++20 DR OPE."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexus_verity.audit.ollama_causal import (
    default_causal_artifact_path,
    run_real_causal_ollama_audit,
)
from nexus_verity.causal.dag_pc import run_causal_discovery_and_backdoor_benchmark
from nexus_verity.dml.double_ml import run_double_ml_simpsons_benchmark
from nexus_verity.kernels.native_ope import (
    active_backend_name,
    run_conformal_and_dr_policy_benchmark,
)
from nexus_verity.types import (
    CausalDiscoveryReport,
    ConformalAndOPEReport,
    DoubleMLReport,
    RealCausalAuditReport,
)


class DMLRequest(BaseModel):
    num_samples: int = Field(default=2000, ge=400, le=6000)
    num_folds: int = Field(default=5, ge=2, le=10)


def create_app() -> FastAPI:
    app = FastAPI(
        title="NexusVerity — Causal Discovery, Double/Debiased ML, Split Conformal & C++20 DR Policy Engine",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "nexus-verity",
            "version": "1.0.0",
            "kernel_backend": active_backend_name(),
        }

    @app.get("/api/v1/causal/discover", response_model=CausalDiscoveryReport)
    def api_discover_causal(num_samples: int = 2500) -> CausalDiscoveryReport:
        return run_causal_discovery_and_backdoor_benchmark(num_samples=num_samples)

    @app.post("/api/v1/dml/estimate", response_model=DoubleMLReport)
    def api_estimate_dml(req: DMLRequest) -> DoubleMLReport:
        return run_double_ml_simpsons_benchmark(
            num_samples=req.num_samples, num_folds=req.num_folds
        )

    @app.get("/api/v1/conformal/evaluate", response_model=ConformalAndOPEReport)
    def api_evaluate_conformal(alpha: float = 0.10) -> ConformalAndOPEReport:
        return run_conformal_and_dr_policy_benchmark(alpha=alpha)

    @app.get("/api/v1/audit/ollama", response_model=RealCausalAuditReport)
    def api_audit_ollama(refresh: bool = False) -> RealCausalAuditReport:
        art = default_causal_artifact_path()
        if not refresh and art.exists():
            return RealCausalAuditReport.model_validate_json(art.read_text(encoding="utf-8"))
        return run_real_causal_ollama_audit(allow_offline_replay=True)

    ui_dist = Path(__file__).resolve().parents[3] / "ui" / "dist"
    assets_dir = ui_dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_model=None)
    def serve_spa() -> FileResponse | HTMLResponse:
        index_html = ui_dist / "index.html"
        if index_html.exists():
            return FileResponse(str(index_html))
        return HTMLResponse(
            "<h1>NexusVerity API Online</h1><p>Build UI via <code>cd ui && npm run build</code>.</p>"
        )

    return app
