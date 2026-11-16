"""Live Ollama (`qwen3.5:4b`) Causal Backdoor, Simpson's Paradox & Conformal CEGIS Audit."""

from nexus_verity.audit.ollama_causal import (
    build_causal_incident_suite,
    run_real_causal_ollama_audit,
)

__all__ = ["build_causal_incident_suite", "run_real_causal_ollama_audit"]
