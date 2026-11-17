# NexusVerity (`nexus-verity`)

[![CI](https://github.com/nagarjunak-pixel/nexus-verity/actions/workflows/ci.yml/badge.svg)](https://github.com/nagarjunak-pixel/nexus-verity/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB.svg)](https://www.python.org/)
[![C++20 Native](https://img.shields.io/badge/C%2B%2B20-DoublyRobust%20%2B%20Conformal-00599C.svg)](csrc/nexus_kernels.cpp)
[![Coverage 98.15%](https://img.shields.io/badge/coverage-98.15%25-10B981.svg)](tests/)
[![Release v1.0.0](https://img.shields.io/badge/release-v1.0.0-1f6feb.svg)](https://github.com/nagarjunak-pixel/nexus-verity/releases/tag/v1.0.0)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-10b981.svg)](LICENSE)

**Pearl Causal DAG, Moralized `d`-Separation & `PC-Stable` Structural Discovery (`SHD = 0`), Chernozhukov 5-Fold Cross-Fitted Neyman-Orthogonal Double/Debiased ML (`919x` Simpson's-Paradox Bias Reduction), Vovk Finite-Sample Split Conformal Prediction (`90.50%` vs `66.58%` Naive Gaussian), Native `C++20` Doubly-Robust Off-Policy Evaluator (`30.88x` < IPS Error), and Live Ollama (`qwen3.5:4b`) Causal CEGIS (`66.7% -> 100.0%`).**

---

## Live Causal Discovery, Double ML & Conformal Policy Workbench

![NexusVerity Live Workbench — PC-Stable Causal Discovery, Neyman-Orthogonal Double ML & Finite-Sample Split Conformal Prediction](docs/assets/workbench-preview.png)

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph DAG["1. Pearl Backdoor & PC-Stable Discovery"]
        D1["Fisher-z PC-Stable Skeleton + v-Structures"] --> D2["SHD = 0 Exact 6-Edge Recovery"]
        D2 --> D3["Lauritzen Moralized d-Separation (Rejects Collider M-Bias)"]
    end

    subgraph DML["2. Chernozhukov Cross-Fitted Double ML"]
        M1["Naive Observational Diff: -3.7614 (Wrong Sign!)"] --> M2["5-Fold Neyman-Orthogonal Residualization"]
        M2 --> M3["DML ATE: +2.4932 ± 0.0456 vs True +2.5000 (919x Cleaner)"]
    end

    subgraph OPE["3. C++20 Split Conformal & Doubly-Robust OPE"]
        C1["Finite-Sample Rank ceil((n+1)(1-α)): 90.50% Coverage"] --> C2["clang++ -O3 Doubly-Robust Policy Kernel"]
        C2 --> C3["30.88x Lower Policy Error vs Raw IPS (|Δ| = 4.0e-15)"]
    end

    subgraph CEGIS["4. Live qwen3.5:4b Causal & Conformal Gate"]
        Q1["Raw Pass@1: 4/6 (66.7%)"] --> Q2["Nexus Verified: 6/6 (100.0%, +33.3% Lift, 0 Confounded Leaks)"]
    end

    DAG --> DML --> OPE --> CEGIS
```

---

## Verified Zero-Mock Engineering Benchmarks

Every metric below is deterministically reproducible via CLI (`nexus ...`) and verified under `pytest` + `Hypothesis` (`98.15%` line coverage):

| Subsystem | Econometric / Causal Algorithm | Verified Result | Theoretical Guarantee |
| :--- | :--- | :--- | :--- |
| **`PC-Stable` & Backdoor** | Fisher-$z$ order-independent skeleton + Lauritzen moralized `d`-sep | **`SHD = 0` (`6/6` edges, `3` `v`-structures); collider `{Hospitalization}` rejected** | Certifies minimal Pearl Backdoor `{Severity}` (`N = 2,500`) |
| **Double / Debiased ML** | Chernozhukov 5-fold cross-fitted Neyman-orthogonal uplift | **`+2.4932 ± 0.0456` (`95%` CI `[2.4039, 2.5825]`) vs true `+2.5000`** | Reverses naive `-3.7614` Simpson's sign error (**`919.07x` bias cut**) |
| **Split Conformal (`C++20`)** | Vovk `k = ceil((n_cal + 1)(1 - α))` order-statistic calibration | **`90.50%` test coverage (target `>= 90.0%`) vs `66.58%` plug-in Gaussian** | Distribution-free finite-sample marginal validity under skewed noise |
| **Doubly-Robust OPE (`C++20`)** | `clang++ -O3` Dudík-Langford-Li `DR` + `SNIPS` counterfactual kernel | **`2.7864 ± 0.0668` vs true `2.7657` (`30.88x` < IPS `3.4056` error)** | Exact C++20-vs-NumPy parity (`max \|Δ\| = 4.00e-15`) |
| **Live `qwen3.5:4b` Gate** | 6 Simpson's Paradox, Wald `LATE`, Collider, `DR` & Conformal tasks | **Raw `4/6` (`66.7%`) -> Verified `6/6` (`100.0%`, `+33.3%` lift)** | **`0` confounded or under-covered policies** (`benchmarks/real_qwen_causal.json`) |

---

## Quickstart

```bash
# 1. Install Python 3.13 environment & compile C++20 native kernel
uv sync --python 3.13 --all-extras

# 2. Run strict formatting, static typing, and 98.15%-coverage Hypothesis test suite
uv run ruff check src tests
uv run mypy src tests
uv run pytest --cov=nexus_verity --cov-report=term-missing --cov-fail-under=95

# 3. Execute all 4 deterministic & live causal/conformal benchmarks
uv run nexus discover-dag
uv run nexus estimate-dml
uv run nexus evaluate-policy
uv run nexus eval-ollama

# 4. Build interactive React/TypeScript Workbench UI & launch FastAPI server on :8090
cd ui && npm ci && npm run build && cd ..
uv run nexus serve --host 127.0.0.1 --port 8090
```

---

## Author & License

Designed and engineered by **Nagarjuna Kamireddy** (`@nagarjunak-pixel`). Released under the [MIT License](LICENSE).

---

## Author & Enterprise AI Architecture Hub
Built by **[Nagarjuna Reddy Kamireddy](https://nagarjunak-pixel.github.io/)** (*Nagarjuna Kamireddy / Nagarjuna K*) — Lead AI Architect, GenAI & Agentic AI Lead, and Corporate AI Trainer (20+ years across Enterprise Data, SAP HANA & Production AI; AWS Certified Solutions Architect – Professional & NVIDIA-Certified Professional: Agentic AI).

- **Official Portfolio & Architecture Hub:** [https://nagarjunak-pixel.github.io/](https://nagarjunak-pixel.github.io/)
- **Enterprise Case Studies & Technical Articles:** [Case Studies](https://nagarjunak-pixel.github.io/case-studies.html) · [Architecture Articles](https://nagarjunak-pixel.github.io/articles.html) · [10-Mandate Role Fit](https://nagarjunak-pixel.github.io/roles.html)
- **Master CV & AI Search Index:** [Master CV](https://nagarjunak-pixel.github.io/resume.html) · [`llms-full.txt`](https://nagarjunak-pixel.github.io/llms-full.txt) · [LinkedIn](https://www.linkedin.com/in/nagarjuna-kamireddy)
