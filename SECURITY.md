# Security & Causal-Safety Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `1.0.x` | Yes |

## Causal & Statistical Soundness Guarantees

1. **Pearl Backdoor & Post-Treatment Collider Gate (`verify_backdoor_criterion`)**:
   - Rejects cyclic structural graphs and blocks conditioning on post-treatment descendants or colliders (`T -> C <- U`), preventing M-bias and selection-induced spurious associations.
2. **Neyman-Orthogonal Cross-Fitted Double ML (`run_double_ml_simpsons_benchmark`)**:
   - Enforces strict out-of-fold nuisance residualization across `K` disjoint folds, eliminating overfitting bias and reversing Simpson's Paradox sign errors.
3. **Distribution-Free Finite-Sample Split Conformal & Doubly-Robust OPE (`csrc/nexus_kernels.cpp`)**:
   - Enforces `k = ceil((n_cal + 1)*(1 - alpha))` order-statistic calibration and propensity clipping `max(eps, p_0(a|x))` to bound importance-weight variance.

## Reporting a Vulnerability

Please report vulnerabilities privately to **Nagarjuna Kamireddy** (`sendemailtonagarjuna@gmail.com`).
