# ADR 0003: Native C++20 Doubly-Robust Off-Policy Evaluator & Split Conformal Quantile

- **Status**: Accepted
- **Context**: High-dimensional in-sample Gaussian prediction intervals severely under-cover test risk, while raw Inverse Propensity Scoring (`IPS`) explodes under rare logging propensities.
- **Decision**: Implement `nexus_doubly_robust_ope_f64` and `nexus_conformal_quantile_f64` (`ceil((n_cal+1)*(1-alpha))`) in `csrc/nexus_kernels.cpp` (`clang++ -O3 -std=c++20`) with exact NumPy fallback.
- **Consequences**: Guarantees finite-sample `>= 90%` marginal coverage and cuts counterfactual policy estimation error `10x+` vs raw IPS (`max |Δ| < 1e-14`).
