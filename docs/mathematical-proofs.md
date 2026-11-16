# Mathematical Proofs & Causal Invariants in `nexus-verity`

## 1. Lauritzen Moralized Ancestral `d`-Separation & Pearl's Backdoor Criterion

Given a directed acyclic graph $G = (V, E)$, a conditioning set $Z \subseteq V \setminus \{T, Y\}$ satisfies **Pearl's Backdoor Criterion** relative to $(T, Y)$ iff:
1. $Z \cap \operatorname{Desc}_G(T) = \emptyset$ (no post-treatment mediators or colliders), and
2. $T \perp\!\!\!\perp_{G_{\underline{T}}} Y \mid Z$, verified exactly by constructing the ancestral closure $A = \operatorname{Anc}_{G_{\underline{T}}}(\{T, Y\} \cup Z)$, moralizing all co-parents in $A$, deleting $Z$, and testing undirected disconnectivity.

Under these two graphical conditions, causal intervention reduces to observational G-computation:

$$\mathbb{E}[Y \mid \operatorname{do}(T = t)] = \sum_{z} \mathbb{E}[Y \mid T = t, Z = z]\, \mathbb{P}(Z = z)$$

---

## 2. Chernozhukov Neyman-Orthogonal Cross-Fitted Double / Debiased ML (`DML`)

In the partially linear structural model $Y = \theta_0 T + m_0(X) + \varepsilon$, $T = g_0(X) + \eta$ ($\mathbb{E}[\varepsilon \mid X, T] = 0$, $\mathbb{E}[\eta \mid X] = 0$), the partialling-out score

$$\psi(W; \theta, \eta_0) = \big(Y - m_0(X) - \theta (T - g_0(X))\big)\big(T - g_0(X)\big)$$

satisfies **Neyman Orthogonality**: its Gateaux derivative with respect to nuisance $\eta = (m, g)$ vanishes at $(\theta_0, \eta_0)$:

$$\left.\partial_\eta \mathbb{E}[\psi(W; \theta_0, \eta)]\right|_{\eta = \eta_0} = 0$$

Combining Neyman orthogonality with $K$-fold sample splitting ($\hat{m}^{(-k)}, \hat{g}^{(-k)}$) yields $\sqrt{N}(\hat{\theta}_{\text{DML}} - \theta_0) \xrightarrow{d} \mathcal{N}(0, \sigma_\psi^2)$ even when nuisance estimators converge at slower $o(N^{-1/4})$ ML rates.

---

## 3. Finite-Sample Split Conformal Coverage & Doubly-Robust Off-Policy Consistency

- **Split Conformal Prediction (Vovk / Lei et al.)**: For exchangeable calibration and test nonconformity scores $R_1, \dots, R_{n_{\text{cal}}}, R_{n_{\text{cal}}+1}$, the rank of $R_{n_{\text{cal}}+1}$ is uniform on $\{1, \dots, n_{\text{cal}}+1\}$. Selecting $\hat{q}$ at 1-indexed order statistic $k = \lceil(n_{\text{cal}}+1)(1-\alpha)\rceil$ guarantees:
  $$\mathbb{P}\big(Y_{\text{test}} \in [\hat{f}(X_{\text{test}}) - \hat{q},\; \hat{f}(X_{\text{test}}) + \hat{q}]\big) \ge 1 - \alpha$$
- **Doubly-Robust Policy Value (`csrc/nexus_kernels.cpp`)**:
  $$\hat{V}_{\text{DR}}(\pi) = \frac{1}{N} \sum_{i=1}^N \left[\hat{\mu}(x_i, \pi(x_i)) + \frac{\mathbb{I}\{a_i = \pi(x_i)\}}{p_0(a_i \mid x_i)}\big(r_i - \hat{\mu}(x_i, a_i)\big)\right]$$
  is consistent if **either** $\hat{\mu} = \mu$ **or** $p_0$ is exact, while subtracting $\hat{\mu}(x_i, a_i)$ dramatically reduces inverse-propensity variance.
