# ADR 0002: Chernozhukov K-Fold Cross-Fitted Neyman-Orthogonal Double/Debiased ML

- **Status**: Accepted
- **Context**: In clinical and enterprise observational data, treatment propensity correlates with baseline risk (`Simpson's Paradox`), while naive ML plug-in estimators suffer $N^{-1/4}$ regularization bias.
- **Decision**: Implement $K$-fold cross-fitted residualization (`y_tilde`, `t_tilde`) with Neyman-orthogonal influence-function standard errors.
- **Consequences**: Reverses Simpson's sign error and recovers true structural uplift `+2.50` within valid asymptotic `95%` confidence bounds.
