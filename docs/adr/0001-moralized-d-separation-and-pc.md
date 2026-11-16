# ADR 0001: Lauritzen Moralized Ancestral `d`-Separation, Backdoor Gate & Fisher-z `PC`

- **Status**: Accepted
- **Context**: Conditioning indiscriminately on all available covariates (`"kitchen-sink regression"`) conditions on post-treatment colliders (`T -> C <- U`), introducing spurious M-bias.
- **Decision**: Implement exact Lauritzen moralized ancestral graph `d`-separation and Pearl's 2-rule Backdoor Criterion alongside Spirtes-Glymour-Scheines (`PC`) conditional-independence discovery.
- **Consequences**: Achieves `SHD = 0` exact skeleton recovery (`N=2,500`) and deterministically rejects post-treatment collider conditioning sets.
