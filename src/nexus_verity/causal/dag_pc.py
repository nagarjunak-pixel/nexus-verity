"""Pearl Causal DAG, Lauritzen Moralized Ancestral `d`-Separation, Backdoor Criterion & `PC` Discovery.

Implements:
1. Exact acyclicity verification + ancestor/descendant closure on `CausalDAG`.
2. Lauritzen-Dawid-Larsen-Leimer moralized ancestral graph equivalence for `d`-separation.
3. Pearl's Backdoor Adjustment Criterion (proving why conditioning on Confounders blocks backdoor
   paths while conditioning on a Collider or Mediator/Descendant introduces severe bias!).
4. Spirtes-Glymour-Scheines (SGS / PC) Fisher-z partial-correlation skeleton & v-structure recovery.
"""

from __future__ import annotations

import itertools
import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from nexus_verity.types import AdjustmentSetCheck, CausalDiscoveryReport


class CausalDAG:
    def __init__(self, nodes: list[str], directed_edges: list[tuple[str, str]]) -> None:
        self.nodes = list(dict.fromkeys(nodes))
        self.node_set = set(self.nodes)
        self.parents: dict[str, set[str]] = {u: set() for u in self.nodes}
        self.children: dict[str, set[str]] = {u: set() for u in self.nodes}

        for u, v in directed_edges:
            if u not in self.node_set or v not in self.node_set:
                raise ValueError(f"Unknown node in edge ({u}, {v})")
            if u == v:
                raise ValueError(f"Self-loop ({u}, {u}) forbidden in DAG")
            self.children[u].add(v)
            self.parents[v].add(u)

        if not self._is_acyclic():
            raise ValueError("Directed cycle detected; CausalDAG must be strictly acyclic")

    def _is_acyclic(self) -> bool:
        indeg = {u: len(self.parents[u]) for u in self.nodes}
        q = deque([u for u, d in indeg.items() if d == 0])
        visited = 0
        while q:
            cur = q.popleft()
            visited += 1
            for ch in self.children[cur]:
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    q.append(ch)
        return visited == len(self.nodes)

    def edges(self) -> list[tuple[str, str]]:
        return sorted((u, v) for u in self.nodes for v in self.children[u])

    def ancestors_of(self, seeds: set[str]) -> set[str]:
        anc = set(seeds)
        q = deque(seeds)
        while q:
            cur = q.popleft()
            for p in self.parents[cur]:
                if p not in anc:
                    anc.add(p)
                    q.append(p)
        return anc

    def descendants_of(self, seed: str) -> set[str]:
        desc: set[str] = set()
        q = deque(self.children[seed])
        while q:
            cur = q.popleft()
            if cur not in desc:
                desc.add(cur)
                q.extend(self.children[cur])
        return desc

    def is_d_separated(
        self,
        x: str,
        y: str,
        conditioning: set[str],
        *,
        removed_outgoing_from: set[str] | None = None,
    ) -> bool:
        """Exact Lauritzen moralized ancestral graph test for (X _||_ Y | Z) in G (or G_{underline X})."""
        if x == y or x in conditioning or y in conditioning:
            raise ValueError("x, y must be distinct and disjoint from conditioning set")
        drop_out = removed_outgoing_from or set()

        eff_parents: dict[str, set[str]] = {u: set() for u in self.nodes}
        for u in self.nodes:
            if u in drop_out:
                continue
            for v in self.children[u]:
                eff_parents[v].add(u)

        # 1. Ancestral closure of {x, y} U conditioning in the effective DAG
        anc = {x, y} | set(conditioning)
        q = deque(anc)
        while q:
            cur = q.popleft()
            for p in eff_parents[cur]:
                if p not in anc:
                    anc.add(p)
                    q.append(p)

        # 2. Moralize: connect all co-parents of any node in `anc`, then make all edges in `anc` undirected
        adj: dict[str, set[str]] = {u: set() for u in anc}
        for v in anc:
            ps = [p for p in eff_parents[v] if p in anc]
            for p in ps:
                adj[p].add(v)
                adj[v].add(p)
            for p1, p2 in itertools.combinations(ps, 2):
                adj[p1].add(p2)
                adj[p2].add(p1)

        # 3. Remove `conditioning` nodes and test reachability from x to y
        blocked = set(conditioning)
        seen = {x}
        bfs = deque([x])
        while bfs:
            cur = bfs.popleft()
            if cur == y:
                return False
            for nxt in adj[cur]:
                if nxt not in blocked and nxt not in seen:
                    seen.add(nxt)
                    bfs.append(nxt)
        return True

    def verify_backdoor_criterion(
        self, treatment: str, outcome: str, z_set: set[str]
    ) -> tuple[bool, str]:
        """Pearl's Backdoor Criterion:
        (i) No node in Z is a descendant of `treatment` in G, AND
        (ii) Z d-separates `treatment` and `outcome` in the mutilated graph G_{underline(treatment)}.
        """
        if treatment in z_set or outcome in z_set:
            return False, "Conditioning set cannot contain treatment or outcome."
        desc_t = self.descendants_of(treatment)
        bad_desc = sorted(z_set & desc_t)
        if bad_desc:
            return (
                False,
                f"Violates Pearl Rule 1: {bad_desc} is a descendant/collider/mediator of {treatment} (post-treatment / M-bias).",
            )
        if not self.is_d_separated(treatment, outcome, z_set, removed_outgoing_from={treatment}):
            return (
                False,
                f"Violates Pearl Rule 2: open backdoor or collider-opened path remains between {treatment} and {outcome}.",
            )
        return (
            True,
            "Valid Pearl Backdoor adjustment set (all backdoor paths blocked, zero collider bias).",
        )


def _partial_corr(corr: NDArray[np.float64], i: int, j: int, cond: tuple[int, ...]) -> float:
    if not cond:
        return float(corr[i, j])
    idx = [i, j, *cond]
    sub = corr[np.ix_(idx, idx)]
    prec = np.linalg.pinv(sub, rcond=1e-9)
    denom = math.sqrt(max(1e-15, float(prec[0, 0] * prec[1, 1])))
    return float(np.clip(-prec[0, 1] / denom, -0.999999, 0.999999))


def pc_discover_skeleton_and_v_structures(
    data: NDArray[np.float64], var_names: list[str], z_threshold: float = 2.58
) -> tuple[list[tuple[str, str]], list[str]]:
    """Spirtes-Glymour-Scheines (PC) Fisher-z conditional independence skeleton & v-structure finder."""
    n_samples, p = data.shape
    corr: NDArray[np.float64] = np.asarray(np.corrcoef(data, rowvar=False), dtype=np.float64)
    adj: dict[int, set[int]] = {i: {j for j in range(p) if j != i} for i in range(p)}
    sep_sets: dict[tuple[int, int], set[int]] = {}

    for ord_k in range(p - 1):
        adj_snapshot = {u: set(nbrs) for u, nbrs in adj.items()}
        pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if j in adj[i]]
        for i, j in pairs:
            found_sep = False
            for endpoint, other in ((i, j), (j, i)):
                neighbors = sorted(adj_snapshot[endpoint] - {other})
                if len(neighbors) < ord_k:
                    continue
                for cond in itertools.combinations(neighbors, ord_k):
                    r = _partial_corr(corr, i, j, cond)
                    dof = max(1, n_samples - len(cond) - 3)
                    fisher_z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(dof)
                    if abs(fisher_z) < z_threshold:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        sep_sets[(i, j)] = set(cond)
                        sep_sets[(j, i)] = set(cond)
                        found_sep = True
                        break
                if found_sep:
                    break

    skeleton = sorted(
        (var_names[i], var_names[j]) for i in range(p) for j in range(i + 1, p) if j in adj[i]
    )

    # Identify unshielded colliders (v-structures): i - k - j with i !~ j and k not in SepSet(i, j)
    v_structs: list[str] = []
    for k in range(p):
        nbrs = sorted(adj[k])
        for i, j in itertools.combinations(nbrs, 2):
            if j not in adj[i]:
                sep = sep_sets.get((i, j), set())
                if k not in sep:
                    v_structs.append(f"{var_names[i]} -> {var_names[k]} <- {var_names[j]}")

    return skeleton, sorted(v_structs)


def run_causal_discovery_and_backdoor_benchmark(num_samples: int = 2500) -> CausalDiscoveryReport:
    """Benchmark Pearl Backdoor + Collider Rejection + PC Structural Discovery on 5-node SEM."""
    # True DAG:
    # Severity (Z) -> Treatment (T), Severity (Z) -> Outcome (Y),
    # GeneticRisk (U) -> Outcome (Y), Treatment (T) -> Outcome (Y),
    # Treatment (T) -> Hospitalization (C) <- GeneticRisk (U)  [C is a post-treatment COLLIDER!]
    variables = ["Severity", "GeneticRisk", "Treatment", "Outcome", "Hospitalization"]
    true_edges = [
        ("Severity", "Treatment"),
        ("Severity", "Outcome"),
        ("GeneticRisk", "Outcome"),
        ("Treatment", "Outcome"),
        ("Treatment", "Hospitalization"),
        ("GeneticRisk", "Hospitalization"),
    ]
    dag = CausalDAG(variables, true_edges)

    rng = np.random.default_rng(2026)
    sev = rng.normal(0.0, 1.0, size=num_samples)
    gen = rng.normal(0.0, 1.0, size=num_samples)
    treat = 1.05 * sev + rng.normal(0.0, 0.75, size=num_samples)
    outcome = 1.25 * treat + 0.95 * sev + 1.15 * gen + rng.normal(0.0, 0.70, size=num_samples)
    hosp = 1.55 * treat - 1.35 * gen + rng.normal(0.0, 0.55, size=num_samples)

    data = np.column_stack([sev, gen, treat, outcome, hosp])
    disc_skel, v_structs = pc_discover_skeleton_and_v_structures(data, variables)

    true_skel = {tuple(sorted((u, v))) for u, v in true_edges}
    found_skel = {tuple(sorted((u, v))) for u, v in disc_skel}
    shd = len(true_skel ^ found_skel)

    candidates: list[set[str]] = [
        set(),
        {"Severity"},
        {"Severity", "Hospitalization"},
        {"Hospitalization"},
        {"Severity", "GeneticRisk"},
    ]
    checks: list[AdjustmentSetCheck] = []
    for cset in candidates:
        ok, reason = dag.verify_backdoor_criterion("Treatment", "Outcome", cset)
        checks.append(
            AdjustmentSetCheck(
                candidate_set=sorted(cset),
                valid_backdoor=ok,
                reason=reason,
            )
        )

    collider_ok, _ = dag.verify_backdoor_criterion(
        "Treatment", "Outcome", {"Severity", "Hospitalization"}
    )
    return CausalDiscoveryReport(
        num_samples=num_samples,
        variables=variables,
        true_edges=true_edges,
        discovered_skeleton_edges=disc_skel,
        discovered_v_structures=v_structs,
        skeleton_shd=shd,
        collider_bias_rejected=(not collider_ok),
        minimal_valid_backdoor_set=["Severity"],
        candidate_checks=checks,
    )
