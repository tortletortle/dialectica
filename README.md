# DIALECTICA

Adaptive multi-agent deliberation and experimental-reasoning research prototype.

## Current research branch: V0.9 — R-IDeA reproduction

V0.9 adds a finite-grid reproduction of Representative, Informative, and De-Amplifying (R-IDeA) for one-dimensional polynomial regression:
- exact conjugate Gaussian posterior and analytic EIG;
- BAD/EIG, R-I, R-IDeA, current-initialization ablation, and best-in-class oracle;
- biased empirical squared MMD with an RBF kernel;
- paper-aligned quadratic misspecification and well-specified controls, plus a cubic high-misspecification stress case;
- paired bootstrap comparisons and lambda/tau sensitivity sweeps;
- source-code audit documenting differences between the paper and supplementary implementation.

## Run

    python -m unittest discover -s tests -v
    python -m dialectica.benchmark_v07
    python -m dialectica.benchmark_v08
    python -m dialectica.benchmark_v09 --runs 20 --budget 9 --lambda 0.5 --tau 0.5 --mmd-metric mmd2 --proxy-margin-tau 2

## V0.9 research artifacts

- dialectica/ridea_v09.py — exact Bayesian regression, EIG, MMD², R-I and R-IDeA.
- dialectica/benchmark_v09.py — reproducible benchmark CLI.
- tests/test_ridea_v09.py — 14 tests covering posterior, EIG, MMD, proxy, scoring and reproducibility.
- docs/v0.9-source-audit.md — paper and supplementary-code audit.
- docs/v0.9-research-note.md — reproduction results, uncertainty and limitations.
- docs/v0.9-ridea-research-plan.md — original experimental plan and acceptance criteria.
- benchmarks/v0.9-results.json — primary and sensitivity summaries.
- benchmarks/v0.9-results-primary.csv — primary benchmark metrics.
- benchmarks/v0.9-sensitivity.csv — lambda/tau sensitivity metrics.

## Main finding

At lambda=0.5 and DeA tau=0.5, the zero-initialized proxy improves over R-I in the quadratic-DGP/linear-learner case, but it does not reliably improve over R-I or Random in the cubic stress case. The oracle using the best-in-class projection performs much better than the learned proxy in the cubic case. Proxy quality and threshold calibration are the next research bottlenecks.

## Important limitations

- V0.7/V0.8 use finite hand-specified hypothesis and likelihood models.
- V0.9 is an equation-level regression reproduction, not a bit-for-bit reproduction of the paper's Pyro/NUTS implementation.
- The true DGP is known only to the simulator and oracle diagnostics; a real deployment needs external evidence or plausible-model ensembles.
- The proxy is currently linear in the supplementary-code reproduction, even in the well-specified quadratic control.
- No real LLM agents, LangGraph orchestration, automated retrieval, hardware experiment execution, or 3D interface are included.
- R-IDeA has not yet been transferred to language-model debate; target distributions and outcome metrics for that transfer remain unresolved.
