# DIALECTICA

Adaptive multi-agent deliberation and experimental-reasoning research prototype.

## Current version: V0.8

V0.8 adds finite-discrete robust information criteria under explicit model misspecification:
- working-model EIG and oracle true-model EIG;
- Expected Generalized Information Gain (EGIG);
- expected posterior-mismatch KL penalty;
- KL-instantiated Robust f-Information Measure;
- support checks for finite posterior divergences;
- weighted-mean ensemble EGIG over plausible true-model scenarios and a worst-case sensitivity score;
- a reproducible synthetic stress test where working-model EIG becomes overconfident and inaccurate.

## Run

    python -m unittest discover -s tests -v
    python -m dialectica.benchmark_v07
    python -m dialectica.benchmark_v08

## Research notes and results

- docs/v0.7-research-note.md — finite Bayesian experiment selection.
- docs/v0.8-research-note.md — EGIG, robust information, and plausible-model ensembles.
- docs/v0.9-ridea-research-plan.md — faithful R-IDeA reproduction plan.
- docs/references.md — core papers.
- benchmarks/v0.7-results.json and benchmarks/v0.7-results.csv.
- benchmarks/v0.8-results.json and benchmarks/v0.8-results.csv.

## Important limitations

- V0.8 uses hand-specified finite categorical true and working likelihoods.
- The true model is an oracle available only in the synthetic benchmark; real deployments need model ensembles, external evidence, or sensitivity analysis.
- Ensemble EGIG depends on the caller-supplied plausible model set and scenario weights; worst-case EGIG is a conservative extension, not canonical EGIG.
- EGIG and RFIM-KL are exact only for supplied likelihood tables and the shared-prior assumption.
- No real LLM agents, LangGraph orchestration, automated retrieval, hardware experiment execution, or 3D interface.
- R-IDeA has been researched but not ported into the finite hypothesis engine; its original regression/active-learning assumptions require a separate reproduction benchmark.
