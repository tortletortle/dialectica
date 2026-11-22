# DIALECTICA

Adaptive multi-agent deliberation and experimental-reasoning research prototype.

## Current version: V0.7

V0.7 adds a finite Bayesian model-discrimination engine:
- exact discrete expected information gain (EIG);
- expected 0–1 Bayes classification error;
- expected value of sample information (EVSI), using explicit utilities;
- EIG/cost and EVSI/cost selection baselines;
- robust EIG across supplied prior scenarios;
- a low-predictive-probability model-mismatch screening diagnostic;
- explicit residual/unknown hypothesis support;
- pairwise predictive separation and a Pareto frontier;
- a reproducible synthetic CPU/memory/GPU/unknown benchmark.

## Run

    python -m unittest discover -s tests -v
    python -m dialectica.benchmark_v07

## Research note and results

Read docs/v0.7-research-note.md and docs/references.md. Machine-readable results are in benchmarks/v0.7-results.json and benchmarks/v0.7-results.csv.

## Important limitations

- Synthetic, hand-authored likelihoods; no real hardware measurements.
- No real LLM agents, LangGraph orchestration, retrieval pipeline, or 3D interface.
- EIG/EVSI are exact only relative to the supplied finite likelihood and utility models.
- The residual hypothesis is explicitly supplied; it is not discovered automatically.
- Robust EIG only varies supplied priors; it does not solve arbitrary model misspecification.
- The mean-separation score is not full classical T-optimal design.
- Predictive surprise is a screening heuristic, not a calibrated statistical test.
