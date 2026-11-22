# DIALECTICA

Adaptive multi-agent deliberation research prototype.

## V0.4
Adds discounted competence estimates, a recent-vs-history drift alarm, weak cross-domain prior transfer, Wilson lower-bound abstention, and pairwise correlated-failure monitoring.

## Run
```bash
python -m unittest discover -s tests -v
python -m dialectica.demo
```

## Important limitations
- Synthetic/scripted agent evaluation only; no real LLM integration.
- Outcomes are evaluator-provided binary labels, not verified ground truth.
- Drift alarm is a simple window heuristic, not a calibrated statistical change-point test.
- Pairwise co-failure frequency is not causal dependence and is unreliable with small samples.
- Wilson bounds assume independent Bernoulli trials; repeated or correlated tasks violate this assumption.
- Cross-domain transfer is a weak prior, not demonstrated transfer learning.
- No semantic argument graph, retrieval, LangGraph runtime, or 3D interface.
