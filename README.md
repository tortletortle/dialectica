# DIALECTICA

Adaptive multi-agent deliberation research prototype.

## V0.5
Adds independent-review aggregation, provenance-aware evidence grouping, explicit minority/dissent retention, and structurally validated counterexample candidates.

## Run
```bash
python -m unittest discover -s tests -v
python -m dialectica.demo
```

## Evidence semantics
Evidence tiers describe provenance and reproducibility metadata, not truth. Distinct independence groups are counted to avoid treating repeated citations from one origin as independent corroboration. Review aggregation is descriptive; a majority vote does not prove a claim. Counterexample validation checks only that a candidate references a known claim; empirical falsification requires a separate test and recorded result.

## Limitations
- Scripted/synthetic agents only; no real LLM integration.
- Reviewer independence is an input assumption, not enforced by separate model instances.
- Evidence authenticity and factual correctness are not automatically verified.
- No live retrieval, code execution sandbox, LangGraph runtime, or 3D interface.
- Prior versions' drift, Wilson-bound and co-failure metrics retain their stated statistical assumptions.
