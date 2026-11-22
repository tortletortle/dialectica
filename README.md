# DIALECTICA

Adaptive multi-agent deliberation research prototype.

## V0.6
Adds:
- explicit hypothesis and critical-assumption decomposition;
- structured discriminating-test specifications for competing explanations;
- opposing-evidence detection with provenance-group deduplication;
- auditable experiment records and uncertainty-aware claim revision.

## Run
```bash
python -m unittest discover -s tests -v
python -m dialectica.demo
```

## Evidence and experiment semantics
V0.6 generates experiment plans; it does not execute experiments. A generated test is only a candidate until feasibility, controls, measurement validity and safety are reviewed. Evidence conflict means opposing records exist for a claim, not that one record is correct. Provenance groups are metadata supplied by the caller, not independently authenticated. Claim revisions must link to an experiment record; inconclusive outcomes cannot be labeled as confirmation.

## Limitations
- Scripted/synthetic agents only; no real LLM integration.
- No automated external retrieval, hardware control, or experiment execution.
- Evidence authenticity and evaluator independence are not automatically verified.
- No semantic argument graph, LangGraph runtime, or 3D interface.
- Prior versions' drift, Wilson-bound and co-failure metrics retain their stated statistical assumptions.
