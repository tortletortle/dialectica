# DIALECTICA

Adaptive multi-agent deliberation research prototype.

## V0.3 scope
This release studies online agent selection under uncertain competence estimates.
It is a simulation harness, **not** a validated reasoning system and does not call LLM APIs.

## Run
```bash
python -m unittest discover -s tests -v
python -m dialectica.demo
```

## Model
Each agent has a Beta posterior per task domain. Thompson Sampling selects an agent using sampled competence, with a bounded exploration floor. Feedback updates the posterior only from externally supplied evaluation outcomes.

## Limitations
- Simulated agents and synthetic tasks only.
- Reward labels are assumed to be supplied by an evaluator; the prototype does not establish truth.
- No semantic duplicate detection, evidence retrieval, LangGraph integration, UI, or 3D scene.
- Results from this simulator cannot support claims about real-world decision accuracy.
