"""V0.8 controlled misspecification benchmark.

The true simulator and working inference model are intentionally different for
some candidate experiments. Common random seeds are used across strategies.
This is a didactic finite-model stress test, not empirical evidence about LLMs.
"""
from __future__ import annotations

import json
import math
import random
from typing import Dict, Mapping

from .v08 import (
    ModelPairExperiment,
    evaluate_robust_information,
    select_robust_experiment,
)


PRIOR = {"A": 0.5, "B": 0.5}


def benchmark_experiments():
    return [
        ModelPairExperiment(
            "overconfident_misspec", 1.0,
            {"A": {"yes": .60, "no": .40}, "B": {"yes": .40, "no": .60}},
            {"A": {"yes": .99, "no": .01}, "B": {"yes": .01, "no": .99}},
        ),
        ModelPairExperiment(
            "spurious_signal", .8,
            {"A": {"yes": .50, "no": .50}, "B": {"yes": .50, "no": .50}},
            {"A": {"yes": .90, "no": .10}, "B": {"yes": .10, "no": .90}},
        ),
        ModelPairExperiment(
            "calibrated_signal", 1.0,
            {"A": {"yes": .75, "no": .25}, "B": {"yes": .25, "no": .75}},
            {"A": {"yes": .75, "no": .25}, "B": {"yes": .25, "no": .75}},
        ),
        ModelPairExperiment(
            "strong_calibrated_signal", 1.4,
            {"A": {"yes": .85, "no": .15}, "B": {"yes": .15, "no": .85}},
            {"A": {"yes": .85, "no": .15}, "B": {"yes": .15, "no": .85}},
        ),
    ]


def _draw(distribution: Mapping[str, float], u: float) -> str:
    cumulative = 0.0
    for outcome, probability in distribution.items():
        cumulative += probability
        if u <= cumulative:
            return outcome
    return next(reversed(distribution))


def _posterior(prior, likelihoods, outcome):
    joint = {h: prior[h] * likelihoods[h][outcome] for h in prior}
    total = sum(joint.values())
    return {h: joint[h] / total for h in prior}


def _kl(p, q):
    value = 0.0
    for h, mass in p.items():
        if mass > 0:
            value += mass * math.log(mass / q[h])
    return value


def run_benchmark(runs: int = 5000, seed: int = 20261123) -> dict:
    if runs < 1:
        raise ValueError("runs must be positive")
    experiments = benchmark_experiments()
    by_id = {e.experiment_id: e for e in experiments}
    strategies = ("random", "working_eig", "true_eig_oracle", "egig", "robust_kl")
    selected = {}
    criterion_scores = {}
    for strategy in strategies:
        if strategy == "random":
            selected[strategy] = "random"
            continue
        choice = select_robust_experiment(PRIOR, experiments, strategy)
        selected[strategy] = choice.experiment_id
        criterion_scores[strategy] = evaluate_robust_information(PRIOR, choice).__dict__

    rows = {}
    for strategy in strategies:
        records = []
        for i in range(runs):
            truth_rng = random.Random(seed + i * 104729)
            design_rng = random.Random(seed + i * 130363 + 17)
            observation_rng = random.Random(seed + i * 155921 + 31)
            truth = "A" if truth_rng.random() < PRIOR["A"] else "B"
            if strategy == "random":
                experiment_id = design_rng.choice(list(by_id))
            else:
                experiment_id = selected[strategy]
            experiment = by_id[experiment_id]
            outcome = _draw(experiment.true_likelihoods[truth], observation_rng.random())
            p_work = _posterior(PRIOR, experiment.working_likelihoods, outcome)
            p_true = _posterior(PRIOR, experiment.true_likelihoods, outcome)
            prediction = max(p_work, key=p_work.get)
            log_loss = -math.log(max(p_work[truth], 1e-15))
            brier = sum((p_work[h] - (1.0 if h == truth else 0.0)) ** 2 for h in PRIOR)
            records.append({
                "correct": prediction == truth,
                "log_loss": log_loss,
                "brier": brier,
                "confidence": max(p_work.values()),
                "posterior_kl": _kl(p_true, p_work),
            })
        rows[strategy] = {
            "selected_experiment": selected[strategy],
            "accuracy": sum(r["correct"] for r in records) / runs,
            "mean_log_loss": sum(r["log_loss"] for r in records) / runs,
            "mean_brier": sum(r["brier"] for r in records) / runs,
            "mean_confidence": sum(r["confidence"] for r in records) / runs,
            "mean_posterior_kl": sum(r["posterior_kl"] for r in records) / runs,
        }

    return {
        "metadata": {
            "runs": runs,
            "seed": seed,
            "prior": PRIOR,
            "designs": len(experiments),
            "note": "Hand-authored finite likelihoods. True likelihoods are known only to the simulator. No LLM or real-world system is evaluated.",
        },
        "candidate_criteria": {
            e.experiment_id: evaluate_robust_information(PRIOR, e).__dict__
            for e in experiments
        },
        "strategy_selection": selected,
        "selected_criterion_scores": criterion_scores,
        "strategies": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
