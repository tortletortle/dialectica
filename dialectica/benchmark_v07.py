"""Synthetic, reproducible benchmark for DIALECTICA V0.7 selection strategies.

The environment is intentionally small and fully specified. It tests the mechanics
of finite-model experiment selection, not real-world scientific discovery.
"""
from __future__ import annotations

import json
import math
import random
from typing import Dict, List, Mapping

from .v07 import Experiment, bayes_update, select_experiment

KNOWN = ("cpu", "memory", "gpu")
TRUTH_PRIOR = {"cpu": 0.30, "memory": 0.30, "gpu": 0.30, "other": 0.10}


def benchmark_experiments(include_other: bool = True) -> List[Experiment]:
    rows = {
        "cpu_frequency": {
            "cpu": {"improve": .82, "unchanged": .14, "worse": .04},
            "memory": {"improve": .12, "unchanged": .78, "worse": .10},
            "gpu": {"improve": .15, "unchanged": .75, "worse": .10},
            "other": {"improve": .33, "unchanged": .34, "worse": .33},
        },
        "memory_bandwidth": {
            "cpu": {"improve": .14, "unchanged": .78, "worse": .08},
            "memory": {"improve": .84, "unchanged": .12, "worse": .04},
            "gpu": {"improve": .13, "unchanged": .79, "worse": .08},
            "other": {"improve": .33, "unchanged": .34, "worse": .33},
        },
        "gpu_swap": {
            "cpu": {"improve": .13, "unchanged": .78, "worse": .09},
            "memory": {"improve": .12, "unchanged": .80, "worse": .08},
            "gpu": {"improve": .86, "unchanged": .10, "worse": .04},
            "other": {"improve": .33, "unchanged": .34, "worse": .33},
        },
        "workload_scale": {
            "cpu": {"improve": .55, "unchanged": .35, "worse": .10},
            "memory": {"improve": .52, "unchanged": .38, "worse": .10},
            "gpu": {"improve": .58, "unchanged": .32, "worse": .10},
            "other": {"improve": .33, "unchanged": .34, "worse": .33},
        },
        "cache_flush": {
            "cpu": {"improve": .22, "unchanged": .68, "worse": .10},
            "memory": {"improve": .25, "unchanged": .65, "worse": .10},
            "gpu": {"improve": .20, "unchanged": .70, "worse": .10},
            "other": {"improve": .70, "unchanged": .20, "worse": .10},
        },
    }
    costs = {"cpu_frequency": 1.0, "memory_bandwidth": 1.2, "gpu_swap": 1.5,
             "workload_scale": .7, "cache_flush": .6}
    means_all = {
        "cpu_frequency": {"cpu": 10.0, "memory": 2.0, "gpu": 2.0, "other": 1.0},
        "memory_bandwidth": {"cpu": 1.0, "memory": 10.0, "gpu": 1.0, "other": 1.0},
        "gpu_swap": {"cpu": 1.0, "memory": 1.0, "gpu": 10.0, "other": 1.0},
        "workload_scale": {"cpu": 6.0, "memory": 5.0, "gpu": 7.0, "other": 3.0},
        "cache_flush": {"cpu": 2.0, "memory": 2.0, "gpu": 2.0, "other": 8.0},
    }
    experiments = []
    for name, likelihoods in rows.items():
        if not include_other:
            likelihoods = {h: likelihoods[h] for h in KNOWN}
        means = means_all[name] if include_other else {h: means_all[name][h] for h in KNOWN}
        experiments.append(Experiment(name, costs[name], likelihoods, means, measurement_variance=4.0))
    return experiments


def _draw(distribution: Mapping[str, float], rng: random.Random) -> str:
    value = rng.random()
    cumulative = 0.0
    for outcome, probability in distribution.items():
        cumulative += probability
        if value <= cumulative:
            return outcome
    return next(reversed(distribution))


def _normalize_known_prior(include_other: bool) -> Dict[str, float]:
    if include_other:
        return dict(TRUTH_PRIOR)
    return {h: 1 / len(KNOWN) for h in KNOWN}


def _simulate_one(strategy: str, run_seed: int, open_world: bool, budget: int) -> dict:
    truth_rng = random.Random(run_seed)
    truth = _draw(TRUTH_PRIOR, truth_rng)
    inference_rng = random.Random(run_seed + 100_003)
    observation_rng = random.Random(run_seed + 200_003)
    prior = _normalize_known_prior(open_world)
    experiments = benchmark_experiments(include_other=open_world)
    full_experiments = {e.experiment_id: e for e in benchmark_experiments(include_other=True)}
    remaining = list(experiments)
    selected = []
    utilities = {
        f"choose_{h}": {candidate: float(candidate == h) for candidate in prior}
        for h in prior
    }
    for _ in range(min(budget, len(remaining))):
        chosen = select_experiment(prior, remaining, strategy=strategy,
                                   utilities=utilities, rng=inference_rng)
        selected.append(chosen.experiment_id)
        outcome = _draw(full_experiments[chosen.experiment_id].likelihoods[truth], observation_rng)
        prior = bayes_update(prior, chosen, outcome)
        remaining = [e for e in remaining if e.experiment_id != chosen.experiment_id]
    predicted = max(prior, key=prior.get)
    true_probability = prior.get(truth, 0.0)
    log_loss = -math.log(max(true_probability, 1e-12))
    return {
        "truth": truth,
        "prediction": predicted,
        "correct": predicted == truth,
        "unknown_truth": truth == "other",
        "unknown_predicted": predicted == "other",
        "posterior_max": max(prior.values()),
        "true_posterior": true_probability,
        "log_loss": log_loss,
        "selected": selected,
    }


def run_benchmark(runs: int = 300, budget: int = 3, seed: int = 20261123) -> dict:
    if runs < 1 or budget < 1:
        raise ValueError("runs and budget must be positive")
    strategies = ("random", "eig", "bayes_error", "eig_per_cost",
                  "evsi_per_cost", "simple_mean_separation")
    report = {"metadata": {
        "runs_per_condition": runs,
        "experiment_budget": budget,
        "seed": seed,
        "truth_prior": TRUTH_PRIOR,
        "note": "Synthetic benchmark; likelihoods are known to the simulator and do not represent empirical hardware behavior.",
    }, "conditions": {}}
    for condition, open_world in (("closed_world_model", False), ("explicit_unknown_model", True)):
        report["conditions"][condition] = {}
        for strategy in strategies:
            records = [_simulate_one(strategy, seed + i * 7919, open_world, budget)
                       for i in range(runs)]
            unknown_records = [r for r in records if r["unknown_truth"]]
            report["conditions"][condition][strategy] = {
                "accuracy": sum(r["correct"] for r in records) / runs,
                "mean_log_loss": sum(r["log_loss"] for r in records) / runs,
                "mean_max_posterior": sum(r["posterior_max"] for r in records) / runs,
                "unknown_recall": (sum(r["unknown_predicted"] for r in unknown_records) / len(unknown_records)
                                   if unknown_records else None),
                "mean_unknown_confidence": (sum(r["posterior_max"] for r in unknown_records) / len(unknown_records)
                                             if unknown_records else None),
                "mean_experiments_used": sum(len(r["selected"]) for r in records) / runs,
            }
    return report


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
