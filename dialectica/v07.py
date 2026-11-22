"""DIALECTICA V0.7: Bayesian model discrimination and experiment selection.

The module operates on explicit finite hypothesis/experiment models. It computes
exact discrete EIG, expected 0-1 Bayes classification error, EVSI, prior-scenario
robust EIG, and a Pareto frontier. It does not verify whether supplied likelihoods
or utility values describe the real world.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
import random
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

TOL = 1e-9


def _distribution(values: Mapping[str, float], label: str) -> Dict[str, float]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    result = {}
    for key, value in values.items():
        if not key or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"invalid probability in {label}")
        result[str(key)] = float(value)
    total = sum(result.values())
    if abs(total - 1.0) > TOL:
        raise ValueError(f"{label} must sum to 1 (got {total})")
    return result


def _prior(prior: Mapping[str, float]) -> Dict[str, float]:
    return _distribution(prior, "prior")


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    cost: float
    likelihoods: Mapping[str, Mapping[str, float]]
    mean_predictions: Optional[Mapping[str, float]] = None
    measurement_variance: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not isinstance(self.cost, (int, float)) or not isfinite(self.cost) or self.cost <= 0:
            raise ValueError("experiment cost must be finite and positive")
        if not self.likelihoods:
            raise ValueError("likelihoods must not be empty")
        normalized = {}
        outcome_set = None
        for hypothesis, probs in self.likelihoods.items():
            if not hypothesis:
                raise ValueError("hypothesis IDs must be non-empty")
            dist = _distribution(probs, f"likelihood[{hypothesis}]")
            if outcome_set is None:
                outcome_set = set(dist)
            elif set(dist) != outcome_set:
                raise ValueError("all hypotheses must use the same outcome labels")
            normalized[hypothesis] = dist
        object.__setattr__(self, "likelihoods", normalized)
        if self.mean_predictions is not None:
            if set(self.mean_predictions) != set(normalized):
                raise ValueError("mean_predictions must cover every hypothesis")
            if any(not isfinite(float(v)) for v in self.mean_predictions.values()):
                raise ValueError("mean predictions must be finite")
            object.__setattr__(self, "mean_predictions", {k: float(v) for k, v in self.mean_predictions.items()})
            if self.measurement_variance is None or not isfinite(self.measurement_variance) or self.measurement_variance <= 0:
                raise ValueError("positive measurement_variance required with mean_predictions")
        elif self.measurement_variance is not None:
            raise ValueError("measurement_variance requires mean_predictions")

    @property
    def hypotheses(self) -> Tuple[str, ...]:
        return tuple(self.likelihoods.keys())

    @property
    def outcomes(self) -> Tuple[str, ...]:
        return tuple(next(iter(self.likelihoods.values())).keys())


@dataclass(frozen=True)
class ExperimentMetrics:
    experiment_id: str
    cost: float
    expected_information_gain: float
    expected_bayes_error: float
    expected_value_sample_information: float
    net_value_sample_information: float
    mean_pairwise_total_variation: float
    minimum_pairwise_total_variation: float
    simple_mean_separation: Optional[float]


def _check_model(prior: Mapping[str, float], experiment: Experiment) -> Dict[str, float]:
    p = _prior(prior)
    if set(p) != set(experiment.likelihoods):
        raise ValueError("prior hypotheses must exactly match experiment likelihood hypotheses")
    return p


def predictive_outcomes(prior: Mapping[str, float], experiment: Experiment) -> Dict[str, float]:
    p = _check_model(prior, experiment)
    return {
        outcome: sum(p[h] * experiment.likelihoods[h][outcome] for h in p)
        for outcome in experiment.outcomes
    }


def posterior_after(prior: Mapping[str, float], experiment: Experiment, outcome: str) -> Dict[str, float]:
    p = _check_model(prior, experiment)
    if outcome not in experiment.outcomes:
        raise ValueError("unknown experiment outcome")
    joint = {h: p[h] * experiment.likelihoods[h][outcome] for h in p}
    evidence = sum(joint.values())
    if evidence <= 0:
        raise ValueError("observed outcome has zero probability under every hypothesis")
    return {h: joint[h] / evidence for h in p}


def bayes_update(prior: Mapping[str, float], experiment: Experiment, outcome: str) -> Dict[str, float]:
    """Return the posterior after one observed outcome."""
    return posterior_after(prior, experiment, outcome)


@dataclass(frozen=True)
class ModelFitDiagnostic:
    outcome_probability: float
    surprise_nats: float
    possible_misspecification: bool


def posterior_predictive_diagnostic(
    prior: Mapping[str, float], experiment: Experiment, outcome: str,
    minimum_probability: float = 0.01,
) -> ModelFitDiagnostic:
    """Flag outcomes assigned very low probability by the current model.

    This is a screening heuristic, not a calibrated goodness-of-fit test or proof
    that an omitted hypothesis exists.
    """
    if not 0 < minimum_probability < 1:
        raise ValueError("minimum_probability must be in (0, 1)")
    probabilities = predictive_outcomes(prior, experiment)
    probability = probabilities.get(outcome, 0.0)
    surprise = float("inf") if probability <= 0 else -log(probability)
    return ModelFitDiagnostic(probability, surprise, probability < minimum_probability)


def expected_information_gain(prior: Mapping[str, float], experiment: Experiment) -> float:
    """Exact mutual information I(H;Y|e), measured in nats."""
    p = _check_model(prior, experiment)
    prior_entropy = -sum(v * log(v) for v in p.values() if v > 0)
    expected_posterior_entropy = 0.0
    for outcome, py in predictive_outcomes(p, experiment).items():
        if py <= 0:
            continue
        post = posterior_after(p, experiment, outcome)
        entropy = -sum(v * log(v) for v in post.values() if v > 0)
        expected_posterior_entropy += py * entropy
    return max(0.0, prior_entropy - expected_posterior_entropy)


def expected_bayes_error(prior: Mapping[str, float], experiment: Experiment) -> float:
    """Expected posterior 0-1 classification error after observing the experiment."""
    p = _check_model(prior, experiment)
    correct_probability = 0.0
    for outcome in experiment.outcomes:
        correct_probability += max(p[h] * experiment.likelihoods[h][outcome] for h in p)
    return min(1.0, max(0.0, 1.0 - correct_probability))


def _validate_utilities(prior: Mapping[str, float], utilities: Mapping[str, Mapping[str, float]]) -> None:
    if not utilities:
        raise ValueError("at least one decision action is required")
    for action, by_hypothesis in utilities.items():
        if not action or set(by_hypothesis) != set(prior):
            raise ValueError("each action must specify utility for every hypothesis")
        if any(not isfinite(float(v)) for v in by_hypothesis.values()):
            raise ValueError("utilities must be finite")


def expected_value_sample_information(
    prior: Mapping[str, float],
    experiment: Experiment,
    utilities: Mapping[str, Mapping[str, float]],
) -> float:
    """EVSI under a finite action/hypothesis utility table."""
    p = _check_model(prior, experiment)
    _validate_utilities(p, utilities)
    current_best = max(sum(p[h] * utilities[a][h] for h in p) for a in utilities)
    expected_post_best = 0.0
    for outcome in experiment.outcomes:
        joint = {h: p[h] * experiment.likelihoods[h][outcome] for h in p}
        py = sum(joint.values())
        if py <= 0:
            continue
        best_unnormalized = max(
            sum(joint[h] * utilities[a][h] for h in p) for a in utilities
        )
        expected_post_best += best_unnormalized
    return max(0.0, expected_post_best - current_best)


def pairwise_total_variation(experiment: Experiment, prior: Mapping[str, float]) -> Tuple[float, float]:
    """Prior-pair-weighted mean and minimum pairwise total-variation distance."""
    p = _check_model(prior, experiment)
    pairs = []
    hypotheses = list(p)
    for i, left in enumerate(hypotheses):
        for right in hypotheses[i + 1:]:
            tv = 0.5 * sum(
                abs(experiment.likelihoods[left][y] - experiment.likelihoods[right][y])
                for y in experiment.outcomes
            )
            pairs.append((p[left] * p[right], tv))
    if not pairs:
        return 0.0, 0.0
    weight = sum(w for w, _ in pairs)
    mean = sum(w * value for w, value in pairs) / weight if weight else 0.0
    return mean, min(value for _, value in pairs)


def simple_hypothesis_mean_separation(experiment: Experiment) -> Optional[float]:
    """Worst-pair squared mean separation / variance for simple hypotheses.

    This is a simple-hypothesis diagnostic inspired by T-optimality; it is not
    the full classical T-optimal criterion for composite models with nuisance
    parameters and an optimized design measure.
    """
    if experiment.mean_predictions is None:
        return None
    values = list(experiment.mean_predictions.values())
    if len(values) < 2:
        return 0.0
    return min(
        (values[i] - values[j]) ** 2
        for i in range(len(values)) for j in range(i + 1, len(values))
    ) / float(experiment.measurement_variance)


def evaluate_experiment(
    prior: Mapping[str, float],
    experiment: Experiment,
    utilities: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> ExperimentMetrics:
    p = _check_model(prior, experiment)
    eig = expected_information_gain(p, experiment)
    error = expected_bayes_error(p, experiment)
    evsi = expected_value_sample_information(p, experiment, utilities) if utilities is not None else 0.0
    mean_tv, min_tv = pairwise_total_variation(experiment, p)
    return ExperimentMetrics(
        experiment.experiment_id, float(experiment.cost), eig, error, evsi,
        evsi - float(experiment.cost), mean_tv, min_tv,
        simple_hypothesis_mean_separation(experiment),
    )


def robust_expected_information_gain(
    prior_scenarios: Sequence[Mapping[str, float]], experiment: Experiment
) -> float:
    """Worst-case EIG across supplied plausible prior scenarios.

    This is robust-to-prior sensitivity, not a full R-IDeA implementation or a
    guarantee against arbitrary likelihood/model misspecification.
    """
    if not prior_scenarios:
        raise ValueError("at least one prior scenario is required")
    return min(expected_information_gain(p, experiment) for p in prior_scenarios)


def add_unknown_hypothesis(
    prior: Mapping[str, float],
    experiments: Sequence[Experiment],
    unknown_id: str,
    unknown_likelihoods: Mapping[str, Mapping[str, float]],
    prior_mass: float = 0.05,
) -> Tuple[Dict[str, float], Tuple[Experiment, ...]]:
    """Add an explicit residual hypothesis with caller-specified predictions.

    The unknown model is not learned automatically. Its predictive distributions
    must be supplied and justified by the caller.
    """
    p = _prior(prior)
    if unknown_id in p or not unknown_id or not 0 < prior_mass < 1:
        raise ValueError("invalid unknown hypothesis ID or prior mass")
    if set(unknown_likelihoods) != {e.experiment_id for e in experiments}:
        raise ValueError("unknown_likelihoods must cover every experiment exactly")
    updated_prior = {h: value * (1 - prior_mass) for h, value in p.items()}
    updated_prior[unknown_id] = prior_mass
    updated_experiments = []
    for experiment in experiments:
        if set(experiment.likelihoods) != set(p):
            raise ValueError("base experiment hypotheses must match prior")
        likelihood = dict(experiment.likelihoods)
        likelihood[unknown_id] = unknown_likelihoods[experiment.experiment_id]
        updated_experiments.append(Experiment(
            experiment.experiment_id, experiment.cost, likelihood,
            mean_predictions=None, measurement_variance=None
        ))
    return updated_prior, tuple(updated_experiments)


def pareto_frontier(metrics: Sequence[ExperimentMetrics]) -> Tuple[ExperimentMetrics, ...]:
    """Non-dominated experiments: maximize EIG/EVSI/TV and minimize cost."""
    frontier = []
    for candidate in metrics:
        dominated = False
        for other in metrics:
            if other.experiment_id == candidate.experiment_id:
                continue
            no_worse = (
                other.expected_information_gain >= candidate.expected_information_gain - TOL
                and other.expected_value_sample_information >= candidate.expected_value_sample_information - TOL
                and other.mean_pairwise_total_variation >= candidate.mean_pairwise_total_variation - TOL
                and other.cost <= candidate.cost + TOL
            )
            strictly_better = (
                other.expected_information_gain > candidate.expected_information_gain + TOL
                or other.expected_value_sample_information > candidate.expected_value_sample_information + TOL
                or other.mean_pairwise_total_variation > candidate.mean_pairwise_total_variation + TOL
                or other.cost < candidate.cost - TOL
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda m: m.experiment_id))


def select_experiment(
    prior: Mapping[str, float],
    experiments: Sequence[Experiment],
    strategy: str = "eig_per_cost",
    utilities: Optional[Mapping[str, Mapping[str, float]]] = None,
    rng: Optional[random.Random] = None,
    prior_scenarios: Optional[Sequence[Mapping[str, float]]] = None,
) -> Experiment:
    if not experiments:
        raise ValueError("no candidate experiments")
    if len({e.experiment_id for e in experiments}) != len(experiments):
        raise ValueError("experiment IDs must be unique")
    if strategy == "random":
        return (rng or random.Random()).choice(list(experiments))
    metrics = [evaluate_experiment(prior, e, utilities) for e in experiments]
    by_id = {e.experiment_id: e for e in experiments}
    if strategy == "eig":
        key = lambda m: (m.expected_information_gain, -m.cost, m.experiment_id)
    elif strategy == "bayes_error":
        key = lambda m: (-m.expected_bayes_error, -m.cost, m.experiment_id)
    elif strategy == "evsi":
        if utilities is None:
            raise ValueError("EVSI selection requires an explicit decision utility table")
        key = lambda m: (m.expected_value_sample_information, -m.cost, m.experiment_id)
    elif strategy == "evsi_per_cost":
        if utilities is None:
            raise ValueError("EVSI selection requires an explicit decision utility table")
        key = lambda m: (m.expected_value_sample_information / m.cost,
                         m.expected_value_sample_information, m.experiment_id)
    elif strategy == "simple_mean_separation":
        if any(m.simple_mean_separation is None for m in metrics):
            raise ValueError("all candidate experiments need mean_predictions for this strategy")
        key = lambda m: (m.simple_mean_separation, -m.cost, m.experiment_id)
    elif strategy == "eig_per_cost":
        key = lambda m: (m.expected_information_gain / m.cost, m.expected_information_gain, m.experiment_id)
    elif strategy == "robust_eig":
        if not prior_scenarios:
            raise ValueError("robust_eig selection requires prior_scenarios")
        robust_scores = {
            e.experiment_id: robust_expected_information_gain(prior_scenarios, e)
            for e in experiments
        }
        return max(experiments, key=lambda e: (robust_scores[e.experiment_id], -e.cost, e.experiment_id))
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    best = max(metrics, key=key)
    return by_id[best.experiment_id]
