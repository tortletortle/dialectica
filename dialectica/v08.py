"""DIALECTICA V0.8: robust information criteria under model misspecification.

Implements finite discrete versions of EGIG and the KL-penalized robust
f-information criterion described by Callahan & Catanach (AISTATS 2026).

Both the true data-generating likelihood and the working inference likelihood
must be supplied explicitly. This is a controlled-model research primitive,
not an automatic oracle for real-world truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
import random
from typing import Dict, Mapping, Optional, Sequence, Tuple

TOL = 1e-9


def _distribution(values: Mapping[str, float], label: str) -> Dict[str, float]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    result: Dict[str, float] = {}
    for key, value in values.items():
        if not key or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"invalid probability in {label}")
        result[str(key)] = float(value)
    total = sum(result.values())
    if abs(total - 1.0) > TOL:
        raise ValueError(f"{label} must sum to 1 (got {total})")
    return result


@dataclass(frozen=True)
class ModelPairExperiment:
    experiment_id: str
    cost: float
    true_likelihoods: Mapping[str, Mapping[str, float]]
    working_likelihoods: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not isinstance(self.cost, (int, float)) or not isfinite(self.cost) or self.cost <= 0:
            raise ValueError("cost must be finite and positive")
        if not self.true_likelihoods or set(self.true_likelihoods) != set(self.working_likelihoods):
            raise ValueError("true and working models must contain the same hypotheses")
        true = {}
        working = {}
        outcomes = None
        for hypothesis in self.true_likelihoods:
            if not hypothesis:
                raise ValueError("hypothesis IDs must be non-empty")
            true_dist = _distribution(self.true_likelihoods[hypothesis],
                                      f"true_likelihood[{hypothesis}]")
            working_dist = _distribution(self.working_likelihoods[hypothesis],
                                          f"working_likelihood[{hypothesis}]")
            if set(true_dist) != set(working_dist):
                raise ValueError("true and working models must use identical outcome labels")
            if outcomes is None:
                outcomes = set(true_dist)
            elif set(true_dist) != outcomes:
                raise ValueError("all hypotheses must use identical outcome labels")
            true[hypothesis] = true_dist
            working[hypothesis] = working_dist
        object.__setattr__(self, "true_likelihoods", true)
        object.__setattr__(self, "working_likelihoods", working)

    @property
    def hypotheses(self) -> Tuple[str, ...]:
        return tuple(self.true_likelihoods)

    @property
    def outcomes(self) -> Tuple[str, ...]:
        return tuple(next(iter(self.true_likelihoods.values())).keys())


@dataclass(frozen=True)
class RobustInformationMetrics:
    experiment_id: str
    cost: float
    working_eig: float
    true_eig: float
    egig: float
    posterior_error_penalty: float
    robust_kl_information: float
    likelihood_error_penalty: float


def _check_prior(prior: Mapping[str, float], experiment: ModelPairExperiment) -> Dict[str, float]:
    p = _distribution(prior, "prior")
    if set(p) != set(experiment.hypotheses):
        raise ValueError("prior hypotheses must exactly match the model pair")
    return p


def _posterior(prior: Mapping[str, float],
               likelihoods: Mapping[str, Mapping[str, float]],
               outcome: str) -> Dict[str, float]:
    if outcome not in next(iter(likelihoods.values())):
        raise ValueError("unknown outcome")
    joint = {h: prior[h] * likelihoods[h][outcome] for h in prior}
    evidence = sum(joint.values())
    if evidence <= 0:
        raise ValueError("outcome has zero probability under this model")
    return {h: joint[h] / evidence for h in prior}


def _predictive(prior: Mapping[str, float],
                likelihoods: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    outcomes = next(iter(likelihoods.values())).keys()
    return {y: sum(prior[h] * likelihoods[h][y] for h in prior) for y in outcomes}


def _kl(p: Mapping[str, float], q: Mapping[str, float], label: str) -> float:
    total = 0.0
    for key, probability in p.items():
        if probability <= 0:
            continue
        other = q.get(key, 0.0)
        if other <= 0:
            raise ValueError(f"{label}: working distribution lacks support for positive true mass")
        total += probability * log(probability / other)
    return total


def expected_information_gain_for_model(
    prior: Mapping[str, float],
    likelihoods: Mapping[str, Mapping[str, float]],
) -> float:
    """Expected KL(posterior || prior) under the supplied model."""
    p = _distribution(prior, "prior")
    predictive = _predictive(p, likelihoods)
    value = 0.0
    for outcome, py in predictive.items():
        if py <= 0:
            continue
        post = _posterior(p, likelihoods, outcome)
        value += py * _kl(post, p, "EIG")
    return max(0.0, value)


def working_eig(prior: Mapping[str, float], experiment: ModelPairExperiment) -> float:
    p = _check_prior(prior, experiment)
    return expected_information_gain_for_model(p, experiment.working_likelihoods)


def true_eig(prior: Mapping[str, float], experiment: ModelPairExperiment) -> float:
    p = _check_prior(prior, experiment)
    return expected_information_gain_for_model(p, experiment.true_likelihoods)


def posterior_error_penalty(prior: Mapping[str, float],
                            experiment: ModelPairExperiment) -> float:
    """E_true[Y] KL(p_true(theta|Y) || p_work(theta|Y))."""
    p = _check_prior(prior, experiment)
    true_predictive = _predictive(p, experiment.true_likelihoods)
    penalty = 0.0
    for outcome, py in true_predictive.items():
        if py <= 0:
            continue
        p_true = _posterior(p, experiment.true_likelihoods, outcome)
        p_work = _posterior(p, experiment.working_likelihoods, outcome)
        penalty += py * _kl(p_true, p_work, "posterior mismatch")
    return max(0.0, penalty)


def expected_generalized_information_gain(
    prior: Mapping[str, float],
    experiment: ModelPairExperiment,
) -> float:
    """EGIG = E_{true joint}[log p_work(theta|y,d) / p(theta)].

    Unlike ordinary EIG, EGIG may be negative under misspecification.
    """
    p = _check_prior(prior, experiment)
    true_predictive = _predictive(p, experiment.true_likelihoods)
    value = 0.0
    for outcome, py in true_predictive.items():
        if py <= 0:
            continue
        p_work = _posterior(p, experiment.working_likelihoods, outcome)
        p_true = _posterior(p, experiment.true_likelihoods, outcome)
        for hypothesis, true_posterior_mass in p_true.items():
            if true_posterior_mass <= 0:
                continue
            if p_work[hypothesis] <= 0:
                raise ValueError("EGIG is not finite: working posterior excludes true posterior support")
            value += py * true_posterior_mass * log(p_work[hypothesis] / p[hypothesis])
    return value


def likelihood_error_penalty(prior: Mapping[str, float],
                             experiment: ModelPairExperiment) -> float:
    """E_prior[KL(p_true(y|theta,d) || p_work(y|theta,d))]."""
    p = _check_prior(prior, experiment)
    penalty = 0.0
    for hypothesis, mass in p.items():
        if mass > 0:
            penalty += mass * _kl(
                experiment.true_likelihoods[hypothesis],
                experiment.working_likelihoods[hypothesis],
                "likelihood mismatch",
            )
    return max(0.0, penalty)


def robust_kl_information(prior: Mapping[str, float],
                          experiment: ModelPairExperiment) -> float:
    """RFIM instantiated with KL: true EIG minus conditional likelihood KL penalty."""
    return true_eig(prior, experiment) - likelihood_error_penalty(prior, experiment)


def evaluate_robust_information(
    prior: Mapping[str, float],
    experiment: ModelPairExperiment,
) -> RobustInformationMetrics:
    return RobustInformationMetrics(
        experiment.experiment_id,
        float(experiment.cost),
        working_eig(prior, experiment),
        true_eig(prior, experiment),
        expected_generalized_information_gain(prior, experiment),
        posterior_error_penalty(prior, experiment),
        robust_kl_information(prior, experiment),
        likelihood_error_penalty(prior, experiment),
    )


def verify_egig_decomposition(prior: Mapping[str, float],
                              experiment: ModelPairExperiment,
                              tolerance: float = 1e-8) -> bool:
    metrics = evaluate_robust_information(prior, experiment)
    return abs(metrics.egig - (
        metrics.true_eig - metrics.posterior_error_penalty
    )) <= tolerance


def select_robust_experiment(
    prior: Mapping[str, float],
    experiments: Sequence[ModelPairExperiment],
    strategy: str = "egig",
    rng: Optional[random.Random] = None,
) -> ModelPairExperiment:
    if not experiments:
        raise ValueError("no candidate experiments")
    if len({e.experiment_id for e in experiments}) != len(experiments):
        raise ValueError("experiment IDs must be unique")
    if strategy == "random":
        return (rng or random.Random()).choice(list(experiments))
    metrics = [(e, evaluate_robust_information(prior, e)) for e in experiments]
    if strategy == "working_eig":
        key = lambda item: (item[1].working_eig, -item[1].cost, item[0].experiment_id)
    elif strategy == "true_eig_oracle":
        key = lambda item: (item[1].true_eig, -item[1].cost, item[0].experiment_id)
    elif strategy == "egig":
        key = lambda item: (item[1].egig, -item[1].cost, item[0].experiment_id)
    elif strategy == "robust_kl":
        key = lambda item: (item[1].robust_kl_information, -item[1].cost, item[0].experiment_id)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return max(metrics, key=key)[0]
