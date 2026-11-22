"""DIALECTICA V0.4 adaptive competence safeguards.

All components consume evaluator-provided binary outcomes. They do not verify truth.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, Dict, Iterable, Mapping, Tuple

@dataclass
class DriftAwareBelief:
    """Discounted Beta belief with a simple recent-vs-history drift alarm."""
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    decay: float = 0.98
    window_size: int = 12
    min_window: int = 8
    drift_threshold: float = 0.30

    def __post_init__(self):
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if self.window_size < 4 or not 2 <= self.min_window <= self.window_size:
            raise ValueError("invalid drift window configuration")
        if not 0 < self.drift_threshold < 1:
            raise ValueError("drift_threshold must be in (0, 1)")
        self.alpha = float(self.prior_alpha)
        self.beta = float(self.prior_beta)
        self.history: Deque[bool] = deque(maxlen=self.window_size)
        self.drift_events = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def effective_n(self) -> float:
        return max(0.0, self.alpha + self.beta - self.prior_alpha - self.prior_beta)

    def update(self, success: bool) -> bool:
        if not isinstance(success, bool):
            raise TypeError("success must be bool")
        # Exponential forgetting: old evidence gradually loses influence.
        self.alpha = self.prior_alpha + self.decay * (self.alpha - self.prior_alpha)
        self.beta = self.prior_beta + self.decay * (self.beta - self.prior_beta)
        self.alpha += float(success)
        self.beta += float(not success)
        self.history.append(success)
        drifted = False
        if len(self.history) == self.window_size:
            values = list(self.history)
            half = len(values) // 2
            old_rate = sum(values[:half]) / half
            recent = values[half:]
            recent_rate = sum(recent) / len(recent)
            if abs(recent_rate - old_rate) >= self.drift_threshold:
                # Retain only the recent segment plus prior, not the stale baseline.
                self.alpha = self.prior_alpha + sum(recent)
                self.beta = self.prior_beta + len(recent) - sum(recent)
                self.drift_events += 1
                self.history = deque(recent, maxlen=self.window_size)
                drifted = True
        return drifted

    @classmethod
    def transferred(cls, source_mean: float, strength: float = 2.0, **kwargs):
        """Create a weak target-domain prior from a source-domain estimate."""
        if not 0 < source_mean < 1 or strength <= 0:
            raise ValueError("source_mean must be in (0,1), strength > 0")
        # Transfer is deliberately capped by caller-supplied small strength.
        return cls(prior_alpha=1 + strength * source_mean,
                   prior_beta=1 + strength * (1 - source_mean), **kwargs)

def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Wilson score lower bound for an observed Bernoulli success rate."""
    if trials < 0 or successes < 0 or successes > trials or z <= 0:
        raise ValueError("invalid binomial counts or z")
    if trials == 0:
        return 0.0
    p = successes / trials
    z2 = z * z
    denom = 1 + z2 / trials
    center = p + z2 / (2 * trials)
    margin = z * sqrt((p * (1 - p) + z2 / (4 * trials)) / trials)
    return max(0.0, (center - margin) / denom)

@dataclass
class AbstentionPolicy:
    min_trials: int = 8
    min_lower_bound: float = 0.45

    def __post_init__(self):
        if self.min_trials < 1 or not 0 <= self.min_lower_bound <= 1:
            raise ValueError("invalid abstention policy")

    def should_abstain(self, successes: int, trials: int) -> bool:
        if trials < self.min_trials:
            return True
        return wilson_lower_bound(successes, trials) < self.min_lower_bound

class CorrelatedFailureMonitor:
    """Tracks pairwise simultaneous failures on matched tasks."""
    def __init__(self):
        self.counts: Dict[Tuple[str, str], list[int]] = {}

    def observe(self, outcomes: Mapping[str, bool]) -> None:
        names = sorted(outcomes)
        if any(not isinstance(v, bool) for v in outcomes.values()):
            raise TypeError("outcomes must be booleans")
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                key = (left, right)
                row = self.counts.setdefault(key, [0, 0])
                row[1] += 1
                if not outcomes[left] and not outcomes[right]:
                    row[0] += 1

    def joint_failure_rate(self, left: str, right: str) -> float:
        key = tuple(sorted((left, right)))
        failures, trials = self.counts.get(key, [0, 0])
        return failures / trials if trials else 0.0

    def penalty(self, agent: str, selected_agents: Iterable[str]) -> float:
        rates = [self.joint_failure_rate(agent, other)
                 for other in selected_agents if other != agent]
        return max(rates, default=0.0)
