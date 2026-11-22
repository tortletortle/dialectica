"""DIALECTICA V0.3: online competence estimation and budgeted agent selection."""
from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional

@dataclass
class BetaBelief:
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    impact: float = 1.0
    uncertainty: float = 1.0
    conflict: float = 1.0
    estimated_cost: float = 1.0

    def priority(self) -> float:
        vals = (self.impact, self.uncertainty, self.conflict, self.estimated_cost)
        if any(v < 0 for v in vals) or self.estimated_cost == 0:
            raise ValueError("task scores must be non-negative and cost > 0")
        return self.impact * self.uncertainty * self.conflict / self.estimated_cost

@dataclass
class AgentProfile:
    name: str
    domains: Dict[str, BetaBelief] = field(default_factory=dict)

    def belief(self, domain: str) -> BetaBelief:
        return self.domains.setdefault(domain, BetaBelief())

@dataclass(frozen=True)
class Selection:
    task_id: str
    agent: str
    sampled_skill: float
    score: float
    exploratory: bool

class ThompsonScheduler:
    """Thompson sampling with a minimum exploration probability."""
    def __init__(self, agents: List[AgentProfile], seed: int = 0,
                 exploration_floor: float = 0.10):
        if not agents:
            raise ValueError("at least one agent is required")
        if not 0.0 <= exploration_floor <= 1.0:
            raise ValueError("exploration_floor must be in [0, 1]")
        self.agents = agents
        self.rng = random.Random(seed)
        self.exploration_floor = exploration_floor
        self.selections: List[Selection] = []

    def select(self, task: Task) -> Selection:
        if task.priority() <= 0:
            raise ValueError("task priority must be positive")
        exploratory = self.rng.random() < self.exploration_floor
        if exploratory:
            agent = self.rng.choice(self.agents)
            sampled = agent.belief(task.domain).sample(self.rng)
        else:
            draws = [(a.belief(task.domain).sample(self.rng), a) for a in self.agents]
            sampled, agent = max(draws, key=lambda pair: (pair[0], pair[1].name))
        score = task.priority() * sampled
        result = Selection(task.task_id, agent.name, sampled, score, exploratory)
        self.selections.append(result)
        return result

    def observe(self, agent_name: str, domain: str, success: bool) -> None:
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean evaluator outcome")
        matches = [a for a in self.agents if a.name == agent_name]
        if len(matches) != 1:
            raise ValueError("agent name must identify exactly one agent")
        matches[0].belief(domain).update(success)

    def snapshot(self) -> dict:
        return {
            a.name: {
                domain: {"alpha": belief.alpha, "beta": belief.beta,
                         "mean": belief.mean}
                for domain, belief in sorted(a.domains.items())
            } for a in self.agents
        }
