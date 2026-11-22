"""DIALECTICA V0.6: hypothesis decomposition and falsification planning.

This module creates auditable experiment plans. It does not execute experiments,
verify evidence authenticity, or decide factual truth.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

class Outcome(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"

@dataclass(frozen=True)
class Assumption:
    assumption_id: str
    statement: str
    criticality: float
    observable_indicator: str
    def __post_init__(self):
        if not self.assumption_id or not self.statement or not self.observable_indicator:
            raise ValueError("assumption fields must be non-empty")
        if not 0 <= self.criticality <= 1:
            raise ValueError("criticality must be in [0, 1]")

@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    claim_id: str
    statement: str
    assumptions: Tuple[Assumption, ...]
    predicted_observations: Tuple[str, ...]
    def __post_init__(self):
        if not all((self.hypothesis_id, self.claim_id, self.statement)):
            raise ValueError("hypothesis identity and statement are required")
        if not self.assumptions or not self.predicted_observations:
            raise ValueError("hypothesis needs assumptions and predictions")

@dataclass(frozen=True)
class CompetingExplanation:
    explanation_id: str
    statement: str
    predicted_observation: str
    def __post_init__(self):
        if not all((self.explanation_id, self.statement, self.predicted_observation)):
            raise ValueError("competing explanation fields are required")

@dataclass(frozen=True)
class DiscriminatingTest:
    test_id: str
    target_hypothesis_id: str
    competing_explanation_id: str
    intervention: str
    measured_variable: str
    expected_if_target: str
    expected_if_competing: str
    pass_threshold: str
    controls: Tuple[str, ...]
    def __post_init__(self):
        fields = (self.test_id, self.target_hypothesis_id, self.competing_explanation_id,
                  self.intervention, self.measured_variable, self.expected_if_target,
                  self.expected_if_competing, self.pass_threshold)
        if not all(fields) or not self.controls:
            raise ValueError("test must specify competing predictions, threshold and controls")
        if self.expected_if_target.strip().lower() == self.expected_if_competing.strip().lower():
            raise ValueError("test does not distinguish the competing explanations")

def generate_discriminating_test(target: Hypothesis,
                                 competing: CompetingExplanation,
                                 test_id: str,
                                 intervention: str,
                                 measured_variable: str,
                                 pass_threshold: str,
                                 controls: Sequence[str]) -> DiscriminatingTest:
    """Build a test specification; the caller must validate feasibility and run it."""
    target_prediction = target.predicted_observations[0]
    return DiscriminatingTest(
        test_id, target.hypothesis_id, competing.explanation_id,
        intervention, measured_variable, target_prediction,
        competing.predicted_observation, pass_threshold, tuple(controls))

@dataclass(frozen=True)
class EvidenceObservation:
    evidence_id: str
    claim_id: str
    polarity: str  # supports / contradicts
    source_group: str
    description: str
    def __post_init__(self):
        if self.polarity not in {"supports", "contradicts"}:
            raise ValueError("polarity must be supports or contradicts")
        if not all((self.evidence_id, self.claim_id, self.source_group, self.description)):
            raise ValueError("evidence provenance fields are required")

@dataclass(frozen=True)
class EvidenceConflict:
    claim_id: str
    supporting_ids: Tuple[str, ...]
    contradicting_ids: Tuple[str, ...]
    independent_support_groups: Tuple[str, ...]
    independent_contradiction_groups: Tuple[str, ...]

def locate_evidence_conflicts(observations: Iterable[EvidenceObservation]) -> List[EvidenceConflict]:
    """Find opposing evidence for the same claim; group duplicates by provenance."""
    grouped: Dict[str, List[EvidenceObservation]] = {}
    seen = set()
    for item in observations:
        if item.evidence_id in seen:
            raise ValueError("duplicate evidence_id")
        seen.add(item.evidence_id)
        grouped.setdefault(item.claim_id, []).append(item)
    conflicts = []
    for claim_id, items in sorted(grouped.items()):
        support = [x for x in items if x.polarity == "supports"]
        oppose = [x for x in items if x.polarity == "contradicts"]
        if support and oppose:
            conflicts.append(EvidenceConflict(
                claim_id,
                tuple(sorted(x.evidence_id for x in support)),
                tuple(sorted(x.evidence_id for x in oppose)),
                tuple(sorted({x.source_group for x in support})),
                tuple(sorted({x.source_group for x in oppose}))))
    return conflicts

@dataclass(frozen=True)
class ExperimentRecord:
    test_id: str
    outcome: Outcome
    observed_result: str
    artifact_uri: str
    evaluator_id: str
    environment: str
    def __post_init__(self):
        if not all((self.test_id, self.observed_result, self.artifact_uri,
                    self.evaluator_id, self.environment)):
            raise ValueError("experiment record requires result, artifact, evaluator and environment")

@dataclass(frozen=True)
class ClaimRevision:
    claim_id: str
    prior_position: str
    revised_position: str
    reason: str
    experiment_test_id: str
    outcome: Outcome
    def __post_init__(self):
        if not all((self.claim_id, self.prior_position, self.revised_position,
                    self.reason, self.experiment_test_id)):
            raise ValueError("revision fields must be non-empty")

def revise_claim_from_experiment(claim_id: str, prior_position: str,
                                 proposed_position: str, reason: str,
                                 experiment: ExperimentRecord) -> ClaimRevision:
    """Record a traceable revision proposal; inconclusive results cannot claim confirmation."""
    if experiment.outcome == Outcome.INCONCLUSIVE and proposed_position != prior_position:
        if "inconclusive" not in reason.lower() and "uncertain" not in reason.lower():
            raise ValueError("inconclusive evidence requires an uncertainty-qualified reason")
    return ClaimRevision(claim_id, prior_position, proposed_position, reason,
                         experiment.test_id, experiment.outcome)
