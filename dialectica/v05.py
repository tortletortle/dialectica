"""V0.5 independent review, evidence provenance and dissent preservation."""
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List, Tuple

class EvidenceTier(IntEnum):
    UNVERIFIED_ASSERTION=0
    MODEL_GENERATED=1
    USER_SUPPLIED=2
    REPRODUCIBLE_ARTIFACT=3
    INDEPENDENTLY_REPLICATED=4

@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id:str
    claim_id:str
    description:str
    tier:EvidenceTier
    source_id:str
    independence_group:str
    verifier_note:str=""
    def __post_init__(self):
        if not all((self.evidence_id,self.claim_id,self.description,self.source_id,self.independence_group)):
            raise ValueError("evidence identity and provenance fields are required")

@dataclass(frozen=True)
class Review:
    reviewer_id:str
    claim_id:str
    stance:str
    rationale:str
    evidence_ids:Tuple[str,...]=()
    def __post_init__(self):
        if self.stance not in {"support","oppose","uncertain"}:
            raise ValueError("invalid stance")
        if not all((self.reviewer_id,self.claim_id,self.rationale)):
            raise ValueError("reviewer, claim and rationale are required")

@dataclass
class ReviewResult:
    claim_id:str
    support_count:int
    oppose_count:int
    uncertain_count:int
    dissenting_reviews:List[Review]
    evidence_groups:Tuple[str,...]
    evidence_tier_max:int
    status:str

class IndependentReviewBoard:
    """One review per reviewer/claim; minority positions remain visible."""
    def __init__(self):
        self.evidence:Dict[str,EvidenceRecord]={}
        self.reviews:Dict[Tuple[str,str],Review]={}
    def add_evidence(self,evidence:EvidenceRecord):
        if evidence.evidence_id in self.evidence: raise ValueError("duplicate evidence_id")
        self.evidence[evidence.evidence_id]=evidence
    def submit(self,review:Review):
        key=(review.claim_id,review.reviewer_id)
        if key in self.reviews: raise ValueError("duplicate reviewer for claim")
        if any(e not in self.evidence for e in review.evidence_ids): raise ValueError("unknown evidence ID")
        if any(self.evidence[e].claim_id!=review.claim_id for e in review.evidence_ids):
            raise ValueError("evidence belongs to another claim")
        self.reviews[key]=review
    def result(self,claim_id:str)->ReviewResult:
        rs=[r for (c,_),r in self.reviews.items() if c==claim_id]
        sup=[r for r in rs if r.stance=="support"]
        opp=[r for r in rs if r.stance=="oppose"]
        unc=[r for r in rs if r.stance=="uncertain"]
        ids={e for r in rs for e in r.evidence_ids}
        records=[self.evidence[e] for e in ids]
        groups=tuple(sorted({e.independence_group for e in records}))
        tier=max((int(e.tier) for e in records),default=0)
        if len(rs)<2: status="insufficient_independent_reviews"
        elif sup and opp: status="contested"
        elif opp and not sup: status="opposed"
        elif sup and not opp and not unc: status="supported_by_reviewers"
        else: status="unresolved"
        if sup and opp: minority=opp if len(opp)<=len(sup) else sup
        elif sup: minority=unc
        elif opp: minority=unc
        else: minority=rs
        return ReviewResult(claim_id,len(sup),len(opp),len(unc),list(minority),groups,tier,status)

def evidence_independence_count(records:Iterable[EvidenceRecord])->int:
    return len({r.independence_group for r in records})

@dataclass(frozen=True)
class CounterexampleCandidate:
    claim_id:str
    violated_assumption:str
    scenario:str
    expected_observation:str
    falsification_test:str
    def __post_init__(self):
        if not all((self.claim_id,self.violated_assumption,self.scenario,self.expected_observation,self.falsification_test)):
            raise ValueError("counterexample fields must be non-empty")

def validate_counterexample(candidate:CounterexampleCandidate,known_claim_ids:Iterable[str])->bool:
    """Structural linkage check only; does not prove or execute falsification."""
    return candidate.claim_id in set(known_claim_ids)
