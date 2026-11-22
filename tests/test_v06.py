import unittest
from dialectica.v06 import *

class V06Tests(unittest.TestCase):
    def make_hypothesis(self):
        a=Assumption("a1","sensor calibration is stable",.9,"calibration drift")
        return Hypothesis("h1","c1","sensor detects occupancy",(a,),
                          ("detection rate exceeds 95% under calibrated conditions",))
    def make_competing(self):
        return CompetingExplanation("alt1","apparent detection is caused by background motion",
                                    "false positives rise when background motion is introduced")
    def test_assumption_validation(self):
        with self.assertRaises(ValueError):
            Assumption("a","bad",1.2,"indicator")
    def test_hypothesis_requires_assumptions(self):
        with self.assertRaises(ValueError):
            Hypothesis("h","c","claim",(),("prediction",))
    def test_generate_distinguishing_test(self):
        t=generate_discriminating_test(self.make_hypothesis(),self.make_competing(),
             "t1","introduce controlled background motion","false-positive rate",
             "false-positive rate remains below 2%",("fixed sensor position","same lighting"))
        self.assertNotEqual(t.expected_if_target,t.expected_if_competing)
        self.assertEqual(len(t.controls),2)
    def test_reject_non_discriminating_test(self):
        h=self.make_hypothesis()
        alt=CompetingExplanation("alt","same","detection rate exceeds 95% under calibrated conditions")
        with self.assertRaises(ValueError):
            generate_discriminating_test(h,alt,"t","i","m","threshold",("control",))
    def test_conflict_detection(self):
        obs=[EvidenceObservation("e1","c1","supports","source-A","positive"),
             EvidenceObservation("e2","c1","contradicts","source-B","negative")]
        found=locate_evidence_conflicts(obs)
        self.assertEqual(len(found),1)
        self.assertEqual(found[0].claim_id,"c1")
    def test_no_conflict_when_same_polarity(self):
        obs=[EvidenceObservation("e1","c1","supports","A","one"),
             EvidenceObservation("e2","c1","supports","B","two")]
        self.assertEqual(locate_evidence_conflicts(obs),[])
    def test_provenance_groups_deduplicated(self):
        obs=[EvidenceObservation("e1","c1","supports","same","one"),
             EvidenceObservation("e2","c1","supports","same","duplicate"),
             EvidenceObservation("e3","c1","contradicts","other","opposition")]
        x=locate_evidence_conflicts(obs)[0]
        self.assertEqual(x.independent_support_groups,("same",))
    def test_duplicate_evidence_id_rejected(self):
        e=EvidenceObservation("e","c","supports","s","d")
        with self.assertRaises(ValueError): locate_evidence_conflicts([e,e])
    def test_experiment_requires_audit_fields(self):
        with self.assertRaises(ValueError):
            ExperimentRecord("t",Outcome.SUPPORTS,"observed","","evaluator","env")
    def test_inconclusive_cannot_be_called_confirmation(self):
        e=ExperimentRecord("t",Outcome.INCONCLUSIVE,"ambiguous","artifact://x","reviewer","lab-A")
        with self.assertRaises(ValueError):
            revise_claim_from_experiment("c","old","new","confirmed",e)
    def test_inconclusive_can_preserve_uncertainty(self):
        e=ExperimentRecord("t",Outcome.INCONCLUSIVE,"ambiguous","artifact://x","reviewer","lab-A")
        r=revise_claim_from_experiment("c","old","old","inconclusive; retain uncertainty",e)
        self.assertEqual(r.outcome,Outcome.INCONCLUSIVE)
    def test_revision_links_experiment(self):
        e=ExperimentRecord("t",Outcome.CONTRADICTS,"threshold missed","artifact://x","reviewer","lab-A")
        r=revise_claim_from_experiment("c","claim holds","claim needs narrowing","reproducible counterexample",e)
        self.assertEqual(r.experiment_test_id,"t")

if __name__=="__main__": unittest.main()
