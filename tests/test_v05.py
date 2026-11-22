import unittest
from dialectica.v05 import *
class V05Tests(unittest.TestCase):
 def ev(self,eid="e1",group="paper-A",claim="c1"):
  return EvidenceRecord(eid,claim,"test artifact",EvidenceTier.REPRODUCIBLE_ARTIFACT,"src-"+eid,group)
 def test_dissent_preserved(self):
  b=IndependentReviewBoard(); b.submit(Review("r1","c1","support","A")); b.submit(Review("r2","c1","oppose","B"))
  x=b.result("c1"); self.assertEqual(x.status,"contested"); self.assertEqual(x.dissenting_reviews[0].reviewer_id,"r2")
 def test_one_review_not_consensus(self):
  b=IndependentReviewBoard(); b.submit(Review("r","c","support","A")); self.assertEqual(b.result("c").status,"insufficient_independent_reviews")
 def test_duplicate_reviewer(self):
  b=IndependentReviewBoard(); b.submit(Review("r","c","support","A"))
  with self.assertRaises(ValueError): b.submit(Review("r","c","oppose","B"))
 def test_unknown_evidence(self):
  b=IndependentReviewBoard()
  with self.assertRaises(ValueError): b.submit(Review("r","c","support","A",("x",)))
 def test_wrong_claim_evidence(self):
  b=IndependentReviewBoard(); b.add_evidence(self.ev(claim="other"))
  with self.assertRaises(ValueError): b.submit(Review("r","c1","support","A",("e1",)))
 def test_duplicate_evidence(self):
  b=IndependentReviewBoard(); b.add_evidence(self.ev())
  with self.assertRaises(ValueError): b.add_evidence(self.ev())
 def test_same_provenance_group_counts_once(self):
  self.assertEqual(evidence_independence_count([self.ev("1","same"),self.ev("2","same")]),1)
 def test_independent_groups_count_separately(self):
  self.assertEqual(evidence_independence_count([self.ev("1","a"),self.ev("2","b")]),2)
 def test_tier_and_provenance_propagate(self):
  b=IndependentReviewBoard(); b.add_evidence(self.ev()); b.submit(Review("r1","c1","support","A",("e1",))); b.submit(Review("r2","c1","support","B",("e1",)))
  x=b.result("c1"); self.assertEqual(x.evidence_groups,("paper-A",)); self.assertEqual(x.evidence_tier_max,3)
 def test_counterexample_link(self):
  c=CounterexampleCandidate("c","assumption","scenario","observation","test")
  self.assertTrue(validate_counterexample(c,["c"])); self.assertFalse(validate_counterexample(c,["other"]))
 def test_empty_counterexample_rejected(self):
  with self.assertRaises(ValueError): CounterexampleCandidate("c","","s","o","t")
if __name__=="__main__": unittest.main()
