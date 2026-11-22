import unittest
from dialectica.v04 import (AbstentionPolicy, CorrelatedFailureMonitor,
                            DriftAwareBelief, wilson_lower_bound)

class V04Tests(unittest.TestCase):
    def test_discounted_evidence_moves_toward_prior(self):
        b = DriftAwareBelief(decay=.5, window_size=20, min_window=8)
        for _ in range(8): b.update(True)
        before = b.mean
        b.update(False)
        self.assertLess(b.mean, before)

    def test_detects_sustained_performance_shift(self):
        b = DriftAwareBelief(window_size=8, min_window=8, drift_threshold=.5)
        for _ in range(8): b.update(True)
        # Clear the first window by using a fresh monitor of the new regime.
        b = DriftAwareBelief(window_size=8, min_window=8, drift_threshold=.5)
        for _ in range(8): b.update(False)
        for _ in range(4): b.update(True)
        self.assertGreaterEqual(b.drift_events, 1)

    def test_transfer_is_weak_prior(self):
        b = DriftAwareBelief.transferred(.9, strength=2)
        self.assertGreater(b.mean, .5)
        self.assertLess(b.effective_n, 1e-9)

    def test_wilson_no_data(self):
        self.assertEqual(wilson_lower_bound(0, 0), 0)

    def test_abstains_on_insufficient_data(self):
        self.assertTrue(AbstentionPolicy(min_trials=8).should_abstain(7, 7))

    def test_abstains_on_low_confidence(self):
        self.assertTrue(AbstentionPolicy(min_trials=8).should_abstain(4, 10))

    def test_accepts_high_confidence(self):
        self.assertFalse(AbstentionPolicy(min_trials=8).should_abstain(19, 20))

    def test_correlated_failures(self):
        m = CorrelatedFailureMonitor()
        m.observe({"a": False, "b": False, "c": True})
        m.observe({"a": False, "b": False, "c": True})
        self.assertEqual(m.joint_failure_rate("a", "b"), 1.0)
        self.assertEqual(m.penalty("a", ["b"]), 1.0)

    def test_invalid_outcome_type(self):
        m = CorrelatedFailureMonitor()
        with self.assertRaises(TypeError): m.observe({"a": 1})

    def test_invalid_binomial_counts(self):
        with self.assertRaises(ValueError): wilson_lower_bound(5, 4)

if __name__ == "__main__":
    unittest.main()
