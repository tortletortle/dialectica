import unittest

from dialectica.v08 import (
    ModelPairExperiment,
    evaluate_robust_information,
    expected_generalized_information_gain,
    expected_information_gain_for_model,
    likelihood_error_penalty,
    posterior_error_penalty,
    robust_kl_information,
    select_robust_experiment,
    true_eig,
    verify_egig_decomposition,
    working_eig,
)
from dialectica.benchmark_v08 import run_benchmark


class V08Tests(unittest.TestCase):
    def setUp(self):
        self.prior = {"A": .5, "B": .5}
        self.correct = ModelPairExperiment("correct", 1.0,
            {"A": {"yes": .85, "no": .15}, "B": {"yes": .15, "no": .85}},
            {"A": {"yes": .85, "no": .15}, "B": {"yes": .15, "no": .85}})
        self.overconfident = ModelPairExperiment("overconfident", 1.0,
            {"A": {"yes": .60, "no": .40}, "B": {"yes": .40, "no": .60}},
            {"A": {"yes": .99, "no": .01}, "B": {"yes": .01, "no": .99}})
        self.spurious = ModelPairExperiment("spurious", .8,
            {"A": {"yes": .50, "no": .50}, "B": {"yes": .50, "no": .50}},
            {"A": {"yes": .90, "no": .10}, "B": {"yes": .10, "no": .90}})

    def test_matching_models_have_zero_posterior_penalty(self):
        self.assertAlmostEqual(posterior_error_penalty(self.prior, self.correct), 0.0)
        self.assertAlmostEqual(likelihood_error_penalty(self.prior, self.correct), 0.0)

    def test_matching_models_egig_equals_eig(self):
        self.assertAlmostEqual(
            expected_generalized_information_gain(self.prior, self.correct),
            true_eig(self.prior, self.correct), places=10)

    def test_egig_decomposition_identity(self):
        self.assertTrue(verify_egig_decomposition(self.prior, self.overconfident))
        self.assertTrue(verify_egig_decomposition(self.prior, self.spurious))

    def test_overconfident_wrong_model_has_negative_egig(self):
        self.assertGreater(working_eig(self.prior, self.overconfident), 0.5)
        self.assertLess(expected_generalized_information_gain(self.prior, self.overconfident), 0.0)

    def test_spurious_signal_is_penalized(self):
        self.assertAlmostEqual(true_eig(self.prior, self.spurious), 0.0, places=10)
        self.assertGreater(working_eig(self.prior, self.spurious), 0.2)
        self.assertLess(expected_generalized_information_gain(self.prior, self.spurious), 0.0)

    def test_egig_can_be_negative(self):
        self.assertLess(expected_generalized_information_gain(self.prior, self.overconfident), 0)

    def test_robust_kl_information_decomposition(self):
        value = robust_kl_information(self.prior, self.overconfident)
        metrics = evaluate_robust_information(self.prior, self.overconfident)
        self.assertAlmostEqual(value, metrics.true_eig - metrics.likelihood_error_penalty)

    def test_robust_kl_not_greater_than_true_eig(self):
        self.assertLessEqual(robust_kl_information(self.prior, self.overconfident),
                             true_eig(self.prior, self.overconfident) + 1e-10)

    def test_egig_prefers_correctly_specified_candidate(self):
        chosen = select_robust_experiment(
            self.prior, [self.overconfident, self.spurious, self.correct], "egig")
        self.assertEqual(chosen.experiment_id, "correct")

    def test_working_eig_prefers_overconfident_candidate(self):
        chosen = select_robust_experiment(
            self.prior, [self.overconfident, self.spurious, self.correct], "working_eig")
        self.assertEqual(chosen.experiment_id, "overconfident")

    def test_oracle_true_eig_prefers_correct_signal(self):
        chosen = select_robust_experiment(
            self.prior, [self.overconfident, self.spurious, self.correct], "true_eig_oracle")
        self.assertEqual(chosen.experiment_id, "correct")

    def test_random_selection_reproducible(self):
        import random
        a = select_robust_experiment(self.prior, [self.correct, self.spurious],
                                     "random", random.Random(17))
        b = select_robust_experiment(self.prior, [self.correct, self.spurious],
                                     "random", random.Random(17))
        self.assertEqual(a.experiment_id, b.experiment_id)

    def test_support_mismatch_rejected_for_egig(self):
        bad = ModelPairExperiment("support", 1.0,
            {"A": {"yes": 1.0, "no": 0.0}, "B": {"yes": 0.0, "no": 1.0}},
            {"A": {"yes": 0.0, "no": 1.0}, "B": {"yes": 0.0, "no": 1.0}})
        with self.assertRaises(ValueError):
            expected_generalized_information_gain(self.prior, bad)

    def test_invalid_probability_sum_rejected(self):
        with self.assertRaises(ValueError):
            ModelPairExperiment("bad", 1.0,
                {"A": {"yes": .2, "no": .2}, "B": {"yes": .5, "no": .5}},
                {"A": {"yes": .5, "no": .5}, "B": {"yes": .5, "no": .5}})

    def test_different_outcome_labels_rejected(self):
        with self.assertRaises(ValueError):
            ModelPairExperiment("bad", 1.0,
                {"A": {"yes": .5, "no": .5}, "B": {"yes": .5, "no": .5}},
                {"A": {"up": .5, "down": .5}, "B": {"up": .5, "down": .5}})

    def test_prior_hypothesis_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            true_eig({"A": 1.0}, self.correct)

    def test_invalid_cost_rejected(self):
        with self.assertRaises(ValueError):
            ModelPairExperiment("bad", 0,
                {"A": {"yes": 1}, "B": {"yes": 1}},
                {"A": {"yes": 1}, "B": {"yes": 1}})

    def test_benchmark_reproducible(self):
        self.assertEqual(run_benchmark(runs=20, seed=123),
                         run_benchmark(runs=20, seed=123))

    def test_benchmark_exposes_misspecification_failure(self):
        result = run_benchmark(runs=100, seed=42)
        rows = result["strategies"]
        self.assertLess(rows["working_eig"]["accuracy"],
                        rows["egig"]["accuracy"])
        self.assertGreater(rows["working_eig"]["mean_posterior_kl"],
                           rows["egig"]["mean_posterior_kl"])


if __name__ == "__main__":
    unittest.main()
