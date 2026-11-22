import math
import unittest

from dialectica.v07 import (
    Experiment, add_unknown_hypothesis, bayes_update, evaluate_experiment,
    expected_bayes_error, expected_information_gain,
    expected_value_sample_information, pareto_frontier,
    posterior_predictive_diagnostic, robust_expected_information_gain,
    select_experiment,
)
from dialectica.benchmark_v07 import run_benchmark


class V07Tests(unittest.TestCase):
    def setUp(self):
        self.prior = {"A": .5, "B": .5}
        self.good = Experiment("good", 1.0, {
            "A": {"yes": .9, "no": .1},
            "B": {"yes": .1, "no": .9},
        })
        self.bad = Experiment("bad", 1.0, {
            "A": {"yes": .5, "no": .5},
            "B": {"yes": .5, "no": .5},
        })

    def test_eig_is_positive_for_discriminating_test(self):
        self.assertGreater(expected_information_gain(self.prior, self.good), 0)
        self.assertAlmostEqual(expected_information_gain(self.prior, self.bad), 0.0)

    def test_eig_bounded_by_prior_entropy(self):
        eig = expected_information_gain(self.prior, self.good)
        self.assertLessEqual(eig, math.log(2) + 1e-9)

    def test_bayes_error_decreases_for_informative_test(self):
        self.assertLess(expected_bayes_error(self.prior, self.good),
                        expected_bayes_error(self.prior, self.bad))

    def test_bayes_update(self):
        posterior = bayes_update(self.prior, self.good, "yes")
        self.assertGreater(posterior["A"], posterior["B"])
        self.assertAlmostEqual(sum(posterior.values()), 1.0)

    def test_zero_probability_outcome_rejected(self):
        impossible = Experiment("impossible", 1, {
            "A": {"yes": 1, "no": 0}, "B": {"yes": 1, "no": 0}
        })
        with self.assertRaises(ValueError):
            bayes_update(self.prior, impossible, "no")

    def test_evsi_nonnegative(self):
        utilities = {"choose_A": {"A": 1, "B": 0}, "choose_B": {"A": 0, "B": 1}}
        self.assertGreaterEqual(expected_value_sample_information(self.prior, self.good, utilities), 0)

    def test_evsi_zero_for_uninformative_experiment(self):
        utilities = {"choose_A": {"A": 1, "B": 0}, "choose_B": {"A": 0, "B": 1}}
        self.assertAlmostEqual(expected_value_sample_information(self.prior, self.bad, utilities), 0)

    def test_eig_per_cost_selects_informative_experiment(self):
        chosen = select_experiment(self.prior, [self.bad, self.good], "eig_per_cost")
        self.assertEqual(chosen.experiment_id, "good")

    def test_evsi_requires_utility_table(self):
        with self.assertRaises(ValueError):
            select_experiment(self.prior, [self.good], "evsi_per_cost")

    def test_evsi_per_cost_selects_informative_experiment(self):
        utilities = {"choose_A": {"A": 1, "B": 0}, "choose_B": {"A": 0, "B": 1}}
        chosen = select_experiment(self.prior, [self.bad, self.good],
                                   "evsi_per_cost", utilities=utilities)
        self.assertEqual(chosen.experiment_id, "good")

    def test_simple_mean_separation_requires_means(self):
        with self.assertRaises(ValueError):
            select_experiment(self.prior, [self.good], "simple_mean_separation")

    def test_robust_eig_uses_worst_prior_scenario(self):
        value = robust_expected_information_gain(
            [{"A": .5, "B": .5}, {"A": .9, "B": .1}], self.good)
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, expected_information_gain(self.prior, self.good) + 1e-9)

    def test_robust_eig_selection_requires_scenarios(self):
        with self.assertRaises(ValueError):
            select_experiment(self.prior, [self.good], "robust_eig")

    def test_robust_eig_selection(self):
        chosen = select_experiment(self.prior, [self.bad, self.good], "robust_eig",
            prior_scenarios=[self.prior, {"A": .8, "B": .2}])
        self.assertEqual(chosen.experiment_id, "good")

    def test_predictive_diagnostic_flags_unmodeled_outcome(self):
        result = posterior_predictive_diagnostic(self.prior, self.good, "unexpected", .2)
        self.assertTrue(result.possible_misspecification)
        self.assertGreater(result.surprise_nats, 0)

    def test_predictive_diagnostic_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            posterior_predictive_diagnostic(self.prior, self.good, "yes", 1.0)

    def test_add_unknown_hypothesis(self):
        prior, experiments = add_unknown_hypothesis(
            self.prior, [self.good], "unknown",
            {"good": {"yes": 1/3, "no": 2/3}}, .1)
        self.assertAlmostEqual(prior["unknown"], .1)
        self.assertAlmostEqual(sum(prior.values()), 1)
        self.assertIn("unknown", experiments[0].likelihoods)

    def test_pareto_frontier(self):
        metrics = [evaluate_experiment(self.prior, self.good),
                   evaluate_experiment(self.prior, self.bad)]
        frontier = pareto_frontier(metrics)
        self.assertEqual([m.experiment_id for m in frontier], ["good"])

    def test_invalid_probability_sum(self):
        with self.assertRaises(ValueError):
            Experiment("bad-prob", 1, {"A": {"x": .2, "y": .2}})

    def test_invalid_prior_mismatch(self):
        with self.assertRaises(ValueError):
            expected_information_gain({"A": 1.0}, self.good)

    def test_benchmark_reproducible(self):
        a = run_benchmark(runs=20, budget=2, seed=123)
        b = run_benchmark(runs=20, budget=2, seed=123)
        self.assertEqual(a, b)

    def test_benchmark_has_closed_and_open_world_conditions(self):
        result = run_benchmark(runs=10, budget=2, seed=42)
        self.assertIn("closed_world_model", result["conditions"])
        self.assertIn("explicit_unknown_model", result["conditions"])


if __name__ == "__main__":
    unittest.main()
