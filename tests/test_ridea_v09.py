import math
import unittest

import dialectica.ridea_v09 as r


class RideaV09Tests(unittest.TestCase):
    def test_conjugate_posterior_degree_zero_matches_closed_form(self):
        model = r.BayesianPolynomialRegressor(degree=0, noise_std=0.1, prior_variance=1.0)
        ys = [1.0, 2.0, 3.0]
        model.fit([0.0, 1.0, -1.0], ys)
        expected_var = 1.0 / (1.0 + len(ys) / 0.01)
        expected_mean = expected_var * sum(ys) / 0.01
        self.assertAlmostEqual(model.mean[0], expected_mean, places=10)
        self.assertAlmostEqual(model.cov[0][0], expected_var, places=10)

    def test_exact_gaussian_eig_is_nonnegative(self):
        model = r.BayesianPolynomialRegressor(degree=1, noise_std=0.1)
        model.fit([-1.0, 1.0], [-1.0, 1.0])
        self.assertGreater(model.eig(2.0), 0.0)
        self.assertGreaterEqual(model.eig(0.0), 0.0)

    def test_mmd_fast_ratio_matches_direct_for_both_conventions(self):
        history = [-2.0, 0.25, 1.5]
        reference = [-2.0, -1.0, 0.0, 1.0, 2.0]
        candidate = 0.75
        for metric in ("mmd", "mmd2"):
            fast = r.MMDReferenceScorer(history, reference, metric=metric).ratio(candidate)
            direct = r.representativeness_ratio(history, candidate, reference, metric=metric)
            self.assertAlmostEqual(fast, direct, places=10)

    def test_mmd_distance_is_zero_for_identical_empirical_samples(self):
        points = [-1.0, 0.0, 1.0]
        self.assertAlmostEqual(r.mmd2_biased(points, points), 0.0, places=12)

    def test_best_in_class_projection_for_quadratic_and_cubic(self):
        grid = [-4.0 + 8.0 * i / 1000 for i in range(1001)]
        quadratic = r.best_in_class_coefficients(2, 1, grid)
        cubic = r.best_in_class_coefficients(3, 1, grid)
        self.assertAlmostEqual(quadratic[0], -5.0 / 3.0, delta=0.01)
        self.assertAlmostEqual(quadratic[1], 2.0, delta=0.01)
        self.assertAlmostEqual(cubic[0], -5.0 / 3.0, delta=0.01)
        self.assertAlmostEqual(cubic[1], 3.92, delta=0.02)

    def test_proxy_zero_initialization_is_deterministic(self):
        model = r.BayesianPolynomialRegressor(degree=1, noise_std=0.1)
        xs = [-2.0, 0.0, 2.0]
        ys = [r.true_function(x, 2) for x in xs]
        model.fit(xs, ys)
        a = r._adam_proxy(xs, ys, model, tau=2.0, steps=50, init="zeros", proxy_degree=1)
        b = r._adam_proxy(xs, ys, model, tau=2.0, steps=50, init="zeros", proxy_degree=1)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)

    def test_ridea_score_is_ri_times_deamplification_weight(self):
        model = r.BayesianPolynomialRegressor(degree=1, noise_std=0.1)
        xs = [-2.0, 1.0]
        ys = [r.true_function(x, 2) for x in xs]
        model.fit(xs, ys)
        candidates = [-2.0, 0.0, 2.0]
        reference = [-2.0, -1.0, 0.0, 1.0, 2.0]
        proxy = r._adam_proxy(xs, ys, model, tau=2.0, steps=40, init="zeros", proxy_degree=1)
        ri = r.acquisition_scores(model, xs, ys, candidates, reference, "ri", lam=0.5, mmd_metric="mmd2")
        ridea = r.acquisition_scores(model, xs, ys, candidates, reference, "ridea", lam=0.5,
            tau=0.25, kappa=1.0, mmd_metric="mmd2", proxy_steps=40,
            proxy_margin_tau=2.0, proxy_init="zeros", proxy_degree=1)
        for i, x in enumerate(candidates):
            gap = abs(model.predict(x) - r._dot(proxy[:-1], r._features(x, 1)))
            expected = ri[i] * r.sigmoid((gap - 0.25) / 1.0)
            self.assertAlmostEqual(ridea[i], expected, places=10)

    def test_adaptive_threshold_path_is_finite(self):
        model = r.BayesianPolynomialRegressor(degree=1, noise_std=0.1)
        xs = [-2.0, 1.0]
        ys = [r.true_function(x, 2) for x in xs]
        model.fit(xs, ys)
        scores = r.acquisition_scores(model, xs, ys, [-2.0, 0.0, 2.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0], "ridea_code_variant", tau=None,
            proxy_margin_tau=2.0, proxy_init="zeros", proxy_degree=1, mmd_metric="mmd2")
        self.assertTrue(all(math.isfinite(x) for x in scores))

    def test_run_one_reproducible(self):
        a = r.run_one(123, "ridea", true_degree=2, model_degree=1, budget=3,
            tau=0.25, mmd_metric="mmd2", proxy_init="zeros", proxy_degree=1)
        b = r.run_one(123, "ridea", true_degree=2, model_degree=1, budget=3,
            tau=0.25, mmd_metric="mmd2", proxy_init="zeros", proxy_degree=1)
        self.assertEqual(a, b)

    def test_small_benchmark_reproducible(self):
        a = r.run_benchmark(runs=3, seed=77, budget=2, proxy_steps=30)
        b = r.run_benchmark(runs=3, seed=77, budget=2, proxy_steps=30)
        self.assertEqual(a, b)

    def test_expected_observation_mse_adds_noise_variance(self):
        result = r.run_one(99, "bad", true_degree=2, model_degree=1, budget=2, noise_std=0.1)
        self.assertAlmostEqual(result["test_mse_observed_expected"], result["test_mse"] + 0.01)

    def test_candidate_order_and_run_seed_are_reproducible(self):
        a = r.run_one(88, "bad", true_degree=2, model_degree=1, budget=4)
        b = r.run_one(88, "bad", true_degree=2, model_degree=1, budget=4)
        self.assertEqual(a["designs"], b["designs"])

    def test_invalid_mmd_metric_rejected(self):
        with self.assertRaises(ValueError):
            r.mmd_distance([0.0], [0.0], metric="wasserstein")

    def test_invalid_noise_scale_rejected(self):
        with self.assertRaises(ValueError):
            r.BayesianPolynomialRegressor(noise_std=0.0)


if __name__ == "__main__":
    unittest.main()
