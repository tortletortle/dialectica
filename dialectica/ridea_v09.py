"""DIALECTICA V0.9: equation-level R-IDeA reproduction for 1D polynomial regression.

Uses an exact conjugate Bayesian linear-Gaussian posterior and EIG. The true
DGP is only used by the simulator/evaluation and the explicitly labelled oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _features(x: float, degree: int) -> List[float]:
    return [x ** j for j in range(degree + 1)]


def _mat_inv(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    a = [[float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ValueError("singular matrix")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [a[row][j] - factor * a[col][j] for j in range(2 * n)]
    return [row[n:] for row in a]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class BayesianPolynomialRegressor:
    degree: int = 1
    noise_std: float = 0.1
    prior_mean: Optional[Sequence[float]] = None
    prior_variance: float = 1.0

    def __post_init__(self) -> None:
        if self.degree < 0 or self.noise_std <= 0 or self.prior_variance <= 0:
            raise ValueError("degree must be non-negative and scales must be positive")
        d = self.degree + 1
        self._prior_mean = list(self.prior_mean) if self.prior_mean is not None else [0.0] * d
        if len(self._prior_mean) != d:
            raise ValueError("prior_mean length must equal degree + 1")
        self.mean = list(self._prior_mean)
        self.cov = [[self.prior_variance if i == j else 0.0 for j in range(d)] for i in range(d)]

    def fit(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have equal lengths")
        d = self.degree + 1
        inv_noise_var = 1.0 / (self.noise_std ** 2)
        precision = [[(1.0 / self.prior_variance if i == j else 0.0) for j in range(d)] for i in range(d)]
        rhs = [self._prior_mean[i] / self.prior_variance for i in range(d)]
        for x, y in zip(xs, ys):
            phi = _features(float(x), self.degree)
            for i in range(d):
                rhs[i] += phi[i] * float(y) * inv_noise_var
                for j in range(d):
                    precision[i][j] += phi[i] * phi[j] * inv_noise_var
        self.cov = _mat_inv(precision)
        self.mean = _mat_vec(self.cov, rhs)

    def predict(self, x: float) -> float:
        return _dot(self.mean, _features(float(x), self.degree))

    def latent_variance(self, x: float) -> float:
        phi = _features(float(x), self.degree)
        cov_phi = _mat_vec(self.cov, phi)
        return max(0.0, _dot(phi, cov_phi))

    def eig(self, x: float) -> float:
        """Exact Gaussian EIG in nats: 0.5 log(1 + phi' Sigma phi / noise_var)."""
        return 0.5 * math.log1p(self.latent_variance(x) / (self.noise_std ** 2))


def true_function(x: float, degree: int = 2) -> float:
    coefficients = [1.0, 2.0, -0.5]
    if degree >= 3:
        coefficients.append(0.2)
    if degree >= 4:
        coefficients.append(0.0)
    return sum(c * (x ** i) for i, c in enumerate(coefficients))


def rbf(x: float, y: float, sigma: float = 1.0) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-((x - y) ** 2) / (2.0 * sigma * sigma))


def mmd2_biased(points: Sequence[float], reference: Sequence[float], sigma: float = 1.0) -> float:
    """Biased empirical MMD^2 with an RBF kernel (includes diagonal terms)."""
    if not points or not reference:
        raise ValueError("points and reference must be non-empty")
    n, m = len(points), len(reference)
    xx = sum(rbf(x, z, sigma) for x in points for z in points) / (n * n)
    yy = sum(rbf(y, z, sigma) for y in reference for z in reference) / (m * m)
    xy = sum(rbf(x, y, sigma) for x in points for y in reference) / (n * m)
    return max(0.0, xx + yy - 2.0 * xy)


def mmd_distance(points: Sequence[float], reference: Sequence[float], sigma: float = 1.0, metric: str = "mmd") -> float:
    value = mmd2_biased(points, reference, sigma)
    if metric == "mmd2":
        return value
    if metric == "mmd":
        return math.sqrt(value)
    raise ValueError("metric must be 'mmd' or 'mmd2'")


class MMDReferenceScorer:
    """Fast candidate-wise MMD ratios against a fixed reference sample."""
    def __init__(self, history: Sequence[float], reference: Sequence[float], sigma: float = 1.0, metric: str = "mmd"):
        if not history or not reference:
            raise ValueError("history and reference must be non-empty")
        if metric not in {"mmd", "mmd2"}:
            raise ValueError("metric must be 'mmd' or 'mmd2'")
        self.history = list(history)
        self.reference = list(reference)
        self.sigma = sigma
        self.metric = metric
        self.n = len(self.history)
        self.m = len(self.reference)
        self.sum_hh = sum(rbf(x, z, sigma) for x in self.history for z in self.history)
        self.sum_hr = sum(rbf(x, y, sigma) for x in self.history for y in self.reference)
        self.sum_rr = sum(rbf(y, z, sigma) for y in self.reference for z in self.reference)
        self.denominator_mmd2 = max(0.0, self.sum_hh/(self.n*self.n) + self.sum_rr/(self.m*self.m) - 2*self.sum_hr/(self.n*self.m))

    def ratio(self, candidate: float, eps: float = 1e-12) -> float:
        cross_h = sum(rbf(x, candidate, self.sigma) for x in self.history)
        cross_r = sum(rbf(candidate, y, self.sigma) for y in self.reference)
        n1 = self.n + 1
        numerator_mmd2 = max(0.0, (self.sum_hh + 2*cross_h + 1.0)/(n1*n1) + self.sum_rr/(self.m*self.m) - 2*(self.sum_hr + cross_r)/(n1*self.m))
        if self.metric == "mmd":
            numerator = math.sqrt(numerator_mmd2)
            denominator = math.sqrt(self.denominator_mmd2)
        else:
            numerator = numerator_mmd2
            denominator = self.denominator_mmd2
        return numerator / max(denominator, eps)


def representativeness_ratio(
    history: Sequence[float], candidate: float, reference: Sequence[float],
    sigma: float = 1.0, metric: str = "mmd", eps: float = 1e-12,
) -> float:
    if not history:
        raise ValueError("R-I requires at least one initial design point")
    denominator = mmd_distance(history, reference, sigma, metric)
    numerator = mmd_distance(list(history) + [candidate], reference, sigma, metric)
    return numerator / max(denominator, eps)


def _adam_proxy(
    history: Sequence[float], ys: Sequence[float], f_model: BayesianPolynomialRegressor,
    tau: float, alpha: float = 1.0, steps: int = 200, lr: float = 0.01,
    init: str = "zeros", proxy_degree: int = 1,
) -> Tuple[float, ...]:
    """Fit proxy g in a configured polynomial basis using paper Eq. (11).

    L(g) = mean((g(x_i)-y_i)^2) + alpha*mean(max(0,tau-|fhat(x_i)-g(x_i)|)).
    The supplementary regression code uses a zero-initialized linear g even
    when the fitted model is quadratic; proxy_degree=1 reproduces that choice.
    """
    if not history:
        return tuple(f_model.mean[:proxy_degree + 1])
    if proxy_degree < 0 or tau < 0 or alpha < 0 or steps < 1 or lr <= 0:
        raise ValueError("invalid proxy optimizer settings")
    f_values = [f_model.predict(x) for x in history]
    if init == "current":
        theta = list(f_model.mean[:proxy_degree + 1])
        if len(theta) < proxy_degree + 1:
            theta.extend([0.0] * (proxy_degree + 1 - len(theta)))
    elif init == "zeros":
        theta = [0.0] * (proxy_degree + 1)
    elif init == "jitter":
        base = list(f_model.mean[:proxy_degree + 1])
        if len(base) < proxy_degree + 1:
            base.extend([0.0] * (proxy_degree + 1 - len(base)))
        theta = [value + (0.05 if i % 2 == 0 else -0.05) for i, value in enumerate(base)]
    else:
        raise ValueError("proxy init must be current, zeros, or jitter")
    d = len(theta)
    m = [0.0] * d
    v = [0.0] * d
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    n = len(history)
    for step in range(1, steps + 1):
        grad = [0.0] * d
        for x, y, fval in zip(history, ys, f_values):
            phi = _features(x, proxy_degree)
            gval = _dot(theta, phi)
            residual = gval - y
            gap = fval - gval
            active_sign = (1.0 if gap > 0 else (-1.0 if gap < 0 else 0.0)) if abs(gap) < tau else 0.0
            for j in range(d):
                grad[j] += (2.0 * residual * phi[j] + alpha * active_sign * phi[j]) / n
        for j in range(d):
            m[j] = beta1 * m[j] + (1 - beta1) * grad[j]
            v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j]
            m_hat = m[j] / (1 - beta1 ** step)
            v_hat = v[j] / (1 - beta2 ** step)
            theta[j] -= lr * m_hat / (math.sqrt(v_hat) + adam_eps)
    loss = 0.0
    for x, y, fval in zip(history, ys, f_values):
        gval = _dot(theta, _features(x, proxy_degree))
        loss += (gval - y) ** 2 + alpha * max(0.0, tau - abs(fval - gval))
    return tuple(theta) + (loss / n,)


def sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-min(z, 700.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def best_in_class_coefficients(true_degree: int, model_degree: int, test_grid: Sequence[float]) -> Tuple[float, ...]:
    """L2 projection of the true DGP onto the learner class under test distribution."""
    d = model_degree + 1
    gram = [[0.0] * d for _ in range(d)]
    rhs = [0.0] * d
    for x in test_grid:
        phi = _features(x, model_degree)
        y = true_function(x, true_degree)
        for i in range(d):
            rhs[i] += phi[i] * y
            for j in range(d):
                gram[i][j] += phi[i] * phi[j]
    return tuple(_mat_vec(_mat_inv(gram), rhs))


def acquisition_scores(
    model: BayesianPolynomialRegressor, history: Sequence[float], ys: Sequence[float],
    candidates: Sequence[float], reference: Sequence[float], method: str,
    lam: float = 0.5, tau: Optional[float] = 0.5, kappa: float = 1.0,
    mmd_metric: str = "mmd2", proxy_steps: int = 200, proxy_alpha: float = 1.0,
    proxy_margin_tau: Optional[float] = 2.0, proxy_init: str = "zeros", proxy_degree: int = 1,
    oracle_g: Optional[Tuple[float, ...]] = None,
) -> List[float]:
    if method not in {"bad", "ri", "ridea", "ridea_current_init", "ridea_oracle", "ridea_code_variant"}:
        raise ValueError("unknown acquisition method")
    if lam < 0 or kappa <= 0:
        raise ValueError("lambda must be non-negative and kappa positive")
    proxy = None
    mmd_scorer = MMDReferenceScorer(history, reference, metric=mmd_metric) if method != "bad" else None
    if method in {"ridea", "ridea_current_init", "ridea_code_variant"}:
        fit_tau = proxy_margin_tau if proxy_margin_tau is not None else (tau if tau is not None else 2.0)
        effective_init = "current" if method == "ridea_current_init" else proxy_init
        proxy = _adam_proxy(history, ys, model, tau=fit_tau, alpha=proxy_alpha, steps=proxy_steps, init=effective_init, proxy_degree=proxy_degree)
    if method == "ridea_oracle":
        if oracle_g is None:
            raise ValueError("ridea_oracle requires best-in-class coefficients")
        proxy = oracle_g
    scores = []
    dea_tau = tau
    if method in {"ridea", "ridea_current_init", "ridea_oracle", "ridea_code_variant"} and dea_tau is None:
        gaps = [abs(model.predict(x) - _dot(proxy[:-1] if method in {"ridea", "ridea_current_init", "ridea_code_variant"} else proxy, _features(x, proxy_degree if method != "ridea_oracle" else model.degree))) for x in candidates]
        dea_tau = _quantile(gaps, 0.8)
    for x in candidates:
        eig = model.eig(x)
        if method == "bad":
            scores.append(eig)
            continue
        ratio = mmd_scorer.ratio(x)
        ri_score = eig * (1.0 - lam * ratio)
        if method == "ri":
            scores.append(ri_score)
            continue
        assert proxy is not None
        gap = abs(model.predict(x) - _dot(proxy[:-1] if method in {"ridea", "ridea_current_init", "ridea_code_variant"} else proxy, _features(x, proxy_degree if method != "ridea_oracle" else model.degree)))
        dea = sigmoid((gap - float(dea_tau)) / kappa)
        scores.append(ri_score * dea)
    return scores


def run_one(
    seed: int, method: str, true_degree: int = 2, model_degree: int = 1,
    budget: int = 5, candidate_count: int = 60, test_count: int = 100,
    noise_std: float = 0.1, lam: float = 0.5, tau: Optional[float] = 0.5,
    kappa: float = 1.0, mmd_metric: str = "mmd2", proxy_steps: int = 200, proxy_alpha: float = 1.0,
    proxy_margin_tau: Optional[float] = 2.0, proxy_init: str = "zeros", proxy_degree: int = 1,
) -> Dict[str, object]:
    if budget < 1 or candidate_count < 3 or test_count < 3:
        raise ValueError("budget, candidate_count, and test_count must be positive")
    init_rng = random.Random(seed + 101)
    noise_rng = random.Random(seed + 202)
    design_rng = random.Random(seed + 303)
    initial_grid = [-4.0 + 0.1 * i for i in range(81)]
    history = [init_rng.choice(initial_grid)]
    ys = [true_function(history[0], true_degree) + noise_rng.gauss(0.0, noise_std)]
    model = BayesianPolynomialRegressor(model_degree, noise_std)
    model.fit(history, ys)
    candidates = [-4.0 + 8.0 * i / (candidate_count - 1) for i in range(candidate_count)]
    test_grid = [-4.0 + 8.0 * i / (test_count - 1) for i in range(test_count)]
    oracle = best_in_class_coefficients(true_degree, model_degree, test_grid)
    trace = []
    for round_idx in range(budget):
        if method == "random":
            # The supplementary polynomial runner samples the 0.1-spaced design grid.
            x = design_rng.choice(initial_grid)
            chosen_score = None
        else:
            # The supplementary implementation randomly permutes the candidate
            # grid before argmax; this matters when acquisition scores tie.
            candidate_order = list(range(len(candidates)))
            design_rng.shuffle(candidate_order)
            shuffled_candidates = [candidates[i] for i in candidate_order]
            scores = acquisition_scores(
                model, history, ys, shuffled_candidates, test_grid, method,
                lam=lam, tau=tau, kappa=kappa, mmd_metric=mmd_metric,
                proxy_steps=proxy_steps, proxy_alpha=proxy_alpha, proxy_margin_tau=proxy_margin_tau, proxy_init=proxy_init, proxy_degree=proxy_degree, oracle_g=oracle,
            )
            best_idx = max(range(len(shuffled_candidates)), key=lambda i: scores[i])
            x, chosen_score = shuffled_candidates[best_idx], scores[best_idx]
        y = true_function(x, true_degree) + noise_rng.gauss(0.0, noise_std)
        history.append(x)
        ys.append(y)
        model.fit(history, ys)
        trace.append({"x": x, "score": chosen_score})
    prediction_errors = [(model.predict(x) - true_function(x, true_degree)) ** 2 for x in test_grid]
    test_mse = sum(prediction_errors) / len(prediction_errors)
    final_mmd2 = mmd2_biased(history, test_grid)
    return {
        "seed": seed, "method": method, "test_mse": test_mse,
        "test_mse_observed_expected": test_mse + noise_std ** 2,
        "final_mmd2": final_mmd2, "final_mmd": math.sqrt(final_mmd2),
        "designs": history, "posterior_mean": model.mean,
        "trace": trace, "oracle_linear": oracle,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take quantile of empty data")
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _bootstrap_ci(values: Sequence[float], seed: int, reps: int = 1000) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return _quantile(means, 0.025), _quantile(means, 0.975)


def run_benchmark(
    runs: int = 20, seed: int = 20261123, budget: int = 9,
    lam: float = 0.5, tau: Optional[float] = 0.5, kappa: float = 1.0,
    mmd_metric: str = "mmd2", proxy_steps: int = 200, proxy_alpha: float = 1.0,
    proxy_margin_tau: Optional[float] = 2.0, proxy_init: str = "zeros", proxy_degree: int = 1, test_count: int = 100,
) -> Dict[str, object]:
    if runs < 2:
        raise ValueError("at least two runs are required")
    conditions = {
        "misspecified_quadratic": {"true_degree": 2, "model_degree": 1},
        "well_specified_quadratic": {"true_degree": 2, "model_degree": 2},
        "misspecified_cubic": {"true_degree": 3, "model_degree": 1},
    }
    methods = ["random", "bad", "ri", "ridea", "ridea_current_init", "ridea_oracle"]
    report: Dict[str, object] = {
        "metadata": {
            "runs": runs, "seed": seed, "budget_after_initial_point": budget,
            "total_designs_including_initial": budget + 1,
            "candidate_count": 60, "test_grid_count": test_count, "noise_std": 0.1,
            "lambda": lam, "tau": tau, "kappa": kappa,
            "mmd_metric_for_acquisition": mmd_metric,
            "mmd_estimator": "biased empirical RBF MMD^2; sqrt applied when metric='mmd'",
            "kernel": "RBF sigma=1.0", "proxy_loss": "mean squared fit error + alpha*mean hinge disagreement",
            "proxy_alpha": proxy_alpha, "proxy_margin_tau": proxy_margin_tau,
            "proxy_init": proxy_init, "proxy_degree": proxy_degree,
            "note": "Finite-grid reproduction of the supplementary fixedmodel path: zero-initialized linear proxy, 200 Adam steps at 0.01, proxy margin threshold 2, RBF sigma 1, and squared MMD ratio. Posterior and EIG are computed exactly by Gaussian conjugacy rather than the authors' Pyro Monte Carlo. Oracle uses the best-in-class L2 projection under the test distribution, not the true DGP function.",
        },
        "conditions": {},
    }
    for condition, config in conditions.items():
        raw: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}
        for run_idx in range(runs):
            run_seed = seed + run_idx * 1009
            for method in methods:
                raw[method].append(run_one(
                    run_seed, method, budget=budget, lam=lam, tau=tau,
                    kappa=kappa, mmd_metric=mmd_metric, proxy_steps=proxy_steps, proxy_alpha=proxy_alpha,
                    proxy_margin_tau=proxy_margin_tau, proxy_init=proxy_init, proxy_degree=proxy_degree, test_count=test_count, **config,
                ))
        rows = {}
        for method in methods:
            mses = [float(r["test_mse"]) for r in raw[method]]
            mmds = [float(r["final_mmd"]) for r in raw[method]]
            rows[method] = {
                "mean_test_mse_function": sum(mses) / runs,
                "mean_test_mse_observed_expected": sum(float(r["test_mse_observed_expected"]) for r in raw[method]) / runs,
                "test_mse_function_ci95_bootstrap": _bootstrap_ci(mses, seed + sum((i + 1) * ord(ch) for i, ch in enumerate(condition + method))),
                "mean_final_mmd2": sum(float(r["final_mmd2"]) for r in raw[method]) / runs,
                "mean_final_mmd": sum(mmds) / runs,
            }
        paired = {}
        base = [float(r["test_mse"]) for r in raw["bad"]]
        for method in ["random", "ri", "ridea", "ridea_current_init", "ridea_oracle"]:
            diffs = [float(raw[method][i]["test_mse"]) - base[i] for i in range(runs)]
            paired[method + "_minus_bad"] = {
                "mean_difference": sum(diffs) / runs,
                "paired_bootstrap_ci95": _bootstrap_ci(diffs, seed + 991 + len(method)),
                "fraction_better_than_bad": sum(1 for d in diffs if d < 0) / runs,
            }
        ri_values = [float(r["test_mse"]) for r in raw["ri"]]
        for method in ["ridea", "ridea_current_init", "ridea_oracle"]:
            diffs = [float(raw[method][i]["test_mse"]) - ri_values[i] for i in range(runs)]
            paired[method + "_minus_ri"] = {
                "mean_difference": sum(diffs) / runs,
                "paired_bootstrap_ci95": _bootstrap_ci(diffs, seed + 1991 + len(method)),
                "fraction_better_than_ri": sum(1 for d in diffs if d < 0) / runs,
            }
        report["conditions"][condition] = {"strategies": rows, "paired_comparisons": paired}
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
