"""Reproducible command-line benchmark for DIALECTICA V0.9 R-IDeA."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .ridea_v09 import run_benchmark


def write_csv(report: dict, path: str | Path) -> None:
    fields = [
        "condition", "strategy", "mean_test_mse_function",
        "mean_test_mse_observed_expected", "ci95_low", "ci95_high",
        "mean_final_mmd2", "mean_final_mmd",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, data in report["conditions"].items():
            for strategy, row in data["strategies"].items():
                low, high = row["test_mse_function_ci95_bootstrap"]
                writer.writerow({
                    "condition": condition,
                    "strategy": strategy,
                    "mean_test_mse_function": row["mean_test_mse_function"],
                    "mean_test_mse_observed_expected": row["mean_test_mse_observed_expected"],
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_final_mmd2": row["mean_final_mmd2"],
                    "mean_final_mmd": row["mean_final_mmd"],
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--budget", type=int, default=9, help="adaptive selections after one random initial design")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--mmd-metric", choices=["mmd", "mmd2"], default="mmd2")
    parser.add_argument("--proxy-margin-tau", type=float, default=2.0)
    parser.add_argument("--proxy-alpha", type=float, default=1.0)
    parser.add_argument("--proxy-init", choices=["zeros", "current", "jitter"], default="zeros")
    parser.add_argument("--proxy-degree", type=int, default=1)
    parser.add_argument("--proxy-steps", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--json-out")
    parser.add_argument("--csv-out")
    args = parser.parse_args()
    report = run_benchmark(
        runs=args.runs, seed=args.seed, budget=args.budget, lam=args.lam,
        tau=args.tau, kappa=args.kappa, mmd_metric=args.mmd_metric,
        proxy_steps=args.proxy_steps, proxy_alpha=args.proxy_alpha,
        proxy_margin_tau=args.proxy_margin_tau, proxy_init=args.proxy_init,
        proxy_degree=args.proxy_degree, test_count=args.test_count,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(report, args.csv_out)
    print(payload)


if __name__ == "__main__":
    main()
