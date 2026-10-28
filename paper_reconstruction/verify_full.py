"""Independent numerical comparison against the archived paper results.

Runs the original synthetic study design; does NOT make new data claims.
Use OPENBLAS_NUM_THREADS=1 to reduce resource contention and numerical drift.
"""
import argparse
import json
from pathlib import Path

from experiment import (make_nodes, run_one_seed, summarize,
                        sensitivity, self_check)

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "archived_results"
METRICS = ("f1", "recall", "specificity", "balanced_accuracy",
           "false_positive_rate", "false_negative_rate", "roc_auc", "pr_auc",
           "threshold")


def compare(actual, expected, identifiers, metrics=METRICS, tolerance=1e-9):
    key = lambda r: tuple(r[item] for item in identifiers)
    a = {key(r): r for r in actual}
    b = {key(r): r for r in expected}
    assert len(a) == len(actual)
    assert len(b) == len(expected)
    assert a.keys() == b.keys(), (
        "Different records: missing=%s, extra=%s" %
        (b.keys() - a.keys(), a.keys() - b.keys())
    )
    largest = 0.0
    for idx, produced in a.items():
        frozen = b[idx]
        for field in metrics:
            delta = abs(float(produced[field]) - float(frozen[field]))
            largest = max(largest, delta)
            assert delta < tolerance, (
                "Mismatch %s field %s: produced %.14g, archived %.14g" %
                (idx, field, produced[field], frozen[field])
            )
    return largest


def read(name):
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def verify_logistic():
    frozen = read("multiseed_results.json")
    seeds = sorted({r["seed"] for r in frozen})
    output = []
    for seed in seeds:
        produced = run_one_seed(seed)
        original = [r for r in frozen if r["seed"] == seed]
        delta = compare(
            produced, original,
            ("seed", "model_name", "paradigm", "setting", "policy")
        )
        print("PASS seed %s: %s policies; max absolute difference %.3g" %
              (seed, len(produced), delta), flush=True)
        output.extend(produced)

    assert len(output) == 240
    summaries = summarize(output)
    original_summaries = read("multiseed_summary.json")
    summary_metrics = [field + suffix for field in METRICS
                       for suffix in ("_mean", "_std")]
    delta = compare(
        summaries, original_summaries,
        ("model_name", "paradigm", "setting", "policy"),
        metrics=summary_metrics,
    )
    print("PASS all 240 records and 24 summary groups; max difference %.3g"
          % delta)


def verify_sensitivity():
    produced = sensitivity(make_nodes(0))
    original = read("source_weight_sensitivity.json")
    delta = compare(produced, original, ("scenario", "policy"))
    print("PASS all 12 source-weight sensitivity observations; max difference %.3g"
          % delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Rerun all ten LR seeds (CPU-intensive)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Rerun four source-weight scenarios")
    options = parser.parse_args()
    self_check()
    print("PASS split diagnostics and validation isolation")
    if options.full:
        verify_logistic()
    if options.sensitivity:
        verify_sensitivity()


if __name__ == "__main__":
    main()
