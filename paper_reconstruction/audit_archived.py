"""Read-only integrity check of supplied archived paper results (no training)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent / "archived_results"
EXPECTED = {
    "classic FedAvg": 0.5570925150993518,
    "source-weighted FedAvg": 0.5554283851973161,
    "all data in one server": 0.5476420597257399,
}


def load(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def main():
    rows = load("multiseed_results.json")
    summaries = load("multiseed_summary.json")
    sensitivity = load("source_weight_sensitivity.json")
    diagnostics = load("node_non_iid_diagnostics.json")
    assert len(rows) == 240, f"Expected 240 records, found {len(rows)}"
    assert len(summaries) == 24
    assert len(sensitivity) == 12
    assert len(diagnostics) == 5
    assert set(r["seed"] for r in rows) == set(range(11, 111, 11))
    keys = [(r["seed"], r["setting"], r["policy"]) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicated seed/setting/policy records"
    assert all(r["policy"] in ("best_f1", "best_balanced", "best_operational")
               for r in rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["paradigm"],
                 row["setting"], row["policy"])].append(row)
    assert len(grouped) == len(summaries), "Inconsistent summary groups"
    for summary in summaries:
        key = (summary["model_name"], summary["paradigm"],
               summary["setting"], summary["policy"])
        vals = grouped[key]
        assert len(vals) == 10
        calc_f1 = sum(row["f1"] for row in vals) / len(vals)
        assert abs(calc_f1 - summary["f1_mean"]) < 1e-10, key
        if summary["policy"] == "best_f1" and summary["setting"] in EXPECTED:
            target = EXPECTED[summary["setting"]]
            assert abs(calc_f1 - target) < 1e-10, key
            print(summary["setting"], f"F1={calc_f1:.9f}")

    ntrain = sum(n["train_samples"] for n in diagnostics)
    assert ntrain == 126006, ntrain
    assert ntrain * 11 == 1386066
    print("Archived results internally consistent; 10 seeds and 126006 training records.")
    print("This check does NOT establish which code generated the results.")


if __name__ == "__main__":
    main()
