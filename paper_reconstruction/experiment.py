"""Paper-aligned reconstruction of the synthetic, single-level FL benchmark.

WARNING: A reconstructed protocol is not the original execution provenance.
This module never overwrites the historical outputs/ folder.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    FEATURES, NODE_GENERATION, PAPER_LOGISTIC_CONFIG, PAPER_MULTI_SEED_VALUES,
    PAPER_NEURAL_NETWORK_CONFIG, PAPER_NODE_SPECS, SOURCE_WEIGHT_SCENARIOS,
    SYNTHETIC_BASE_LOGIT, SYNTHETIC_INTERACTIONS, SYNTHETIC_LOGIT_COEFFS,
    TEST_SIZE, VAL_SIZE,
)
from model import copy_params, init_model_params, predict_proba_model, train_model

# Node-specific beta shape parameters, Gaussian dependency for rainfall_3h,
# and categorical territorial risk. Inspired by the archived generator.
PROFILES = {
    "national": ((2.1, 5.2), (0.10, 0.08), (2.8, 3.2), (2.3, 2.9),
                 (2.1, 4.8), (2.7, 2.4), (1.3, 7.5), (1.2, 9.0),
                 (2.2, 4.5), (0.40, 0.42, 0.18)),
    "regional": ((2.0, 5.0), (0.10, 0.10), (2.5, 3.5), (2.0, 3.0),
                 (1.8, 5.0), (2.5, 2.5), (1.2, 8.0), (1.1, 10.0),
                 (1.3, 7.0), (0.45, 0.40, 0.15)),
    "local": ((1.8, 5.5), (0.08, 0.15), (2.0, 4.0), (1.8, 3.8),
              (1.5, 4.5), (2.0, 3.0), (2.5, 4.0), (2.0, 5.0),
              (1.2, 8.0), (0.55, 0.30, 0.15)),
    "sensor": ((1.7, 5.5), (0.07, 0.12), (2.0, 4.2), (2.4, 3.0),
               (2.0, 4.2), (2.3, 3.0), (1.1, 9.0), (1.1, 10.0),
               (2.8, 3.8), (0.50, 0.35, 0.15)),
    "citizen": ((1.6, 5.8), (0.06, 0.18), (1.8, 4.6), (1.6, 4.0),
                (1.4, 4.8), (1.8, 3.4), (3.2, 3.6), (2.6, 4.2),
                (1.1, 8.5), (0.60, 0.28, 0.12)),
}


@dataclass
class Node:
    name: str
    node_type: str
    source_weight: float
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    prevalence: float
    label_noise: float


def generate_synthetic(spec: dict, offset: int = 0):
    """Generate bounded, intentionally non-IID synthetic observations."""
    t = spec["node_type"]
    shape = PROFILES[t]
    rng = np.random.default_rng(int(spec["seed"]) + int(offset))
    n = int(spec["n_samples"])
    one_hour = rng.beta(*shape[0], n)
    three_hours = np.clip(one_hour + rng.normal(*shape[1], n), 0.0, 1.0)
    other = [rng.beta(*ab, n) for ab in shape[2:9]]
    risk = rng.choice([0.2, 0.5, 0.8], n, p=shape[9])
    X = np.column_stack([one_hour, three_hours, *other, risk])
    assert X.shape == (n, len(FEATURES))

    coefficients = np.array([SYNTHETIC_LOGIT_COEFFS[f] for f in FEATURES])
    logits = SYNTHETIC_BASE_LOGIT + X @ coefficients
    for a, b, coefficient in SYNTHETIC_INTERACTIONS:
        logits += coefficient * X[:, FEATURES.index(a)] * X[:, FEATURES.index(b)]
    logits += NODE_GENERATION[t]["node_bias"]
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    y = rng.binomial(1, prob).astype(int)
    flip = float(NODE_GENERATION[t]["label_noise"])
    mask = rng.random(n) < flip
    y[mask] = 1 - y[mask]
    return X.astype(float), y, flip


def stratified_split(X: np.ndarray, y: np.ndarray, seed: int):
    """Classwise stratified partition recovered from archived node diagnostics.

    Generate a per-node RNG using the same seed used for synthetic generation,
    shuffle each class (0 then 1) independently, and use FLOOR classwise
    sample counts: 25% test, 15% validation, the remainder training.
    Unlike rounding global 60/15/25 counts, this yields the historical
    126,006 training records across all five reference nodes.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = int(len(idx) * TEST_SIZE)
        n_val = int(len(idx) * VAL_SIZE)
        test.extend(idx[:n_test])
        val.extend(idx[n_test:n_test+n_val])
        train.extend(idx[n_test+n_val:])
    partitions = [np.asarray(p, dtype=int) for p in (train, val, test)]
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert sum(map(len, partitions)) == len(y)
    return [(X[idx], y[idx]) for idx in partitions]


def make_nodes(seed_offset: int = 0, sample_cap: int | None = None):
    nodes = []
    for spec in PAPER_NODE_SPECS:
        spec = dict(spec)
        if sample_cap is not None:
            spec["n_samples"] = min(sample_cap, spec["n_samples"])
        X, y, flip = generate_synthetic(spec, seed_offset)
        (Xt, yt), (Xv, yv), (Xs, ys) = stratified_split(
            X, y, seed=int(spec["seed"]) + int(seed_offset)
        )
        nodes.append(Node(
            name=spec["name"], node_type=spec["node_type"],
            source_weight=float(spec["source_weight"]),
            X_train=Xt, y_train=yt, X_val=Xv, y_val=yv,
            X_test=Xs, y_test=ys, prevalence=float(np.mean(y)),
            label_noise=flip,
        ))
    return nodes


def aggregate_weights(nodes, weighted: bool):
    a = np.array([len(n.y_train) * (n.source_weight if weighted else 1.0)
                  for n in nodes], dtype=float)
    if np.any(a < 0) or not np.any(a > 0):
        raise ValueError("Invalid source weights")
    return a / a.sum()


def train_federated(nodes, cfg, weighted: bool, record_history: bool = False):
    """Single synchronous orchestrator, not regional-national hierarchy."""
    global_params = init_model_params(cfg, len(FEATURES))
    weights = aggregate_weights(nodes, weighted)
    rounds = int(cfg["federated_rounds"])
    history = []
    for t in range(rounds):
        updates = []
        for node in nodes:
            local, _ = train_model(
                node.X_train, node.y_train, cfg,
                params_init=copy_params(global_params),
                epochs=int(cfg["local_epochs"]),
            )
            updates.append(local)
        if cfg["kind"] == "logistic":
            global_params = sum(float(w) * x for w, x in zip(weights, updates))
        else:
            # Keras models are represented as lists of trainable arrays.
            arrays = []
            for i in range(len(updates[0]["weights"])):
                arrays.append(sum(float(w) * update["weights"][i]
                                  for w, update in zip(weights, updates)))
            global_params = copy_params(updates[0])
            global_params["weights"] = arrays
        if record_history:
            history.append({"round": t + 1, "aggregation": "source_weighted" if weighted else "classic"})
    return global_params, history


def pooled_predictions(params, nodes, kind: str, split: str):
    assert split in ("val", "test"), "thresholds must not use training/test ambiguously"
    labels, probs = [], []
    for n in nodes:
        X = getattr(n, "X_" + split)
        y = getattr(n, "y_" + split)
        labels.append(y)
        probs.append(predict_proba_model(X, params, kind))
    return np.concatenate(labels), np.concatenate(probs)


def auc_and_ap(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan"), float("nan")
    order = np.argsort(-p, kind="stable")
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    roc = float(np.trapz(
        np.r_[0.0, tp / n_pos, 1.0], np.r_[0.0, fp / n_neg, 1.0]
    ))
    ap = float(np.sum((tp / (tp + fp))[yy == 1]) / n_pos)
    return roc, ap


def threshold_metrics(y, p, threshold: float):
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fn = int(np.sum((y == 1) & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "specificity": spec, "f1": f1, "balanced_accuracy": (recall + spec) / 2,
            "false_positive_rate": 1 - spec, "false_negative_rate": 1 - recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def policies_from_validation(y_val, p_val):
    """Operating thresholds are fitted ONLY on the held-out validation set."""
    candidates = [threshold_metrics(y_val, p_val, float(t))
                  for t in np.linspace(0.05, 0.95, 91)]
    possible = [r for r in candidates if r["recall"] >= 0.90]
    return {
        "best_f1": max(candidates, key=lambda z: z["f1"])["threshold"],
        "best_balanced": max(candidates, key=lambda z: z["balanced_accuracy"])["threshold"],
        "best_operational": max(possible, key=lambda z: z["precision"])["threshold"]
        if possible else None,
    }


def evaluate(params, nodes, kind, seed, paradigm, setting):
    yv, pv = pooled_predictions(params, nodes, kind, split="val")
    threshold_map = policies_from_validation(yv, pv)
    yt, pt = pooled_predictions(params, nodes, kind, split="test")
    roc, ap = auc_and_ap(yt, pt)
    rows = []
    for policy, thr in threshold_map.items():
        if thr is None:
            continue
        metric = threshold_metrics(yt, pt, thr)
        rows.append(dict(seed=seed, model_name="Logistic Regression" if kind == "logistic"
                         else "Keras/TensorFlow Deep Neural Network",
                         setting=setting, paradigm=paradigm, policy=policy,
                         **metric, roc_auc=roc, pr_auc=ap))
    return rows


def run_one_seed(seed: int, cfg=PAPER_LOGISTIC_CONFIG, sample_cap=None):
    nodes = make_nodes(seed, sample_cap=sample_cap)
    kind = cfg["kind"]
    rows = []
    for n in nodes:
        params, _ = train_model(n.X_train, n.y_train, cfg)
        rows.extend(evaluate(params, nodes, kind, seed, "local",
                             "trained on " + n.name))
    for name, weighted in (("classic FedAvg", False),
                           ("source-weighted FedAvg", True)):
        params, _ = train_federated(nodes, cfg, weighted)
        rows.extend(evaluate(params, nodes, kind, seed, "federated", name))
    Xtr = np.concatenate([n.X_train for n in nodes])
    ytr = np.concatenate([n.y_train for n in nodes])
    params, _ = train_model(Xtr, ytr, cfg)
    rows.extend(evaluate(params, nodes, kind, seed, "centralized",
                         "all data in one server"))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_name"], row["paradigm"], row["setting"], row["policy"])].append(row)
    result = []
    for key, entries in sorted(groups.items()):
        entry = dict(zip(("model_name", "paradigm", "setting", "policy"), key))
        entry["n_seeds"] = len(entries)
        for field in ("f1", "recall", "specificity", "balanced_accuracy",
                      "false_positive_rate", "false_negative_rate", "roc_auc", "pr_auc", "threshold"):
            values = np.array([v[field] for v in entries])
            entry[field + "_mean"] = float(values.mean())
            entry[field + "_std"] = float(values.std(ddof=0))
        result.append(entry)
    return result


def sensitivity(nodes):
    original = [n.source_weight for n in nodes]
    rows = []
    try:
        for name, scenario in SOURCE_WEIGHT_SCENARIOS.items():
            for n in nodes:
                n.source_weight = float(scenario[n.node_type])
            params, _ = train_federated(nodes, PAPER_LOGISTIC_CONFIG, weighted=True)
            rows.extend(dict(scenario=name, **r) for r in
                        evaluate(params, nodes, "logistic", 0, "federated",
                                 "source-weighted FedAvg"))
    finally:
        for n, w in zip(nodes, original):
            n.source_weight = w
    return rows


def diagnostics(nodes):
    return [dict(node_name=n.name, node_type=n.node_type,
                 train_samples=len(n.y_train), val_samples=len(n.y_val),
                 test_samples=len(n.y_test), positive_rate=float(n.prevalence),
                 label_noise=n.label_noise, source_weight=n.source_weight)
            for n in nodes]


def save_json_and_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def self_check():
    """Fast smoke test of data partitioning and validation-only thresholds."""
    full_nodes = make_nodes(0)
    expected_counts = [(36001,8999,15000),(27002,6749,11249),
                       (18001,4499,7500),(30002,7499,12499),
                       (15000,3750,6250)]
    counts = [(len(n.y_train),len(n.y_val),len(n.y_test)) for n in full_nodes]
    assert counts == expected_counts, (counts, expected_counts)
    assert sum(len(n.y_train) for n in full_nodes) == 126006
    # Feature means are a fingerprint of the original RNG/split sequence.
    path = Path(__file__).resolve().parent / "archived_results" / "node_non_iid_diagnostics.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    for node, saved in zip(full_nodes, reference):
        assert node.name == saved["node_name"]
        assert abs(float(node.X_train[:, 0].mean()) - saved["rainfall_1h_train_mean"]) < 1e-12
        assert abs(float(node.X_train[:, 3].mean()) - saved["river_level_train_mean"]) < 1e-12
    nodes = make_nodes(11, sample_cap=100)
    assert len(nodes) == 5
    assert all(n.X_train.shape[1] == len(FEATURES) for n in nodes)
    assert all(len(n.y_val) > 0 and len(n.y_test) > 0 for n in nodes)
    weights = aggregate_weights(nodes, weighted=True)
    assert abs(float(weights.sum()) - 1) < 1e-12
    # Strictly perturbing test labels cannot affect validation threshold search.
    yv, pv = pooled_predictions(np.zeros(len(FEATURES) + 1), nodes, "logistic", "val")
    t_before = policies_from_validation(yv, pv)
    nodes[0].y_test[:] = 1 - nodes[0].y_test
    yv, pv = pooled_predictions(np.zeros(len(FEATURES) + 1), nodes, "logistic", "val")
    assert t_before == policies_from_validation(yv, pv)
    print("Self-check OK: synthetic nodes, splits and validation threshold isolation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true",
                        help="Quick test; no model training or output")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run ten full synthetic logistic seeds (costly)")
    parser.add_argument("--neural", action="store_true",
                        help="Run separate single-seed Keras reference (costly)")
    parser.add_argument("--output", default="reconstructed_outputs",
                        help="Never use outputs/ for reconstructed experiments")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.all_seeds and not args.neural:
        parser.error("Choose --self-check, --all-seeds, or --neural")
    dest = Path(args.output).resolve()
    if dest.name == "outputs":
        parser.error("The historical outputs/ directory is read-only for this reconstruction")
    nodes0 = make_nodes(0)
    save_json_and_csv(dest / "node_diagnostics", diagnostics(nodes0))
    save_json_and_csv(dest / "source_weight_sensitivity", sensitivity(nodes0))
    if args.all_seeds:
        all_rows = []
        for seed in PAPER_MULTI_SEED_VALUES:
            all_rows.extend(run_one_seed(seed))
            print("Completed logistic seed", seed, flush=True)
        save_json_and_csv(dest / "multiseed_results", all_rows)
        save_json_and_csv(dest / "multiseed_summary", summarize(all_rows))
    if args.neural:
        # Separate single-seed exploratory reference, not a ten-seed estimate.
        rows = run_one_seed(0, cfg=PAPER_NEURAL_NETWORK_CONFIG)
        save_json_and_csv(dest / "neural_reference", rows)
    ntrain = sum(len(n.y_train) for n in nodes0)
    counts = {
        "centralized_logistic_scalars": ntrain * (len(FEATURES) + 1),
        "federated_logistic_scalars": 2 * PAPER_LOGISTIC_CONFIG["federated_rounds"]
        * len(nodes0) * (len(FEATURES) + 1),
        "federated_neural_scalars": 2 * PAPER_NEURAL_NETWORK_CONFIG["federated_rounds"]
        * len(nodes0) * 11777,
    }
    (dest / "communication_proxy.json").write_text(json.dumps(counts, indent=2))
    print("Reconstruction artifacts written to", dest)


if __name__ == "__main__":
    main()
