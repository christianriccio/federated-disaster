"""Salvataggio di risultati e grafici per la sezione sperimentale."""

import csv
import json
from pathlib import Path

import numpy as np

from config import FEATURES


def to_builtin(value):
    """Converte tipi NumPy in tipi Python serializzabili."""

    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_builtin(item) for item in value]

    return value


def flatten_result(row):
    """Appiattisce una riga risultato per CSV/JSON tabellare."""

    metrics = row["metrics"]

    return {
        "label": row["label"],
        "model_name": row["model_name"],
        "paradigm": row["paradigm"],
        "setting": row["setting"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1": metrics["f1"],
        "false_positive_rate": metrics["false_positive_rate"],
        "false_negative_rate": metrics["false_negative_rate"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "selected_threshold": metrics.get("threshold", 0.5),
        "train_seconds": row["train_seconds"],
        "eval_seconds": row["eval_seconds"],
        "moved_scalars": row["moved_scalars"],
    }


def flatten_thresholds(experiments):
    """Converte threshold search in righe CSV/JSON."""

    rows = []

    for experiment in experiments:
        for threshold_block in experiment.get("thresholds", []):
            for policy in ["best_f1", "best_balanced", "best_operational"]:
                selected = threshold_block["search"].get(policy)
                if selected is None:
                    continue

                row = {
                    "model_name": threshold_block["model_name"],
                    "paradigm": threshold_block["paradigm"],
                    "setting": threshold_block["setting"],
                    "policy": policy,
                }
                row.update(selected)
                rows.append(row)

    return rows


def flatten_histories(experiments):
    """Converte gli storici federati in righe tabellari."""

    rows = []

    for experiment in experiments:
        for history_block in experiment["histories"]:
            for item in history_block["history"]:
                rows.append({
                    "model_name": experiment["model_name"],
                    "setting": history_block["setting"],
                    "round": item["round"],
                    "train_loss": item["train_loss"],
                    "train_accuracy": item["train_accuracy"],
                    "train_precision": item["train_precision"],
                    "train_recall": item["train_recall"],
                    "train_f1": item["train_f1"],
                })

    return rows


def write_csv(path, rows):
    """Scrive righe dizionario in CSV."""

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_node_diagnostics(nodes):
    """Crea diagnostica non-IID per ogni nodo."""

    if not nodes:
        return []

    classic_total = sum(len(node.y_train) for node in nodes)
    weighted_total = sum(len(node.y_train) * node.source_weight for node in nodes)
    rows = []

    for node in nodes:
        row = {
            "node_name": node.name,
            "node_type": node.node_type,
            "source_weight": node.source_weight,
            "train_samples": len(node.y_train),
            "test_samples": len(node.y_test),
            "positive_rate_train": float(np.mean(node.y_train)),
            "positive_rate_test": float(np.mean(node.y_test)),
            "classic_aggregation_weight": len(node.y_train) / classic_total,
            "source_weighted_aggregation_weight": (len(node.y_train) * node.source_weight) / weighted_total,
        }

        for idx, feature in enumerate(FEATURES):
            row[f"{feature}_train_mean"] = float(np.mean(node.X_train[:, idx]))
            row[f"{feature}_train_std"] = float(np.std(node.X_train[:, idx]))

        rows.append(row)

    return rows


def aggregate_multiseed_rows(rows):
    """Aggrega righe multi-seed in media/deviazione standard."""

    grouped = {}

    for row in rows:
        key = (row["model_name"], row["paradigm"], row["setting"], row["policy"])
        grouped.setdefault(key, []).append(row)

    summary = []
    metrics = [
        "f1",
        "recall",
        "specificity",
        "balanced_accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "roc_auc",
        "pr_auc",
        "threshold",
    ]

    for (model_name, paradigm, setting, policy), group_rows in grouped.items():
        item = {
            "model_name": model_name,
            "paradigm": paradigm,
            "setting": setting,
            "policy": policy,
            "n_seeds": len(group_rows),
        }

        for metric in metrics:
            values = np.asarray([row[metric] for row in group_rows], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))

        summary.append(item)

    return summary


def flatten_paper_ready_results(result_rows, threshold_rows):
    """Crea una tabella compatta gia' orientata alle tabelle del paper."""

    movement_by_key = {
        (row["model_name"], row["paradigm"], row["setting"]): row["moved_scalars"]
        for row in result_rows
    }
    train_time_by_key = {
        (row["model_name"], row["paradigm"], row["setting"]): row["train_seconds"]
        for row in result_rows
    }
    paper_rows = []

    for row in result_rows:
        paper_rows.append({
            "model_name": row["model_name"],
            "paradigm": row["paradigm"],
            "setting": row["setting"],
            "policy": "fixed_threshold_0.50",
            "threshold": row["selected_threshold"],
            "f1": row["f1"],
            "recall": row["recall"],
            "specificity": row["specificity"],
            "balanced_accuracy": row["balanced_accuracy"],
            "false_positive_rate": row["false_positive_rate"],
            "false_negative_rate": row["false_negative_rate"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "moved_scalars": row["moved_scalars"],
            "train_seconds": row["train_seconds"],
        })

    for row in threshold_rows:
        key = (row["model_name"], row["paradigm"], row["setting"])
        paper_rows.append({
            "model_name": row["model_name"],
            "paradigm": row["paradigm"],
            "setting": row["setting"],
            "policy": row["policy"],
            "threshold": row["threshold"],
            "f1": row["f1"],
            "recall": row["recall"],
            "specificity": row["specificity"],
            "balanced_accuracy": row["balanced_accuracy"],
            "false_positive_rate": row["false_positive_rate"],
            "false_negative_rate": row["false_negative_rate"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "moved_scalars": movement_by_key.get(key, 0),
            "train_seconds": train_time_by_key.get(key, 0.0),
        })

    return paper_rows


def save_experiment_artifacts(results, experiments, output_dir, nodes=None, multiseed_rows=None):
    """Salva risultati, storici e grafici in output_dir."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_rows = [to_builtin(flatten_result(row)) for row in results]
    history_rows = [to_builtin(row) for row in flatten_histories(experiments)]
    threshold_rows = [to_builtin(row) for row in flatten_thresholds(experiments)]
    node_rows = [to_builtin(row) for row in flatten_node_diagnostics(nodes or [])]
    multiseed_rows = [to_builtin(row) for row in (multiseed_rows or [])]
    multiseed_summary_rows = [to_builtin(row) for row in aggregate_multiseed_rows(multiseed_rows)]
    paper_rows = [to_builtin(row) for row in flatten_paper_ready_results(result_rows, threshold_rows)]

    write_csv(output_path / "results_summary.csv", result_rows)
    write_csv(output_path / "federated_round_history.csv", history_rows)
    write_csv(output_path / "threshold_search_summary.csv", threshold_rows)
    write_csv(output_path / "node_non_iid_diagnostics.csv", node_rows)
    write_csv(output_path / "multiseed_results.csv", multiseed_rows)
    write_csv(output_path / "multiseed_summary.csv", multiseed_summary_rows)
    write_csv(output_path / "paper_ready_results.csv", paper_rows)

    with (output_path / "results_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result_rows, handle, indent=2)

    with (output_path / "federated_round_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history_rows, handle, indent=2)

    with (output_path / "threshold_search_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(threshold_rows, handle, indent=2)

    with (output_path / "node_non_iid_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(node_rows, handle, indent=2)

    with (output_path / "multiseed_results.json").open("w", encoding="utf-8") as handle:
        json.dump(multiseed_rows, handle, indent=2)

    with (output_path / "multiseed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(multiseed_summary_rows, handle, indent=2)

    with (output_path / "paper_ready_results.json").open("w", encoding="utf-8") as handle:
        json.dump(paper_rows, handle, indent=2)

    plots_path = output_path / "plots"
    plots_path.mkdir(exist_ok=True)
    write_convergence_svg(
        plots_path / "convergence_train_loss.svg",
        history_rows,
        metric="train_loss",
        title="Federated convergence - train loss",
        y_label="Loss",
    )
    write_convergence_svg(
        plots_path / "convergence_train_f1.svg",
        history_rows,
        metric="train_f1",
        title="Federated convergence - train F1",
        y_label="F1-score",
    )
    write_convergence_svg(
        plots_path / "convergence_train_accuracy.svg",
        history_rows,
        metric="train_accuracy",
        title="Federated convergence - train accuracy",
        y_label="Accuracy",
    )
    write_convergence_svg(
        plots_path / "convergence_train_precision.svg",
        history_rows,
        metric="train_precision",
        title="Federated convergence - train precision",
        y_label="Precision",
    )
    write_convergence_svg(
        plots_path / "convergence_train_recall.svg",
        history_rows,
        metric="train_recall",
        title="Federated convergence - train recall",
        y_label="Recall",
    )
    write_bar_svg(
        plots_path / "final_f1_comparison.svg",
        result_rows,
        metric="f1",
        title="Final F1 comparison",
        y_label="F1-score",
    )
    write_bar_svg(
        plots_path / "final_accuracy_comparison.svg",
        result_rows,
        metric="accuracy",
        title="Final accuracy comparison",
        y_label="Accuracy",
    )
    write_bar_svg(
        plots_path / "training_time_comparison.svg",
        result_rows,
        metric="train_seconds",
        title="Training time comparison",
        y_label="Seconds",
    )
    write_bar_svg(
        plots_path / "data_movement_comparison.svg",
        result_rows,
        metric="moved_scalars",
        title="Data/parameter movement comparison",
        y_label="Moved scalar values",
    )

    return {
        "summary_csv": output_path / "results_summary.csv",
        "summary_json": output_path / "results_summary.json",
        "history_csv": output_path / "federated_round_history.csv",
        "history_json": output_path / "federated_round_history.json",
        "threshold_csv": output_path / "threshold_search_summary.csv",
        "threshold_json": output_path / "threshold_search_summary.json",
        "node_diagnostics_csv": output_path / "node_non_iid_diagnostics.csv",
        "node_diagnostics_json": output_path / "node_non_iid_diagnostics.json",
        "multiseed_csv": output_path / "multiseed_results.csv",
        "multiseed_json": output_path / "multiseed_results.json",
        "multiseed_summary_csv": output_path / "multiseed_summary.csv",
        "multiseed_summary_json": output_path / "multiseed_summary.json",
        "paper_ready_csv": output_path / "paper_ready_results.csv",
        "paper_ready_json": output_path / "paper_ready_results.json",
        "loss_plot": plots_path / "convergence_train_loss.svg",
        "f1_plot": plots_path / "convergence_train_f1.svg",
        "accuracy_plot": plots_path / "convergence_train_accuracy.svg",
        "precision_plot": plots_path / "convergence_train_precision.svg",
        "recall_plot": plots_path / "convergence_train_recall.svg",
        "final_f1_plot": plots_path / "final_f1_comparison.svg",
        "final_accuracy_plot": plots_path / "final_accuracy_comparison.svg",
        "training_time_plot": plots_path / "training_time_comparison.svg",
        "data_movement_plot": plots_path / "data_movement_comparison.svg",
    }


def group_series(history_rows, metric):
    """Raggruppa lo storico in curve modello/impostazione."""

    grouped = {}

    for row in history_rows:
        key = f"{row['model_name']} - {row['setting']}"
        grouped.setdefault(key, []).append((row["round"], row[metric]))

    for values in grouped.values():
        values.sort(key=lambda item: item[0])

    return grouped


def short_label(label):
    """Abbrevia etichette lunghe per legende e assi."""

    replacements = [
        ("Keras/TensorFlow Deep Neural Network", "Keras DNN"),
        ("Keras/TensorFlow Neural Network", "Keras NN"),
        ("Logistic Regression", "LogReg"),
        ("Federated source-weighted FedAvg", "Fed weighted"),
        ("Federated classic FedAvg", "Fed classic"),
        ("Centralized oracle", "Centralized"),
        ("source-weighted FedAvg", "Weighted"),
        ("classic FedAvg", "Classic"),
        ("all data in one server", "Centralized"),
    ]

    value = label
    for old, new in replacements:
        value = value.replace(old, new)

    if len(value) > 34:
        value = value[:31] + "..."

    return value


def write_convergence_svg(path, history_rows, metric, title, y_label):
    """Disegna un grafico SVG senza dipendenze esterne."""

    if not history_rows:
        return

    series = group_series(history_rows, metric)
    width = 980
    height = 560
    left = 80
    right = 40
    top = 55
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_rounds = [point[0] for values in series.values() for point in values]
    all_values = [point[1] for values in series.values() for point in values]
    min_round = min(all_rounds)
    max_round = max(all_rounds)
    min_value = min(all_values)
    max_value = max(all_values)

    if min_value == max_value:
        min_value -= 0.01
        max_value += 0.01

    pad = (max_value - min_value) * 0.08
    min_value -= pad
    max_value += pad

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    def x_scale(round_idx):
        if max_round == min_round:
            return left
        return left + (round_idx - min_round) / (max_round - min_round) * plot_width

    def y_scale(value):
        return top + (max_value - value) / (max_value - min_value) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="{left + plot_width / 2 - 25}" y="{height - 20}" font-family="Arial" font-size="14">Round</text>',
        f'<text x="20" y="{top + plot_height / 2}" font-family="Arial" font-size="14" transform="rotate(-90 20 {top + plot_height / 2})">{y_label}</text>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_height
        value = max_value - ratio * (max_value - min_value)
        svg.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>')

    for tick in range(0, 6):
        ratio = tick / 5
        round_value = min_round + ratio * (max_round - min_round)
        x = left + ratio * plot_width
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 5}" stroke="#222"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Arial" font-size="12">{round_value:.0f}</text>')

    legend_width = 235
    legend_height = 20 + 22 * len(series)
    legend_x = left + plot_width - legend_width - 14
    legend_y = top + 14
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" fill="white" stroke="#d0d0d0" stroke-width="1" opacity="0.92"/>')

    for idx, (label, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{x_scale(round_idx):.1f},{y_scale(value):.1f}" for round_idx, value in values)
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        item_y = legend_y + 18 + idx * 22
        svg.append(f'<line x1="{legend_x + 12}" y1="{item_y}" x2="{legend_x + 34}" y2="{item_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 42}" y="{item_y + 4}" font-family="Arial" font-size="11">{short_label(label)}</text>')

    svg.append("</svg>")

    path.write_text("\n".join(svg), encoding="utf-8")


def write_bar_svg(path, result_rows, metric, title, y_label):
    """Disegna un bar chart SVG per metriche finali."""

    if not result_rows:
        return

    width = 1080
    height = 620
    left = 90
    right = 40
    top = 60
    bottom = 150
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row[metric]) for row in result_rows]
    max_value = max(values)
    min_value = 0.0

    if metric in {"accuracy", "precision", "recall", "f1"}:
        max_value = max(1.0, max_value)
    else:
        max_value = max_value * 1.15 if max_value > 0 else 1.0

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    bar_gap = 18
    bar_width = max(28, (plot_width - bar_gap * (len(result_rows) + 1)) / max(len(result_rows), 1))

    def y_scale(value):
        return top + (max_value - value) / (max_value - min_value) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="22" y="{top + plot_height / 2}" font-family="Arial" font-size="14" transform="rotate(-90 22 {top + plot_height / 2})">{y_label}</text>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_height
        value = max_value - ratio * (max_value - min_value)
        svg.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3g}</text>')

    for idx, row in enumerate(result_rows):
        value = float(row[metric])
        x = left + bar_gap + idx * (bar_width + bar_gap)
        y = y_scale(value)
        color = colors[idx % len(colors)]
        height_value = top + plot_height - y
        label = short_label(row["label"])
        value_text = f"{value:.3f}" if value < 1000 else f"{value:,.0f}"

        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height_value:.1f}" fill="{color}" opacity="0.86"/>')
        svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value_text}</text>')
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{top + plot_height + 18}" '
            f'text-anchor="end" font-family="Arial" font-size="11" '
            f'transform="rotate(-35 {x + bar_width / 2:.1f} {top + plot_height + 18})">{label}</text>'
        )

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")
