"""Metriche e funzioni di valutazione dei modelli."""

import numpy as np

from model import predict_proba_model


def roc_auc_score_binary(y_true, y_prob):
    """Calcola ROC-AUC binaria senza dipendenze esterne."""

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))

    return float(np.trapz(tpr, fpr))


def pr_auc_score_binary(y_true, y_prob):
    """Calcola area precision-recall come average precision."""

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    positives = np.sum(y_true == 1)

    if positives == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    ranks = np.arange(1, len(sorted_true) + 1)
    precision = tp / ranks

    return float(np.sum(precision[sorted_true == 1]) / positives)


def classification_metrics(y_true, y_prob, threshold=0.5, include_auc=True):
    """Calcola metriche binarie e conteggi TP/TN/FP/FN."""

    # Converte probabilita' in classi 0/1 usando la soglia.
    y_pred = (y_prob >= threshold).astype(int)

    # Conteggi della matrice di confusione.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # max(..., 1) evita divisioni per zero.
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    # 1e-12 evita divisione per zero se precision + recall e' 0.
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    false_positive_rate = fp / max(fp + tn, 1)
    false_negative_rate = fn / max(fn + tp, 1)
    balanced_accuracy = 0.5 * (recall + specificity)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    if include_auc:
        metrics["roc_auc"] = roc_auc_score_binary(y_true, y_prob)
        metrics["pr_auc"] = pr_auc_score_binary(y_true, y_prob)

    return metrics


def threshold_search(y_true, y_prob, min_recall=0.90, thresholds=None):
    """Cerca soglie utili per F1, balanced accuracy e vincolo operativo."""

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    results = []

    auc_metrics = {
        "roc_auc": roc_auc_score_binary(y_true, y_prob),
        "pr_auc": pr_auc_score_binary(y_true, y_prob),
    }

    for threshold in thresholds:
        metrics = classification_metrics(
            y_true,
            y_prob,
            threshold=threshold,
            include_auc=False,
        )
        row = {"threshold": float(threshold)}
        row.update(metrics)
        row.update(auc_metrics)
        results.append(row)

    best_f1 = max(results, key=lambda row: row["f1"])
    best_balanced = max(results, key=lambda row: row["balanced_accuracy"])

    feasible = [row for row in results if row["recall"] >= min_recall]
    best_operational = max(feasible, key=lambda row: row["precision"]) if feasible else None

    return {
        "best_f1": best_f1,
        "best_balanced": best_balanced,
        "best_operational": best_operational,
        "all_thresholds": results,
    }


def collect_model_predictions(params, nodes, model_kind, split="test"):
    """Raccoglie y e probabilita' su tutti i nodi per uno split."""

    all_y = []
    all_prob = []

    for node in nodes:
        if split == "train":
            X = node.X_train
            y = node.y_train
        elif split == "test":
            X = node.X_test
            y = node.y_test
        else:
            raise ValueError("split must be train or test")

        all_y.append(y)
        all_prob.append(predict_proba_model(X, params, model_kind))

    return np.concatenate(all_y), np.concatenate(all_prob)


def evaluate_model(params, nodes, model_kind, threshold=0.5, title="Model evaluation"):
    """Valuta un modello su ogni nodo e sull'unione globale dei test set."""

    print(title)
    print("-" * len(title))

    # Accumulatori per costruire il test set globale.
    all_y = []
    all_prob = []

    for node in nodes:
        # Predice sul test set locale del nodo.
        y_prob = predict_proba_model(node.X_test, params, model_kind)

        # Calcola metriche locali.
        metrics = classification_metrics(node.y_test, y_prob, threshold=threshold)

        # Salva i dati per le metriche globali.
        all_y.append(node.y_test)
        all_prob.append(y_prob)

        print(node.name)
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Balanced acc: {metrics['balanced_accuracy']:.3f}")
        print(f"  F1-score:  {metrics['f1']:.3f}")
        print(f"  ROC-AUC/PR-AUC: {metrics['roc_auc']:.3f}/{metrics['pr_auc']:.3f}")
        print(f"  FPR/FNR: {metrics['false_positive_rate']:.3f}/{metrics['false_negative_rate']:.3f}")
        print(f"  TP/FP/FN/TN: {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")
        print()

    # concatenate unisce array locali in un unico array globale.
    all_y = np.concatenate(all_y)
    all_prob = np.concatenate(all_prob)

    # Metriche globali sull'unione dei test set.
    global_metrics = classification_metrics(all_y, all_prob, threshold=threshold)

    print("Global test set")
    print(f"  Accuracy:  {global_metrics['accuracy']:.3f}")
    print(f"  Precision: {global_metrics['precision']:.3f}")
    print(f"  Recall:    {global_metrics['recall']:.3f}")
    print(f"  Specificity: {global_metrics['specificity']:.3f}")
    print(f"  Balanced acc: {global_metrics['balanced_accuracy']:.3f}")
    print(f"  F1-score:  {global_metrics['f1']:.3f}")
    print(f"  ROC-AUC/PR-AUC: {global_metrics['roc_auc']:.3f}/{global_metrics['pr_auc']:.3f}")
    print(f"  FPR/FNR: {global_metrics['false_positive_rate']:.3f}/{global_metrics['false_negative_rate']:.3f}")
    print(f"  TP/FP/FN/TN: {global_metrics['tp']}/{global_metrics['fp']}/{global_metrics['fn']}/{global_metrics['tn']}")
    print()

    return global_metrics
