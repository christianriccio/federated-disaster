"""Training federato con FedAvg classico e FedAvg pesato per sorgente."""

import numpy as np

from metrics import classification_metrics
from model import (
    binary_cross_entropy,
    init_model_params,
    predict_proba_model,
    train_model,
    weighted_average_params,
)


def compute_aggregation_weights(nodes, use_source_weights=True):
    """Calcola i pesi normalizzati dell'orchestratore.

    FedAvg classico usa solo n_i. La variante proposta usa n_i * w_i, dove
    w_i rappresenta affidabilita'/rilevanza della fonte.
    """

    effective_sizes = []

    for node in nodes:
        node_weight = node.source_weight if use_source_weights else 1.0
        effective_sizes.append(len(node.y_train) * node_weight)

    effective_sizes = np.asarray(effective_sizes, dtype=float)
    total_size = np.sum(effective_sizes)

    if total_size <= 0:
        raise ValueError("The sum of aggregation weights must be positive")

    return effective_sizes / total_size


def fedavg_aggregate(local_params, nodes, use_source_weights=True):
    """Aggrega i modelli locali con media pesata dei parametri."""

    aggregation_weights = compute_aggregation_weights(nodes, use_source_weights=use_source_weights)
    return weighted_average_params(local_params, aggregation_weights)


def federated_training(
    nodes,
    model_config,
    rounds=30,
    local_epochs=20,
    threshold=0.5,
    use_source_weights=True,
    verbose=True,
):
    """Esegue piu' round di training federato."""

    # Numero di feature del problema.
    n_features = nodes[0].X_train.shape[1]

    # Modello globale iniziale.
    global_params = init_model_params(model_config, n_features)

    # Storico delle metriche di training per ogni round.
    history = []
    aggregation_weights = compute_aggregation_weights(nodes, use_source_weights=use_source_weights)

    for round_idx in range(1, rounds + 1):
        # Parametri addestrati localmente in questo round.
        local_params = []

        for node in nodes:
            # Ogni nodo parte dal modello globale corrente e fa training locale.
            params_local, _ = train_model(
                node.X_train,
                node.y_train,
                model_config,
                params_init=global_params,
                epochs=local_epochs,
            )

            local_params.append(params_local)

        # Il server aggrega i modelli locali in un nuovo modello globale.
        global_params = fedavg_aggregate(local_params, nodes, use_source_weights=use_source_weights)

        # Valutazione sul training globale, ottenuto unendo virtualmente i nodi.
        all_y_train = []
        all_prob_train = []

        for node in nodes:
            prob = predict_proba_model(node.X_train, global_params, model_config["kind"])
            all_y_train.append(node.y_train)
            all_prob_train.append(prob)

        all_y_train = np.concatenate(all_y_train)
        all_prob_train = np.concatenate(all_prob_train)

        train_loss = binary_cross_entropy(all_y_train, all_prob_train)
        train_metrics = classification_metrics(all_y_train, all_prob_train, threshold=threshold)

        # Salva metriche utili per analizzare l'andamento dei round.
        history.append({
            "round": round_idx,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
        })

        # Stampa solo alcuni round per non produrre troppo output.
        if verbose and (round_idx == 1 or round_idx % 5 == 0 or round_idx == rounds):
            print(
                f"Round {round_idx:02d} | "
                f"loss={train_loss:.4f} | "
                f"acc={train_metrics['accuracy']:.3f} | "
                f"precision={train_metrics['precision']:.3f} | "
                f"recall={train_metrics['recall']:.3f} | "
                f"f1={train_metrics['f1']:.3f}"
            )

    return global_params, history, aggregation_weights
