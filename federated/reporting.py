"""Funzioni di stampa e confronto dei risultati."""

import numpy as np


def print_node_summary(nodes):
    """Stampa dimensione e tasso di positivi per ogni nodo."""

    for node in nodes:
        # mean su etichette 0/1 restituisce la percentuale di classe positiva.
        train_rate = node.y_train.mean()
        test_rate = node.y_test.mean()

        print(node.name)
        print(f"  Type: {node.node_type}")
        print(f"  Source weight: {node.source_weight:.2f}")
        print(f"  Train samples: {len(node.y_train)}")
        print(f"  Test samples:  {len(node.y_test)}")
        print(f"  Positive rate train: {train_rate:.3f}")
        print(f"  Positive rate test:  {test_rate:.3f}")
        print()


def print_model_weights(theta, feature_names):
    """Stampa i pesi del modello ordinati per importanza assoluta."""

    # theta[0] e' l'intercetta perche' add_intercept aggiunge una colonna iniziale.
    intercept = theta[0]

    # theta[1:] contiene i pesi delle feature vere e proprie.
    weights = theta[1:]

    print(f"Intercept: {intercept:.4f}")
    print()

    # abs(weights) prende il valore assoluto dei pesi.
    # argsort restituisce gli indici che ordinerebbero l'array in modo crescente.
    # [::-1] inverte l'ordine, ottenendo dal peso piu' grande al piu' piccolo.
    sorted_idx = np.argsort(np.abs(weights))[::-1]

    for idx in sorted_idx:
        name = feature_names[idx]
        value = weights[idx]

        # Interpretazione semplice del segno del peso.
        direction = "increases risk" if value > 0 else "decreases risk"
        print(f"{name:25s} {value: .4f}   {direction}")


def print_aggregation_weights(nodes, aggregation_weights, use_source_weights=True):
    """Stampa i pesi effettivi usati dall'orchestratore federato."""

    print("Federated aggregation weights")
    print("-" * 40)

    for node, weight in zip(nodes, aggregation_weights):
        source_weight = node.source_weight if use_source_weights else 1.0
        effective_size = len(node.y_train) * source_weight
        print(
            f"{node.name}: "
            f"n_train={len(node.y_train)}, "
            f"source_weight={source_weight:.2f}, "
            f"effective={effective_size:.1f}, "
            f"aggregation={weight:.3f}"
        )

    print()


def print_summary_comparison(results):
    """Stampa confronto sintetico tra modelli federati e centralizzati."""

    print("Summary comparison")
    print("-" * 96)
    print(
        f"{'Experiment':52s} "
        f"{'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'Bal':>6s} {'F1':>6s} "
        f"{'Train s':>9s} {'Moved scalars':>15s}"
    )
    print("-" * 96)

    for row in results:
        metrics = row["metrics"]
        train_seconds = row.get("train_seconds", 0.0)
        moved_scalars = row.get("moved_scalars", 0)

        print(
            f"{row['label'][:52]:52s} "
            f"{metrics['accuracy']:6.3f} "
            f"{metrics['precision']:6.3f} "
            f"{metrics['recall']:6.3f} "
            f"{metrics['specificity']:6.3f} "
            f"{metrics['balanced_accuracy']:6.3f} "
            f"{metrics['f1']:6.3f} "
            f"{train_seconds:9.2f} "
            f"{moved_scalars:15,d}"
        )

    print()


def print_benchmark_interpretation(results):
    """Stampa lettura sintetica del trade-off federato/centralizzato."""

    print("Benchmark interpretation")
    print("-" * 40)

    model_names = []
    for row in results:
        model_name = row["model_name"]
        if model_name not in model_names:
            model_names.append(model_name)

    for model_name in model_names:
        model_rows = [row for row in results if row["model_name"] == model_name]
        centralized = next(row for row in model_rows if row["paradigm"] == "centralized")
        federated_rows = [row for row in model_rows if row["paradigm"] == "federated"]
        best_federated = max(federated_rows, key=lambda row: row["metrics"]["f1"])

        f1_gap = best_federated["metrics"]["f1"] - centralized["metrics"]["f1"]
        time_delta = best_federated["train_seconds"] - centralized["train_seconds"]

        if best_federated["moved_scalars"] > 0:
            movement_ratio = centralized["moved_scalars"] / best_federated["moved_scalars"]
        else:
            movement_ratio = float("inf")

        print(model_name)
        print(f"  Best federated setting: {best_federated['setting']}")
        print(f"  F1 federated vs centralized: {best_federated['metrics']['f1']:.3f} vs {centralized['metrics']['f1']:.3f} ({f1_gap:+.3f})")
        print(f"  Training time federated vs centralized: {best_federated['train_seconds']:.2f}s vs {centralized['train_seconds']:.2f}s ({time_delta:+.2f}s)")
        print(f"  Data/parameter movement reduction: about {movement_ratio:.1f}x fewer scalar values moved")

        if time_delta > 0:
            print("  Note: in this single-process simulation federated training is not necessarily faster; the gain is mainly locality/privacy and reduced data movement.")
        else:
            print("  Note: in this run federated training is also faster on wall-clock time.")

        print()

    neural_rows = [
        row for row in results
        if "Neural" in row["model_name"] and row["paradigm"] == "centralized"
    ]
    logistic_rows = [
        row for row in results
        if row["model_name"] == "Logistic Regression" and row["paradigm"] == "federated"
    ]

    if neural_rows and logistic_rows:
        centralized_neural = neural_rows[0]
        best_federated_logistic = max(logistic_rows, key=lambda row: row["metrics"]["f1"])

        print("Cross-model reading")
        print(f"  Centralized neural network F1: {centralized_neural['metrics']['f1']:.3f}")
        print(f"  Best federated logistic benchmark F1: {best_federated_logistic['metrics']['f1']:.3f}")
        print("  This makes the paper comparison explicit: accuracy/F1 are compared together with the cost of centralizing data.")
        print()
