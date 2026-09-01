

from time import perf_counter

import numpy as np

from artifacts import save_experiment_artifacts
from config import FEATURES, LOGISTIC_CONFIG, MULTI_SEED_VALUES, NEURAL_NETWORK_CONFIG, NODE_SPECS, OUTPUT_DIR, THRESHOLD
from data import create_federated_nodes
from federated import federated_training
from metrics import collect_model_predictions, evaluate_model, threshold_search
from model import count_model_parameters, is_tensorflow_available, train_model
from reporting import (
    print_benchmark_interpretation,
    print_aggregation_weights,
    print_model_weights,
    print_node_summary,
    print_summary_comparison,
)


def train_local_models(nodes, model_config):
    """Addestra un modello separato su ogni nodo e lo valuta su tutti i nodi."""

    # Dizionario: chiave = nome del nodo, valore = parametri del modello locale.
    local_models = {}

    for node in nodes:
        # Addestra il modello usando solo i dati locali del nodo.
        params_local, _ = train_model(
            node.X_train,
            node.y_train,
            model_config,
        )

        # Salva il modello locale nel dizionario.
        local_models[node.name] = params_local

        print(f"Local {model_config['name']} trained on: {node.name}")

        # Valuta il modello locale anche sugli altri nodi, per capire quanto
        # generalizza fuori dal nodo su cui e' stato addestrato.
        evaluate_model(
            params_local,
            nodes,
            model_config["kind"],
            threshold=THRESHOLD,
            title=f"Evaluation of local {model_config['name']}: {node.name}",
        )

    return local_models


def selected_policy_rows(search):
    """Restituisce le policy di soglia disponibili per una threshold search."""

    rows = []

    for policy in ["best_f1", "best_balanced", "best_operational"]:
        metrics = search.get(policy)
        if metrics is None:
            continue
        row = {"policy": policy}
        row.update(metrics)
        rows.append(row)

    return rows


def train_all_local_baselines(nodes, model_config):
    """Addestra un modello local-only per ogni nodo, senza sceglierne uno sul test."""

    results = []
    threshold_blocks = []

    for node in nodes:
        start = perf_counter()
        params_local, _ = train_model(node.X_train, node.y_train, model_config)
        train_seconds = perf_counter() - start

        eval_start = perf_counter()
        fixed_metrics = evaluate_model(
            params_local,
            nodes,
            model_config["kind"],
            threshold=THRESHOLD,
            title=f"Evaluation of local-only {model_config['name']} trained on {node.name}",
        )
        eval_seconds = perf_counter() - eval_start
        y_true, y_prob = collect_model_predictions(params_local, nodes, model_config["kind"])
        search = threshold_search(y_true, y_prob, min_recall=0.90)

        results.append({
            "label": f"{model_config['name']} - Local-only trained on {node.name}",
            "model_name": model_config["name"],
            "paradigm": "local",
            "setting": f"trained on {node.name}",
            "metrics": fixed_metrics,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "moved_scalars": 0,
            "threshold_search": search,
        })
        threshold_blocks.append({
            "model_name": model_config["name"],
            "paradigm": "local",
            "setting": f"trained on {node.name}",
            "search": search,
        })

    return results, threshold_blocks


def append_multiseed_policy_rows(rows, seed, model_name, paradigm, setting, search):
    """Aggiunge al multi-seed una riga per ogni policy di soglia disponibile."""

    for policy_row in selected_policy_rows(search):
        rows.append({
            "seed": seed,
            "model_name": model_name,
            "setting": setting,
            "paradigm": paradigm,
            "policy": policy_row["policy"],
            "f1": policy_row["f1"],
            "recall": policy_row["recall"],
            "specificity": policy_row["specificity"],
            "balanced_accuracy": policy_row["balanced_accuracy"],
            "false_positive_rate": policy_row["false_positive_rate"],
            "false_negative_rate": policy_row["false_negative_rate"],
            "roc_auc": policy_row["roc_auc"],
            "pr_auc": policy_row["pr_auc"],
            "threshold": policy_row["threshold"],
        })


def train_centralized_oracle(nodes, model_config):
    """Addestra un benchmark centralizzato che vede tutti i dati di training.

    Questo modello non rispetta il vincolo federato, perche' unisce i dati dei
    nodi. Serve come confronto ideale, o oracle benchmark.
    """

    # vstack impila verticalmente le matrici X_train dei vari nodi.
    X_train_centralized = np.vstack([node.X_train for node in nodes])

    # concatenate unisce i vettori y_train in un unico vettore.
    y_train_centralized = np.concatenate([node.y_train for node in nodes])

    # Addestra un unico modello su tutti i dati centralizzati.
    params_centralized, _ = train_model(
        X_train_centralized,
        y_train_centralized,
        model_config,
    )

    return params_centralized


def centralized_data_movement(nodes):
    """Stima quanti valori scalari devono arrivare al server centralizzato."""

    n_features = nodes[0].X_train.shape[1]
    total_train_samples = sum(len(node.y_train) for node in nodes)

    # +1 rappresenta la label y trasferita insieme alle feature X.
    return total_train_samples * (n_features + 1)


def federated_parameter_movement(nodes, params, rounds):
    """Stima quanti valori scalari vengono scambiati nel training federato."""

    n_params = count_model_parameters(params)

    # Per ogni round: server -> nodo e nodo -> server.
    return 2 * rounds * len(nodes) * n_params


def shifted_node_specs(seed_offset):
    """Crea una copia dei nodi con seed traslati per esperimenti multi-seed."""

    specs = []

    for spec in NODE_SPECS:
        copied = dict(spec)
        copied["seed"] = spec["seed"] + seed_offset
        specs.append(copied)

    return specs


def run_multiseed_logistic_summary():
    """Esegue una sintesi multi-seed leggera per la regressione logistica."""

    rows = []

    for seed in MULTI_SEED_VALUES:
        nodes = create_federated_nodes(node_specs=shifted_node_specs(seed))

        for setting, use_source_weights in [
            ("classic FedAvg", False),
            ("source-weighted FedAvg", True),
        ]:
            params, _, _ = federated_training(
                nodes,
                LOGISTIC_CONFIG,
                rounds=LOGISTIC_CONFIG["federated_rounds"],
                local_epochs=LOGISTIC_CONFIG["local_epochs"],
                threshold=THRESHOLD,
                use_source_weights=use_source_weights,
                verbose=False,
            )
            y_true, y_prob = collect_model_predictions(params, nodes, LOGISTIC_CONFIG["kind"])
            search = threshold_search(y_true, y_prob, min_recall=0.90)
            append_multiseed_policy_rows(
                rows,
                seed,
                LOGISTIC_CONFIG["name"],
                "federated",
                setting,
                search,
            )

        params = train_centralized_oracle(nodes, LOGISTIC_CONFIG)
        y_true, y_prob = collect_model_predictions(params, nodes, LOGISTIC_CONFIG["kind"])
        search = threshold_search(y_true, y_prob, min_recall=0.90)
        append_multiseed_policy_rows(
            rows,
            seed,
            LOGISTIC_CONFIG["name"],
            "centralized",
            "all data in one server",
            search,
        )

    return rows


def print_benchmark_design(nodes):
    """Spiega chiaramente quale benchmark viene eseguito."""

    total_train = sum(len(node.y_train) for node in nodes)
    total_test = sum(len(node.y_test) for node in nodes)

    print("Benchmark design")
    print("-" * 40)
    print("1. Logistic Regression: federated FedAvg vs centralized all-data baseline")
    print("2. Keras/TensorFlow Neural Network: federated FedAvg vs centralized all-data baseline")
    print("3. Federated setting includes both classic FedAvg and source-weighted FedAvg")
    print(f"4. Total samples: train={total_train}, test={total_test}")
    print("5. Timing is measured on this single machine; data movement is estimated separately")
    print()


def run_experiment(nodes, model_config, train_local=False):
    """Esegue locale, federato e centralizzato per un modello."""

    print("=" * 70)
    print(model_config["name"])
    print("=" * 70)

    local_results = []
    local_thresholds = []
    if train_local:
        local_results, local_thresholds = train_all_local_baselines(nodes, model_config)
        print("Local-only baselines")
        print("-" * 40)
        print(f"Saved one local-only row for each node: {len(local_results)} models")
        print()

    # FedAvg classico: pesi proporzionali solo alla numerosita' locale.
    start = perf_counter()
    params_fedavg, classic_history, classic_weights = federated_training(
        nodes,
        model_config,
        rounds=model_config["federated_rounds"],
        local_epochs=model_config["local_epochs"],
        threshold=THRESHOLD,
        use_source_weights=False,
    )
    fedavg_train_seconds = perf_counter() - start
    fedavg_moved_scalars = federated_parameter_movement(
        nodes,
        params_fedavg,
        model_config["federated_rounds"],
    )

    print_aggregation_weights(nodes, classic_weights, use_source_weights=False)

    start = perf_counter()
    fedavg_metrics = evaluate_model(
        params_fedavg,
        nodes,
        model_config["kind"],
        threshold=THRESHOLD,
        title=f"Evaluation of federated {model_config['name']} - classic FedAvg",
    )
    fedavg_eval_seconds = perf_counter() - start
    fedavg_y, fedavg_prob = collect_model_predictions(params_fedavg, nodes, model_config["kind"])
    fedavg_thresholds = threshold_search(fedavg_y, fedavg_prob, min_recall=0.90)

    # FedAvg pesato: n_i modulato dal source_weight configurato per il nodo.
    start = perf_counter()
    params_weighted, weighted_history, weighted_weights = federated_training(
        nodes,
        model_config,
        rounds=model_config["federated_rounds"],
        local_epochs=model_config["local_epochs"],
        threshold=THRESHOLD,
        use_source_weights=True,
    )
    weighted_train_seconds = perf_counter() - start
    weighted_moved_scalars = federated_parameter_movement(
        nodes,
        params_weighted,
        model_config["federated_rounds"],
    )

    print_aggregation_weights(nodes, weighted_weights, use_source_weights=True)

    start = perf_counter()
    weighted_metrics = evaluate_model(
        params_weighted,
        nodes,
        model_config["kind"],
        threshold=THRESHOLD,
        title=f"Evaluation of federated {model_config['name']} - source-weighted FedAvg",
    )
    weighted_eval_seconds = perf_counter() - start
    weighted_y, weighted_prob = collect_model_predictions(params_weighted, nodes, model_config["kind"])
    weighted_thresholds = threshold_search(weighted_y, weighted_prob, min_recall=0.90)

    # Addestra e valuta il modello centralizzato di confronto.
    start = perf_counter()
    params_centralized = train_centralized_oracle(nodes, model_config)
    centralized_train_seconds = perf_counter() - start
    centralized_moved_scalars = centralized_data_movement(nodes)

    start = perf_counter()
    centralized_metrics = evaluate_model(
        params_centralized,
        nodes,
        model_config["kind"],
        threshold=THRESHOLD,
        title=f"Evaluation of centralized {model_config['name']} - Oracle benchmark",
    )
    centralized_eval_seconds = perf_counter() - start
    centralized_y, centralized_prob = collect_model_predictions(params_centralized, nodes, model_config["kind"])
    centralized_thresholds = threshold_search(centralized_y, centralized_prob, min_recall=0.90)

    print("Runtime and movement summary")
    print("-" * 40)
    print(f"Federated classic FedAvg: train={fedavg_train_seconds:.2f}s, eval={fedavg_eval_seconds:.2f}s, moved_scalars={fedavg_moved_scalars:,}")
    print(f"Federated weighted FedAvg: train={weighted_train_seconds:.2f}s, eval={weighted_eval_seconds:.2f}s, moved_scalars={weighted_moved_scalars:,}")
    print(f"Centralized oracle:        train={centralized_train_seconds:.2f}s, eval={centralized_eval_seconds:.2f}s, moved_scalars={centralized_moved_scalars:,}")
    print()

    experiment = {
        "model_name": model_config["name"],
        "fedavg_params": params_fedavg,
        "weighted_params": params_weighted,
        "centralized_params": params_centralized,
        "histories": [
            {
                "setting": "classic FedAvg",
                "history": classic_history,
            },
            {
                "setting": "source-weighted FedAvg",
                "history": weighted_history,
            },
        ],
        "thresholds": local_thresholds + [
            {
                "model_name": model_config["name"],
                "paradigm": "federated",
                "setting": "classic FedAvg",
                "search": fedavg_thresholds,
            },
            {
                "model_name": model_config["name"],
                "paradigm": "federated",
                "setting": "source-weighted FedAvg",
                "search": weighted_thresholds,
            },
            {
                "model_name": model_config["name"],
                "paradigm": "centralized",
                "setting": "all data in one server",
                "search": centralized_thresholds,
            },
        ],
        "results": local_results + [
            {
                "label": f"{model_config['name']} - Federated classic FedAvg",
                "model_name": model_config["name"],
                "paradigm": "federated",
                "setting": "classic FedAvg",
                "metrics": fedavg_metrics,
                "train_seconds": fedavg_train_seconds,
                "eval_seconds": fedavg_eval_seconds,
                "moved_scalars": fedavg_moved_scalars,
                "threshold_search": fedavg_thresholds,
            },
            {
                "label": f"{model_config['name']} - Federated source-weighted FedAvg",
                "model_name": model_config["name"],
                "paradigm": "federated",
                "setting": "source-weighted FedAvg",
                "metrics": weighted_metrics,
                "train_seconds": weighted_train_seconds,
                "eval_seconds": weighted_eval_seconds,
                "moved_scalars": weighted_moved_scalars,
                "threshold_search": weighted_thresholds,
            },
            {
                "label": f"{model_config['name']} - Centralized oracle",
                "model_name": model_config["name"],
                "paradigm": "centralized",
                "setting": "all data in one server",
                "metrics": centralized_metrics,
                "train_seconds": centralized_train_seconds,
                "eval_seconds": centralized_eval_seconds,
                "moved_scalars": centralized_moved_scalars,
                "threshold_search": centralized_thresholds,
            },
        ],
    }

    return experiment


def main():
    """Esegue l'intero esperimento."""

    # Crea i nodi federati sintetici.
    nodes = create_federated_nodes()

    # Stampa una sintesi iniziale dei dataset locali.
    print_node_summary(nodes)
    print_benchmark_design(nodes)

    logistic_experiment = run_experiment(nodes, LOGISTIC_CONFIG, train_local=True)

    experiments = [logistic_experiment]

    if is_tensorflow_available():
        neural_experiment = run_experiment(nodes, NEURAL_NETWORK_CONFIG, train_local=False)
        experiments.append(neural_experiment)
    else:
        print("=" * 70)
        print(NEURAL_NETWORK_CONFIG["name"])
        print("=" * 70)
        print("TensorFlow/Keras is not installed, so the neural-network experiment was not executed.")
        print("Install it with: python3 -m pip install -r requirements.txt")
        print()

    # Stampa i pesi del modello federato ordinati per importanza.
    print("Source-weighted federated logistic model weights")
    print("-" * 25)
    print_model_weights(logistic_experiment["weighted_params"], FEATURES)
    print()

    # Stampa il confronto finale.
    all_results = [row for experiment in experiments for row in experiment["results"]]
    print_summary_comparison(all_results)
    print_benchmark_interpretation(all_results)

    artifacts = save_experiment_artifacts(
        all_results,
        experiments,
        OUTPUT_DIR,
        nodes=nodes,
        multiseed_rows=run_multiseed_logistic_summary(),
    )

    print("Saved paper artifacts")
    print("-" * 40)
    for label, path in artifacts.items():
        print(f"{label}: {path}")
    print()


# Questo blocco fa partire main() solo quando esegui direttamente:
# python federated/main.py
# Se invece importi questo file da un altro modulo, main() non parte da solo.
if __name__ == "__main__":
    main()
