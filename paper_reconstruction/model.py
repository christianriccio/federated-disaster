"""Modelli binari: regressione logistica e rete neurale Keras/TensorFlow."""

import numpy as np


KERAS_MODEL_KIND = "keras_neural_network"


def import_tensorflow():
    """Importa TensorFlow solo quando serve davvero."""

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow/Keras is required for the neural-network experiment. "
            "Install it with: python3 -m pip install -r requirements.txt"
        ) from exc

    return tf


def is_tensorflow_available():
    """Restituisce True se TensorFlow e Keras sono disponibili."""

    try:
        import_tensorflow()
    except Exception:
        return False

    return True


def sigmoid(z):
    """Trasforma logit reali in probabilita' tra 0 e 1."""

    # clip evita overflow numerici quando exp riceve valori troppo grandi.
    z = np.clip(z, -40, 40)

    # exp lavora elemento per elemento se z e' un array NumPy.
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X):
    """Aggiunge una colonna di 1 alla matrice delle feature X."""

    # X.shape[0] e' il numero di campioni.
    # np.ones crea una colonna di 1, cioe' il termine di intercetta.
    ones = np.ones((X.shape[0], 1))

    # hstack affianca orizzontalmente la colonna di 1 e la matrice X.
    return np.hstack([ones, X])


def predict_proba(X, theta):
    """Predice la probabilita' della classe positiva per ogni campione."""

    # Aggiunge l'intercetta per rendere compatibile X con theta.
    X_aug = add_intercept(X)

    # @ e' il prodotto matriciale: produce un logit per ogni riga.
    return sigmoid(X_aug.dot(theta))


def predict_proba_model(X, params, model_kind):
    """Predice probabilita' usando il modello richiesto."""

    if model_kind == "logistic":
        return predict_proba(X, params)
    if model_kind == "neural_network":
        return predict_proba_neural_network(X, params)
    if model_kind == KERAS_MODEL_KIND:
        return predict_proba_keras_neural_network(X, params)

    raise ValueError(f"Unsupported model kind: {model_kind}")


def binary_cross_entropy(y_true, y_prob, sample_weight=None, eps=1e-12):
    """Calcola la loss binary cross-entropy."""

    # Evita log(0), che darebbe infinito.
    y_prob = np.clip(y_prob, eps, 1 - eps)

    # Media della loss su tutti i campioni.
    losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    if sample_weight is None:
        loss = np.mean(losses)
    else:
        loss = np.sum(sample_weight * losses) / np.sum(sample_weight)

    return loss


def binary_class_weight_dict(y, class_weight=None):
    """Calcola pesi di classe per problemi binari sbilanciati."""

    if class_weight is None:
        return None

    if isinstance(class_weight, dict):
        return class_weight

    if class_weight != "balanced":
        raise ValueError("class_weight must be None, 'balanced' or a dict")

    n_samples = len(y)
    n_positive = max(int(np.sum(y == 1)), 1)
    n_negative = max(int(np.sum(y == 0)), 1)

    return {
        0: n_samples / (2 * n_negative),
        1: n_samples / (2 * n_positive),
    }


def sample_weights_from_classes(y, class_weight=None):
    """Espande i pesi di classe in un peso per ogni campione."""

    weights = binary_class_weight_dict(y, class_weight=class_weight)

    if weights is None:
        return None

    return np.where(y == 1, weights[1], weights[0]).astype(float)


def compute_gradient(X, y, theta, sample_weight=None):
    """Calcola il gradiente della loss rispetto ai parametri theta."""

    # Aggiunge l'intercetta alla matrice X.
    X_aug = add_intercept(X)

    # Probabilita' predette dal modello corrente.
    y_prob = sigmoid(X_aug.dot(theta))

    # Formula vettoriale del gradiente medio della regressione logistica.
    errors = y_prob - y

    if sample_weight is None:
        grad = X_aug.T.dot(errors) / len(y)
    else:
        grad = X_aug.T.dot(sample_weight * errors) / np.sum(sample_weight)

    return grad


def train_logistic_regression(
    X,
    y,
    theta_init=None,
    lr=0.15,
    epochs=200,
    l2=0.001,
    class_weight=None,
):
    """Addestra una regressione logistica con discesa del gradiente."""

    # Numero di feature senza intercetta.
    n_features = X.shape[1]

    if theta_init is None:
        # +1 per il parametro di intercetta.
        theta = np.zeros(n_features + 1)
    else:
        # copy evita di modificare l'array iniziale passato dal chiamante.
        theta = theta_init.copy()

    # Lista dell'andamento della loss durante il training.
    losses = []
    sample_weight = sample_weights_from_classes(y, class_weight=class_weight)

    # La matrice aumentata con l'intercetta viene costruita UNA volta sola:
    # ricrearla a ogni epoca (come faceva compute_gradient) domina il tempo
    # di training sui dataset grandi.
    X_aug = add_intercept(X)
    weight_total = None if sample_weight is None else np.sum(sample_weight)

    for _ in range(epochs):
        # Probabilita' predette dal modello corrente.
        y_prob = sigmoid(X_aug.dot(theta))

        # Formula vettoriale del gradiente medio della regressione logistica.
        errors = y_prob - y

        if sample_weight is None:
            grad = X_aug.T.dot(errors) / len(y)
        else:
            grad = X_aug.T.dot(sample_weight * errors) / weight_total

        # Penalita' L2 sui pesi.
        reg = l2 * theta

        # Non regolarizziamo l'intercetta.
        reg[0] = 0.0

        # Aggiornamento dei parametri.
        theta -= lr * (grad + reg)

        # Calcola e salva la loss dopo l'aggiornamento.
        y_prob = sigmoid(X_aug.dot(theta))
        loss = binary_cross_entropy(y, y_prob, sample_weight=sample_weight)
        losses.append(loss)

    return theta, losses


def init_neural_network(n_features, hidden_units=12, seed=42):
    """Inizializza una rete neurale shallow per classificazione binaria."""

    rng = np.random.default_rng(seed)

    return {
        "W1": rng.normal(0.0, 0.15, size=(n_features, hidden_units)),
        "b1": np.zeros(hidden_units),
        "W2": rng.normal(0.0, 0.15, size=hidden_units),
        "b2": np.array(0.0),
    }


def copy_params(params):
    """Copia parametri NumPy, vettoriali o dizionari."""

    if isinstance(params, dict):
        copied = {}
        for key, value in params.items():
            if isinstance(value, list):
                copied[key] = [item.copy() for item in value]
            elif hasattr(value, "copy"):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    if isinstance(params, list):
        return [value.copy() for value in params]

    return params.copy()


def zeros_like_params(params):
    """Crea parametri nulli con la stessa struttura di params."""

    if isinstance(params, dict):
        if "weights" in params:
            return {
                "weights": [np.zeros_like(value) for value in params["weights"]],
                "model_config": params.get("model_config", {}),
            }

        return {key: np.zeros_like(value) for key, value in params.items()}

    if isinstance(params, list):
        return [np.zeros_like(value) for value in params]

    return np.zeros_like(params)


def weighted_average_params(local_params, aggregation_weights):
    """Media pesata di parametri vettoriali o dizionari NumPy."""

    global_params = zeros_like_params(local_params[0])

    if isinstance(global_params, dict) and "weights" in global_params:
        for params, weight in zip(local_params, aggregation_weights):
            for idx in range(len(global_params["weights"])):
                global_params["weights"][idx] += weight * params["weights"][idx]
        return global_params

    if isinstance(global_params, dict):
        for params, weight in zip(local_params, aggregation_weights):
            for key in global_params:
                global_params[key] += weight * params[key]
        return global_params

    if isinstance(global_params, list):
        for params, weight in zip(local_params, aggregation_weights):
            for idx in range(len(global_params)):
                global_params[idx] += weight * params[idx]
        return global_params

    for params, weight in zip(local_params, aggregation_weights):
        global_params += weight * params

    return global_params


def count_model_parameters(params):
    """Conta quanti scalari compongono un modello."""

    if isinstance(params, dict):
        if "weights" in params:
            return int(sum(value.size for value in params["weights"]))
        return int(sum(value.size for value in params.values()))

    if isinstance(params, list):
        return int(sum(value.size for value in params))

    return int(params.size)


def build_keras_neural_network(n_features, model_config):
    """Costruisce il modello Keras usato nel benchmark neurale."""

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(model_config.get("seed", 42))

    model = tf.keras.Sequential(name="federated_disaster_alert_nn")
    model.add(tf.keras.layers.Input(shape=(n_features,)))

    architecture = model_config.get("architecture")

    if architecture:
        for block in architecture:
            block_type = block["type"]

            if block_type == "dense":
                regularizer = tf.keras.regularizers.l2(block.get("l2", model_config.get("l2", 0.0)))
                model.add(
                    tf.keras.layers.Dense(
                        block["units"],
                        activation=block.get("activation"),
                        kernel_regularizer=regularizer,
                    )
                )
            elif block_type == "batch_norm":
                model.add(tf.keras.layers.BatchNormalization())
            elif block_type == "activation":
                model.add(tf.keras.layers.Activation(block.get("name", "relu")))
            elif block_type == "dropout":
                model.add(tf.keras.layers.Dropout(block.get("rate", 0.0)))
            elif block_type == "output":
                model.add(
                    tf.keras.layers.Dense(
                        block.get("units", 1),
                        activation=block.get("activation", "sigmoid"),
                    )
                )
            else:
                raise ValueError(f"Unsupported Keras architecture block: {block_type}")
    else:
        regularizer = tf.keras.regularizers.l2(model_config.get("l2", 0.0))
        dropout_rates = model_config.get("dropout_rates", [])

        for idx, units in enumerate(model_config.get("hidden_layers", [128, 64, 32])):
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=None,
                    kernel_regularizer=regularizer,
                )
            )

            if model_config.get("use_batch_norm", True):
                model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Activation(model_config.get("activation", "swish")))

            dropout_rate = dropout_rates[idx] if idx < len(dropout_rates) else 0.0
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    if hasattr(tf.keras.optimizers, "AdamW"):
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=model_config.get("lr", 0.001),
            weight_decay=model_config.get("weight_decay", 0.0),
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.get("lr", 0.001))

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def keras_params_from_model(model, model_config):
    """Impacchetta pesi Keras e configurazione architetturale."""

    return {
        "weights": model.get_weights(),
        "model_config": {
            "architecture": list(model_config.get("architecture", [])),
            "hidden_layers": list(model_config.get("hidden_layers", [128, 64, 32])),
            "activation": model_config.get("activation", "swish"),
            "dropout_rates": list(model_config.get("dropout_rates", [])),
            "use_batch_norm": model_config.get("use_batch_norm", True),
            "lr": model_config.get("lr", 0.001),
            "weight_decay": model_config.get("weight_decay", 0.0),
            "l2": model_config.get("l2", 0.0),
            "seed": model_config.get("seed", 42),
        },
    }


def get_keras_weights(params):
    """Estrae la lista pesi da una struttura Keras."""

    if isinstance(params, dict) and "weights" in params:
        return params["weights"]

    return params


def init_keras_neural_network(n_features, model_config):
    """Inizializza una rete Keras e restituisce i suoi pesi."""

    model = build_keras_neural_network(n_features, model_config)
    return keras_params_from_model(model, model_config)


def predict_proba_keras_neural_network(X, params):
    """Predice probabilita' con una rete Keras rappresentata dai suoi pesi."""

    model_config = params.get("model_config", {}) if isinstance(params, dict) else {}
    model = build_keras_neural_network(X.shape[1], model_config)
    model.set_weights(get_keras_weights(params))
    y_prob = model.predict(X, verbose=0).reshape(-1)

    return y_prob


def train_keras_neural_network(
    X,
    y,
    model_config,
    params_init=None,
    epochs=None,
):
    """Addestra una rete neurale Keras e restituisce i pesi addestrati."""

    model = build_keras_neural_network(X.shape[1], model_config)

    if params_init is not None:
        model.set_weights(get_keras_weights(copy_params(params_init)))

    callbacks = []
    if params_init is None and model_config.get("early_stopping", False):
        tf = import_tensorflow()
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=model_config.get("early_stopping_patience", 6),
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X,
        y,
        epochs=epochs if epochs is not None else model_config.get("epochs", 25),
        batch_size=model_config.get("batch_size", 256),
        verbose=model_config.get("verbose", 0),
        shuffle=True,
        validation_split=model_config.get("validation_split", 0.0) if params_init is None else 0.0,
        callbacks=callbacks,
        class_weight=binary_class_weight_dict(y, model_config.get("class_weight")),
    )

    losses = [float(value) for value in history.history.get("loss", [])]
    return keras_params_from_model(model, model_config), losses


def neural_network_forward(X, params):
    """Forward pass della rete neurale."""

    z1 = X.dot(params["W1"]) + params["b1"]
    a1 = np.tanh(z1)
    logits = a1.dot(params["W2"]) + params["b2"]
    y_prob = sigmoid(logits)

    return z1, a1, y_prob


def predict_proba_neural_network(X, params):
    """Predice probabilita' con la rete neurale."""

    _, _, y_prob = neural_network_forward(X, params)
    return y_prob


def train_neural_network(
    X,
    y,
    params_init=None,
    hidden_units=12,
    lr=0.08,
    epochs=200,
    l2=0.0005,
    seed=42,
):
    """Addestra una piccola rete neurale con un hidden layer tanh."""

    n_features = X.shape[1]

    if params_init is None:
        params = init_neural_network(n_features, hidden_units=hidden_units, seed=seed)
    else:
        params = copy_params(params_init)

    losses = []
    n_samples = len(y)

    for _ in range(epochs):
        _, a1, y_prob = neural_network_forward(X, params)

        dlogits = (y_prob - y) / n_samples

        grad_W2 = a1.T.dot(dlogits) + l2 * params["W2"]
        grad_b2 = np.sum(dlogits)

        da1 = np.outer(dlogits, params["W2"])
        dz1 = da1 * (1 - a1**2)

        grad_W1 = X.T.dot(dz1) + l2 * params["W1"]
        grad_b1 = np.sum(dz1, axis=0)

        params["W2"] -= lr * grad_W2
        params["b2"] -= lr * grad_b2
        params["W1"] -= lr * grad_W1
        params["b1"] -= lr * grad_b1

        losses.append(binary_cross_entropy(y, y_prob))

    return params, losses


def init_model_params(model_config, n_features):
    """Inizializza i parametri in base alla configurazione del modello."""

    if model_config["kind"] == "logistic":
        return np.zeros(n_features + 1)

    if model_config["kind"] == "neural_network":
        return init_neural_network(
            n_features,
            hidden_units=model_config.get("hidden_units", 12),
            seed=model_config.get("seed", 42),
        )

    if model_config["kind"] == KERAS_MODEL_KIND:
        return init_keras_neural_network(n_features, model_config)

    raise ValueError(f"Unsupported model kind: {model_config['kind']}")


def train_model(X, y, model_config, params_init=None, epochs=None):
    """Wrapper comune per addestrare i modelli supportati."""

    model_kind = model_config["kind"]
    train_epochs = epochs if epochs is not None else model_config.get("epochs", 200)

    if model_kind == "logistic":
        return train_logistic_regression(
            X,
            y,
            theta_init=params_init,
            lr=model_config.get("lr", 0.15),
            epochs=train_epochs,
            l2=model_config.get("l2", 0.001),
            class_weight=model_config.get("class_weight"),
        )

    if model_kind == "neural_network":
        return train_neural_network(
            X,
            y,
            params_init=params_init,
            hidden_units=model_config.get("hidden_units", 12),
            lr=model_config.get("lr", 0.08),
            epochs=train_epochs,
            l2=model_config.get("l2", 0.0005),
            seed=model_config.get("seed", 42),
        )

    if model_kind == KERAS_MODEL_KIND:
        return train_keras_neural_network(
            X,
            y,
            model_config,
            params_init=params_init,
            epochs=train_epochs,
        )

    raise ValueError(f"Unsupported model kind: {model_kind}")