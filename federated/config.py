"""Costanti condivise dalla simulazione di Federated Learning."""

# SEED e' il seme globale di riferimento.
# Un seed rende riproducibili gli esperimenti: a parita' di codice e seed,
# NumPy genera la stessa sequenza di numeri pseudo-casuali.
SEED = 42

# FEATURES contiene i nomi delle colonne della matrice X.
# L'ordine deve restare uguale all'ordine con cui le feature vengono unite
# nella funzione generate_node_data.
FEATURES = [
    "rainfall_1h",             # pioggia nell'ultima ora
    "rainfall_3h",             # pioggia nelle ultime 3 ore
    "rainfall_24h",            # pioggia nelle ultime 24 ore
    "river_level",             # livello del fiume
    "river_delta_1h",          # variazione del livello del fiume nell'ultima ora
    "soil_saturation",         # saturazione del suolo
    "event_reports",           # segnalazioni/eventi riportati
    "road_closures",           # chiusure stradali
    "sensor_anomaly",          # anomalie rilevate dai sensori
    "territorial_risk_class",  # classe di rischio del territorio
]

# Range fisici/operativi usati per portare dataset reali nello stesso spazio
# numerico dei dati sintetici. La normalizzazione condivisa e' importante:
# se i nodi usassero scale diverse, i vettori dei pesi non sarebbero
# confrontabili ne' aggregabili in modo metodologicamente pulito.
FEATURE_RANGES = {
    "rainfall_1h": (0.0, 80.0),
    "rainfall_3h": (0.0, 160.0),
    "rainfall_24h": (0.0, 300.0),
    "river_level": (0.0, 8.0),
    "river_delta_1h": (-2.0, 2.0),
    "soil_saturation": (0.0, 100.0),
    "event_reports": (0.0, 500.0),
    "road_closures": (0.0, 100.0),
    "sensor_anomaly": (0.0, 1.0),
    "territorial_risk_class": (0.0, 1.0),
}

TARGET_COLUMN = "alert"

# Intercetta globale usata nel generatore sintetico. Valori piu' negativi
# rendono la classe positiva piu' rara; valori meno negativi la rendono piu'
# frequente. Serve a controllare il positive rate della simulazione.
SYNTHETIC_BASE_LOGIT = -8.00

# Configurazione dei nodi sintetici. Il source_weight rappresenta un fattore
# di affidabilita'/rilevanza istituzionale usato dall'orchestratore.
NODE_SPECS = [
    {
        "name": "Nodo A - Fonte nazionale / INGV-DPC",
        "node_type": "national",
        "n_samples": 60000,
        "seed": 101,
        "source_weight": 1.80,
    },
    {
        "name": "Nodo B - Regione / Centro Funzionale",
        "node_type": "regional",
        "n_samples": 45000,
        "seed": 202,
        "source_weight": 1.35,
    },
    {
        "name": "Nodo C - Comune / COC",
        "node_type": "local",
        "n_samples": 30000,
        "seed": 303,
        "source_weight": 1.00,
    },
    {
        "name": "Nodo D - Sensori / Infrastrutture critiche",
        "node_type": "sensor",
        "n_samples": 50000,
        "seed": 404,
        "source_weight": 1.20,
    },
    {
        "name": "Nodo E - Segnalazioni cittadini",
        "node_type": "citizen",
        "n_samples": 25000,
        "seed": 505,
        "source_weight": 0.55,
    },
]

TEST_SIZE = 0.25

LOGISTIC_CONFIG = {
    "name": "Logistic Regression",
    "kind": "logistic",
    "lr": 0.12,
    "epochs": 900,
    "federated_rounds": 60,
    "local_epochs": 18,
    "l2": 0.001,
    "class_weight": "balanced",
}

NEURAL_NETWORK_CONFIG = {
    "name": "Keras/TensorFlow Deep Neural Network",
    "kind": "keras_neural_network",
    "architecture": [
        {"type": "dense", "units": 128, "l2": 0.0001},
        {"type": "batch_norm"},
        {"type": "activation", "name": "swish"},
        {"type": "dropout", "rate": 0.20},
        {"type": "dense", "units": 64, "l2": 0.0001},
        {"type": "batch_norm"},
        {"type": "activation", "name": "swish"},
        {"type": "dropout", "rate": 0.15},
        {"type": "dense", "units": 32, "l2": 0.0001},
        {"type": "batch_norm"},
        {"type": "activation", "name": "swish"},
        {"type": "dropout", "rate": 0.10},
        {"type": "output", "units": 1, "activation": "sigmoid"},
    ],
    "lr": 0.0008,
    "weight_decay": 0.0001,
    "epochs": 40,
    "federated_rounds": 25,
    "local_epochs": 3,
    "batch_size": 512,
    "l2": 0.0001,
    "class_weight": "balanced",
    "early_stopping": True,
    "early_stopping_patience": 6,
    "validation_split": 0.12,
    "seed": 777,
    "verbose": 0,
}

THRESHOLD = 0.5

OUTPUT_DIR = "outputs"

MULTI_SEED_VALUES = [11, 22, 33]
