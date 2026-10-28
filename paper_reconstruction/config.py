"""Costanti condivise dalla simulazione di Federated Learning."""

# EXPERIMENT_MODE seleziona la configurazione:
# - "quick": debug rapido (pochi campioni, pochi round, 3 seed);
# - "paper": esperimento completo per i risultati del paper.
EXPERIMENT_MODE = "paper"

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

# --- Generatore sintetico -------------------------------------------------
# Il target y=1 rappresenta una escalation operativa/condizione di allerta
# rilevante in una finestra temporale breve. La calibrazione produce un
# positive rate globale di circa il 7-13% per nodo e un ceiling teorico
# (classificatore Bayes-ottimo) di ROC-AUC ~0.85 globale, con qualita'
# decrescente lungo la filiera: nazionale ~0.92, cittadini ~0.70.
# Questo gradiente di qualita' e' coerente con i source_weight assegnati.

# Intercetta globale del generatore. Valori piu' negativi rendono la classe
# positiva piu' rara; valori meno negativi la rendono piu' frequente.
SYNTHETIC_BASE_LOGIT = -20.0

# Coefficienti del logit generativo, nello stesso ordine di FEATURES.
SYNTHETIC_LOGIT_COEFFS = {
    "rainfall_1h": 2.4,
    "rainfall_3h": 3.3,
    "rainfall_24h": 5.4,
    "river_level": 5.1,
    "river_delta_1h": 4.2,
    "soil_saturation": 4.8,
    "event_reports": 3.6,
    "road_closures": 3.0,
    "sensor_anomaly": 4.5,
    "territorial_risk_class": 3.6,
}

# Interazioni moltiplicative (feature_a, feature_b, coefficiente).
SYNTHETIC_INTERACTIONS = [
    ("rainfall_24h", "soil_saturation", 5.4),
    ("river_level", "river_delta_1h", 3.6),
]

# Bias additivo e label noise per tipo di nodo. Il label noise simula la
# diversa affidabilita' delle fonti: bassa per la fonte nazionale, alta per
# le segnalazioni dei cittadini.
NODE_GENERATION = {
    "national": {"node_bias": -0.40, "label_noise": 0.008},
    "regional": {"node_bias": -0.10, "label_noise": 0.012},
    "local":    {"node_bias": 0.35,  "label_noise": 0.018},
    "sensor":   {"node_bias": 0.20,  "label_noise": 0.014},
    "citizen":  {"node_bias": 0.90,  "label_noise": 0.030},
}

# --- Nodi -----------------------------------------------------------------
# Il source_weight rappresenta un fattore di affidabilita'/rilevanza
# istituzionale usato dall'orchestratore nel source-weighted FedAvg.

_NODE_BASE = [
    {"name": "Nodo A - Fonte nazionale / INGV-DPC", "node_type": "national", "seed": 101, "source_weight": 1.80},
    {"name": "Nodo B - Regione / Centro Funzionale", "node_type": "regional", "seed": 202, "source_weight": 1.35},
    {"name": "Nodo C - Comune / COC", "node_type": "local", "seed": 303, "source_weight": 1.00},
    {"name": "Nodo D - Sensori / Infrastrutture critiche", "node_type": "sensor", "seed": 404, "source_weight": 1.20},
    {"name": "Nodo E - Segnalazioni cittadini", "node_type": "citizen", "seed": 505, "source_weight": 0.55},
]

_PAPER_SAMPLES = {"national": 60000, "regional": 45000, "local": 30000, "sensor": 50000, "citizen": 25000}
_QUICK_SAMPLES = {"national": 6000, "regional": 4500, "local": 3000, "sensor": 5000, "citizen": 2500}

PAPER_NODE_SPECS = [dict(spec, n_samples=_PAPER_SAMPLES[spec["node_type"]]) for spec in _NODE_BASE]
QUICK_NODE_SPECS = [dict(spec, n_samples=_QUICK_SAMPLES[spec["node_type"]]) for spec in _NODE_BASE]

# Frazioni dello split locale per nodo. La validation serve ESCLUSIVAMENTE
# alla scelta delle soglie operative; il test set resta intoccato fino alla
# valutazione finale.
VAL_SIZE = 0.15
TEST_SIZE = 0.25

PAPER_LOGISTIC_CONFIG = {
    "name": "Logistic Regression",
    "kind": "logistic",
    "lr": 0.12,
    "epochs": 900,
    "federated_rounds": 60,
    "local_epochs": 18,
    "l2": 0.001,
    "class_weight": "balanced",
}

QUICK_LOGISTIC_CONFIG = dict(
    PAPER_LOGISTIC_CONFIG,
    epochs=250,
    federated_rounds=12,
    local_epochs=8,
)

# Nota architetturale: la rete NON usa BatchNormalization. FedAvg media anche
# le statistiche mobili dei layer BN, cosa nota per essere problematica con
# nodi non-IID (cfr. letteratura FedBN). Meglio evitarla nel prototipo.
PAPER_NEURAL_NETWORK_CONFIG = {
    "name": "Keras/TensorFlow Deep Neural Network",
    "kind": "keras_neural_network",
    "architecture": [
        {"type": "dense", "units": 128, "l2": 0.0001, "activation": "swish"},
        {"type": "dropout", "rate": 0.20},
        {"type": "dense", "units": 64, "l2": 0.0001, "activation": "swish"},
        {"type": "dropout", "rate": 0.15},
        {"type": "dense", "units": 32, "l2": 0.0001, "activation": "swish"},
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

QUICK_NEURAL_NETWORK_CONFIG = dict(
    PAPER_NEURAL_NETWORK_CONFIG,
    epochs=10,
    federated_rounds=5,
    local_epochs=2,
)

PAPER_MULTI_SEED_VALUES = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
QUICK_MULTI_SEED_VALUES = [11, 22, 33]

# --- Sensitivity analysis sui source weights -------------------------------
# Scenari usati per verificare quanto il source-weighted FedAvg dipenda dai
# pesi scelti. "inverted" assegna i pesi baseline in ordine di rango inverso.
SOURCE_WEIGHT_SCENARIOS = {
    "uniform": {"national": 1.00, "regional": 1.00, "local": 1.00, "sensor": 1.00, "citizen": 1.00},
    "baseline": {"national": 1.80, "regional": 1.35, "local": 1.00, "sensor": 1.20, "citizen": 0.55},
    "extreme": {"national": 3.00, "regional": 1.80, "local": 1.00, "sensor": 1.50, "citizen": 0.25},
    "inverted": {"national": 0.55, "regional": 1.00, "local": 1.35, "sensor": 1.20, "citizen": 1.80},
}

# --- Selezione della configurazione attiva ---------------------------------

if EXPERIMENT_MODE == "paper":
    NODE_SPECS = PAPER_NODE_SPECS
    LOGISTIC_CONFIG = PAPER_LOGISTIC_CONFIG
    NEURAL_NETWORK_CONFIG = PAPER_NEURAL_NETWORK_CONFIG
    MULTI_SEED_VALUES = PAPER_MULTI_SEED_VALUES
elif EXPERIMENT_MODE == "quick":
    NODE_SPECS = QUICK_NODE_SPECS
    LOGISTIC_CONFIG = QUICK_LOGISTIC_CONFIG
    NEURAL_NETWORK_CONFIG = QUICK_NEURAL_NETWORK_CONFIG
    MULTI_SEED_VALUES = QUICK_MULTI_SEED_VALUES
else:
    raise ValueError("EXPERIMENT_MODE must be 'paper' or 'quick'")

# Soglia fissa di riferimento (le soglie operative vengono invece selezionate
# sulla validation tramite threshold_search).
THRESHOLD = 0.5

# Recall minima richiesta dalla policy operativa best_operational.
MIN_OPERATIONAL_RECALL = 0.90

OUTPUT_DIR = "outputs"