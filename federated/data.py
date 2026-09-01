"""Generazione, preprocessing e costruzione dei nodi federati."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import FEATURE_RANGES, FEATURES, NODE_SPECS, SYNTHETIC_BASE_LOGIT, TARGET_COLUMN, TEST_SIZE
from model import sigmoid


@dataclass
class FederatedNode:
    """Contenitore dei dati associati a un singolo nodo federato.

    @dataclass genera automaticamente:
    - __init__, cioe' il costruttore;
    - __repr__, cioe' la rappresentazione testuale;
    - __eq__, cioe' il confronto con ==.
    """

    # Nome descrittivo del nodo, per esempio "Nodo A - Regione".
    name: str

    # Matrice delle feature usate per addestrare il modello locale.
    X_train: np.ndarray

    # Etichette 0/1 corrispondenti a X_train.
    y_train: np.ndarray

    # Matrice delle feature usate per valutare il modello.
    X_test: np.ndarray

    # Etichette 0/1 corrispondenti a X_test.
    y_test: np.ndarray

    # Tipo di fonte: nazionale, regionale, locale, sensori, cittadini, ecc.
    node_type: str

    # Fattore configurabile usato dall'orchestratore nell'aggregazione.
    source_weight: float = 1.0


@dataclass
class SourceDataSchema:
    """Descrive come trasformare un CSV reale in un nodo federato.

    feature_mapping mappa le feature canoniche del modello verso le colonne
    reali della fonte. Esempio:
    {
        "rainfall_24h": "prec_24h_mm",
        "event_reports": ["calls_112", "app_reports"],
        "sensor_anomaly": {"column": "anomaly_score", "scale": 0.01},
        "territorial_risk_class": 0.5,
    }
    """

    path: str
    name: str
    node_type: str
    source_weight: float = 1.0
    seed: int = 42
    target_column: str = TARGET_COLUMN
    feature_mapping: Optional[dict] = None
    test_size: float = TEST_SIZE


class SharedFeatureProcessor:
    """Preprocessing condiviso tra nodi sintetici e dataset reali.

    La rete federata puo' aggregare correttamente i parametri solo se ogni nodo
    usa lo stesso ordine di feature e la stessa scala numerica.
    """

    def __init__(self, feature_names=None, feature_ranges=None):
        self.feature_names = list(feature_names or FEATURES)
        self.feature_ranges = dict(feature_ranges or FEATURE_RANGES)

    def resolve_feature_values(self, columns, mapping_rule, n_rows):
        """Risolve una regola di mapping in un vettore numerico raw."""

        if isinstance(mapping_rule, str):
            if mapping_rule not in columns:
                raise ValueError(f"Missing mapped source column: {mapping_rule}")
            return np.asarray(columns[mapping_rule], dtype=float)

        if isinstance(mapping_rule, (int, float)):
            return np.full(n_rows, float(mapping_rule))

        if isinstance(mapping_rule, (list, tuple)):
            if not mapping_rule:
                raise ValueError("Empty list in feature mapping")

            values = [self.resolve_feature_values(columns, item, n_rows) for item in mapping_rule]
            return np.mean(np.column_stack(values), axis=1)

        if isinstance(mapping_rule, dict):
            if "constant" in mapping_rule:
                values = np.full(n_rows, float(mapping_rule["constant"]))
            elif "column" in mapping_rule:
                values = self.resolve_feature_values(columns, mapping_rule["column"], n_rows)
            elif "columns" in mapping_rule:
                values = self.resolve_feature_values(columns, mapping_rule["columns"], n_rows)
            else:
                raise ValueError(f"Unsupported mapping rule: {mapping_rule}")

            scale = float(mapping_rule.get("scale", 1.0))
            offset = float(mapping_rule.get("offset", 0.0))
            return values * scale + offset

        raise ValueError(f"Unsupported mapping rule: {mapping_rule}")

    def canonicalize_columns(self, columns, feature_mapping=None):
        """Porta colonne eterogenee nello schema canonico FEATURES."""

        if not columns:
            raise ValueError("No columns available")

        first_column = next(iter(columns.values()))
        n_rows = len(first_column)
        feature_mapping = feature_mapping or {}
        canonical_columns = {}

        for feature in self.feature_names:
            mapping_rule = feature_mapping.get(feature, feature)
            canonical_columns[feature] = self.resolve_feature_values(columns, mapping_rule, n_rows)

        return canonical_columns

    def transform_raw_columns(self, columns, feature_mapping=None):
        """Converte colonne raw in una matrice normalizzata e ordinata."""

        columns = self.canonicalize_columns(columns, feature_mapping=feature_mapping)
        normalized_columns = []

        for feature in self.feature_names:
            if feature not in columns:
                raise ValueError(f"Missing required feature column: {feature}")

            values = np.asarray(columns[feature], dtype=float)
            min_value, max_value = self.feature_ranges[feature]
            midpoint = (min_value + max_value) / 2

            values = np.where(np.isfinite(values), values, midpoint)

            if max_value == min_value:
                scaled = np.zeros_like(values)
            else:
                scaled = (values - min_value) / (max_value - min_value)

            normalized_columns.append(np.clip(scaled, 0.0, 1.0))

        return np.column_stack(normalized_columns).astype(float)

    def load_csv(self, path, target_column=TARGET_COLUMN, feature_mapping=None):
        """Carica un CSV reale con intestazioni e restituisce X, y normalizzati."""

        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")

        if data.ndim == 0:
            data = np.array([data], dtype=data.dtype)

        if target_column not in data.dtype.names:
            raise ValueError(f"Missing target column: {target_column}")

        columns = {name: data[name] for name in data.dtype.names}
        X = self.transform_raw_columns(columns, feature_mapping=feature_mapping)
        y = np.asarray(columns[target_column], dtype=int)

        return X, y


def train_test_split_local(X, y, test_size=0.25, seed=42):
    """Divide un dataset locale in training set e test set."""

    # Crea un generatore casuale locale, indipendente da altri generatori.
    local_rng = np.random.default_rng(seed)

    # len(y) restituisce il numero di campioni.
    n = len(y)

    # permutation(n) restituisce gli indici da 0 a n-1 in ordine casuale.
    idx = local_rng.permutation(n)

    # Calcola quanti campioni mettere nel test set.
    test_n = int(n * test_size)

    # I primi indici mescolati vanno al test set.
    test_idx = idx[:test_n]

    # Gli indici rimanenti vanno al training set.
    train_idx = idx[test_n:]

    # L'indicizzazione NumPy X[train_idx] seleziona le righe corrispondenti.
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def generate_node_data(node_type, n_samples, seed=42):
    """Genera dati sintetici per un tipo specifico di nodo federato."""

    # Ogni nodo usa il proprio generatore per rendere i dati riproducibili.
    local_rng = np.random.default_rng(seed)

    label_noise = 0.0

    if node_type == "national":
        rainfall_1h = local_rng.beta(2.1, 5.2, n_samples)
        rainfall_3h = np.clip(rainfall_1h + local_rng.normal(0.10, 0.08, n_samples), 0, 1)
        rainfall_24h = local_rng.beta(2.8, 3.2, n_samples)
        river_level = local_rng.beta(2.3, 2.9, n_samples)
        river_delta_1h = local_rng.beta(2.1, 4.8, n_samples)
        soil_saturation = local_rng.beta(2.7, 2.4, n_samples)
        event_reports = local_rng.beta(1.3, 7.5, n_samples)
        road_closures = local_rng.beta(1.2, 9.0, n_samples)
        sensor_anomaly = local_rng.beta(2.2, 4.5, n_samples)
        territorial_risk_class = local_rng.choice([0.2, 0.5, 0.8], n_samples, p=[0.40, 0.42, 0.18])
        node_bias = -0.03
        label_noise = 0.015

    elif node_type == "regional":
        # beta(a, b, n) genera n valori tra 0 e 1 da una distribuzione Beta.
        rainfall_1h = local_rng.beta(2.0, 5.0, n_samples)

        # normal(media, deviazione, n) genera rumore gaussiano.
        # clip(..., 0, 1) mantiene i valori normalizzati tra 0 e 1.
        rainfall_3h = np.clip(rainfall_1h + local_rng.normal(0.10, 0.10, n_samples), 0, 1)

        rainfall_24h = local_rng.beta(2.5, 3.5, n_samples)
        river_level = local_rng.beta(2.0, 3.0, n_samples)
        river_delta_1h = local_rng.beta(1.8, 5.0, n_samples)
        soil_saturation = local_rng.beta(2.5, 2.5, n_samples)
        event_reports = local_rng.beta(1.2, 8.0, n_samples)
        road_closures = local_rng.beta(1.1, 10.0, n_samples)
        sensor_anomaly = local_rng.beta(1.3, 7.0, n_samples)

        # choice sceglie valori discreti secondo le probabilita' p.
        territorial_risk_class = local_rng.choice([0.2, 0.5, 0.8], n_samples, p=[0.45, 0.40, 0.15])
        node_bias = -0.10
        label_noise = 0.025

    elif node_type == "local":
        rainfall_1h = local_rng.beta(1.8, 5.5, n_samples)
        rainfall_3h = np.clip(rainfall_1h + local_rng.normal(0.08, 0.15, n_samples), 0, 1)
        rainfall_24h = local_rng.beta(2.0, 4.0, n_samples)
        river_level = local_rng.beta(1.8, 3.8, n_samples)
        river_delta_1h = local_rng.beta(1.5, 4.5, n_samples)
        soil_saturation = local_rng.beta(2.0, 3.0, n_samples)
        event_reports = local_rng.beta(2.5, 4.0, n_samples)
        road_closures = local_rng.beta(2.0, 5.0, n_samples)
        sensor_anomaly = local_rng.beta(1.2, 8.0, n_samples)
        territorial_risk_class = local_rng.choice([0.2, 0.5, 0.8], n_samples, p=[0.55, 0.30, 0.15])
        node_bias = 0.05
        label_noise = 0.040

    elif node_type == "sensor":
        rainfall_1h = local_rng.beta(1.7, 5.5, n_samples)
        rainfall_3h = np.clip(rainfall_1h + local_rng.normal(0.07, 0.12, n_samples), 0, 1)
        rainfall_24h = local_rng.beta(2.0, 4.2, n_samples)
        river_level = local_rng.beta(2.4, 3.0, n_samples)
        river_delta_1h = local_rng.beta(2.0, 4.2, n_samples)
        soil_saturation = local_rng.beta(2.3, 3.0, n_samples)
        event_reports = local_rng.beta(1.1, 9.0, n_samples)
        road_closures = local_rng.beta(1.1, 10.0, n_samples)
        sensor_anomaly = local_rng.beta(2.8, 3.8, n_samples)
        territorial_risk_class = local_rng.choice([0.2, 0.5, 0.8], n_samples, p=[0.50, 0.35, 0.15])
        node_bias = 0.10
        label_noise = 0.030

    elif node_type == "citizen":
        rainfall_1h = local_rng.beta(1.6, 5.8, n_samples)
        rainfall_3h = np.clip(rainfall_1h + local_rng.normal(0.06, 0.18, n_samples), 0, 1)
        rainfall_24h = local_rng.beta(1.8, 4.6, n_samples)
        river_level = local_rng.beta(1.6, 4.0, n_samples)
        river_delta_1h = local_rng.beta(1.4, 4.8, n_samples)
        soil_saturation = local_rng.beta(1.8, 3.4, n_samples)
        event_reports = local_rng.beta(3.2, 3.6, n_samples)
        road_closures = local_rng.beta(2.6, 4.2, n_samples)
        sensor_anomaly = local_rng.beta(1.1, 8.5, n_samples)
        territorial_risk_class = local_rng.choice([0.2, 0.5, 0.8], n_samples, p=[0.60, 0.28, 0.12])
        node_bias = 0.12
        label_noise = 0.095

    else:
        # ValueError segnala che il parametro ricevuto non e' valido.
        raise ValueError("node_type must be one of: national, regional, local, sensor, citizen")

    # column_stack affianca vettori come colonne di una matrice.
    # Ogni riga di X e' un campione; ogni colonna corrisponde a FEATURES.
    X = np.column_stack([
        rainfall_1h,
        rainfall_3h,
        rainfall_24h,
        river_level,
        river_delta_1h,
        soil_saturation,
        event_reports,
        road_closures,
        sensor_anomaly,
        territorial_risk_class,
    ])

    # logit e' il punteggio prima della sigmoid.
    # I termini moltiplicativi modellano interazioni tra variabili.
    logit = (
        SYNTHETIC_BASE_LOGIT
        + 0.80 * rainfall_1h
        + 1.10 * rainfall_3h
        + 1.80 * rainfall_24h
        + 1.70 * river_level
        + 1.40 * river_delta_1h
        + 1.60 * soil_saturation
        + 1.20 * event_reports
        + 1.00 * road_closures
        + 1.50 * sensor_anomaly
        + 1.20 * territorial_risk_class
        + 1.80 * rainfall_24h * soil_saturation
        + 1.20 * river_level * river_delta_1h
        + node_bias
    )

    # La sigmoid converte il logit in probabilita'.
    probability = sigmoid(logit)

    # binomial(1, p, n) genera classi Bernoulli: 1 con probabilita' p, altrimenti 0.
    y = local_rng.binomial(1, probability, n_samples)

    if label_noise > 0:
        noisy_idx = local_rng.random(n_samples) < label_noise
        y[noisy_idx] = 1 - y[noisy_idx]

    # astype assicura tipi numerici espliciti per feature ed etichette.
    return X.astype(float), y.astype(int)


def create_federated_nodes(node_specs=None, test_size=TEST_SIZE):
    """Crea i nodi federati della simulazione."""

    node_specs = node_specs or NODE_SPECS
    nodes = []

    for spec in node_specs:
        name = spec["name"]
        node_type = spec["node_type"]
        n_samples = spec["n_samples"]
        seed = spec["seed"]
        source_weight = spec.get("source_weight", 1.0)

        # Genera il dataset locale del nodo.
        X, y = generate_node_data(node_type, n_samples, seed=seed)

        # Divide il dataset locale in train e test.
        X_train, X_test, y_train, y_test = train_test_split_local(X, y, test_size=test_size, seed=seed)

        # Crea il contenitore del nodo e lo aggiunge alla lista.
        nodes.append(
            FederatedNode(
                name=name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                node_type=node_type,
                source_weight=source_weight,
            )
        )

    return nodes


def create_node_from_csv(
    path,
    name,
    node_type,
    source_weight=1.0,
    seed=42,
    test_size=TEST_SIZE,
    target_column=TARGET_COLUMN,
    feature_mapping=None,
    processor=None,
):
    """Crea un nodo federato partendo da un CSV reale.

    Il CSV puo' avere colonne gia' canoniche oppure colonne specifiche della
    fonte, descritte tramite feature_mapping.
    """

    processor = processor or SharedFeatureProcessor()
    X, y = processor.load_csv(
        path,
        target_column=target_column,
        feature_mapping=feature_mapping,
    )
    X_train, X_test, y_train, y_test = train_test_split_local(X, y, test_size=test_size, seed=seed)

    return FederatedNode(
        name=name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        node_type=node_type,
        source_weight=source_weight,
    )


def create_nodes_from_csv_schemas(schemas, processor=None):
    """Crea piu' nodi reali da schemi eterogenei.

    Ogni schema puo' essere una SourceDataSchema oppure un dizionario con le
    stesse chiavi. Questo permette di mischiare fonti nazionali, regionali,
    comunali, sensori e segnalazioni con nomi colonna diversi, mantenendo una
    matrice finale X coerente per tutti i nodi.
    """

    processor = processor or SharedFeatureProcessor()
    nodes = []

    for schema in schemas:
        if isinstance(schema, dict):
            schema = SourceDataSchema(**schema)

        nodes.append(
            create_node_from_csv(
                path=schema.path,
                name=schema.name,
                node_type=schema.node_type,
                source_weight=schema.source_weight,
                seed=schema.seed,
                test_size=schema.test_size,
                target_column=schema.target_column,
                feature_mapping=schema.feature_mapping,
                processor=processor,
            )
        )

    return nodes
