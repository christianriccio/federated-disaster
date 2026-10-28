"""Optional physical-CSV feature mapping adapter inherited from the historical repository.
This interface is not validated on institutional datasets in the paper.
"""
import numpy as np
from config import FEATURE_RANGES, FEATURES, TARGET_COLUMN

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

