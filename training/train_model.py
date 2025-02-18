#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import joblib

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ─────────────── Konfiguration der bekannten Features ───────────────
# Pflichtspalten (müssen vorhanden sein)
REQUIRED_FEATURES = ["result_bin", "duration_sec", "error_count", "commits_count"]

# Spalten, die komplett entfernt werden sollen
REMOVED_FEATURES = ["build_number", "build_url", "parameters"]

# Bekannte numerische Features
KNOWN_NUMERIC = [
    "duration_sec", "commits_count", "estimated_duration_sec", "commit_authors_count",
    "total_commit_msg_length", "culprits_count", "culprit_ratio", "error_count",
    "build_year", "duration_diff", "duration_ratio"
]

# Bekannte kategoriale Features
KNOWN_CATEGORICAL = [
    "built_on", "change_set_kind", "executor_name", "trigger_types",
    "build_weekday", "build_month"
]

# Bekannte zeitbasierte Features
KNOWN_TIME = ["build_date", "build_time", "build_hour"]


# ─────────────── Custom Transformer für Zeitfeatures ───────────────
class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer, der:
      - Eine Datumsspalte (z.B. build_date) in Jahr, Monat, Tag und Wochentag zerlegt.
      - Zyklische Transformation (sin/cos) für Spalten wie build_time bzw. build_hour durchführt.
    Die Originalspalten werden entfernt.
    """

    def __init__(self, date_cols=None, cyclical_cols=None):
        self.date_cols = date_cols if date_cols is not None else []
        self.cyclical_cols = cyclical_cols if cyclical_cols is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Verarbeite Datumsspalten
        for col in self.date_cols:
            if col in X.columns:
                try:
                    X[col] = pd.to_datetime(X[col], errors='coerce', dayfirst=True)
                    X[f"{col}_year"] = X[col].dt.year
                    X[f"{col}_month"] = X[col].dt.month
                    X[f"{col}_day"] = X[col].dt.day
                    X[f"{col}_weekday"] = X[col].dt.weekday
                    X.drop(columns=[col], inplace=True)
                except Exception as e:
                    logging.warning("Fehler beim Verarbeiten der Datumsspalte '%s': %s", col, e)
        # Verarbeite zyklische Zeitspalten
        for col in self.cyclical_cols:
            if col in X.columns:
                try:
                    # Falls die Spalte nicht numerisch ist, versuche sie zu parsen und die Stunde zu extrahieren
                    if not np.issubdtype(X[col].dtype, np.number):
                        X[col] = pd.to_datetime(X[col], errors='coerce').dt.hour
                    # Annahme: Werte im Bereich [0,23]
                    X[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / 24)
                    X[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / 24)
                    X.drop(columns=[col], inplace=True)
                except Exception as e:
                    logging.warning("Fehler bei der zyklischen Transformation der Spalte '%s': %s", col, e)
        return X


# ─────────────── Modellwahl ───────────────
def get_model_by_name(model_name: str):
    model_name_lower = model_name.lower()
    if model_name_lower == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name_lower == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    elif model_name_lower == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    elif model_name_lower == "xgboost" and HAS_XGBOOST:
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        logging.warning("Unbekanntes oder nicht verfügbares Modell '%s', verwende RandomForest.", model_name)
        return RandomForestClassifier(n_estimators=100, random_state=42)


# ─────────────── Argument Parser ───────────────
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid ML-Trainingsskript")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Pfad zur Trainings-CSV-Datei.")
    parser.add_argument("--model_name", type=str, default="random_forest",
                        help="Wählbares Modell (random_forest, gradient_boosting, logistic_regression, xgboost).")
    parser.add_argument("--output_model", type=str, default="model.pkl",
                        help="Zieldatei für das trainierte Modell.")
    return parser.parse_args()


# ─────────────── CSV-Daten laden ───────────────
def load_data(input_csv: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(input_csv)
        logging.info("Daten aus '%s' erfolgreich geladen. Spalten: %s", input_csv, list(data.columns))
        return data
    except Exception as e:
        logging.error("Fehler beim Laden der Datei '%s': %s", input_csv, e)
        sys.exit(1)


# ─────────────── Pipeline-Aufbau ───────────────
def build_training_pipeline(model, numeric_features, categorical_features, time_features):
    # Transformer für Zeitfeatures: build_date wird zerlegt, build_time und build_hour zyklisch transformiert.
    time_transformer = TimeFeaturesExtractor(
        date_cols=["build_date"] if "build_date" in time_features else [],
        cyclical_cols=[col for col in time_features if col in ["build_time", "build_hour"]]
    )
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ], remainder="drop")

    pipeline = Pipeline([
        ("time_transform", time_transformer),
        ("preprocessor", preprocessor),
        ("clf", model)
    ])
    return pipeline


# ─────────────── Hauptfunktion ───────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_arguments()

    # Daten laden
    df = load_data(args.input_csv)

    # Pflichtspalten prüfen
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            logging.error("Fehler: Pflichtspalte '%s' fehlt in den Daten.", col)
            sys.exit(1)

    # Label (Zielvariable) extrahieren und kodieren
    y = df["result_bin"].copy()
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Features (X)
    X = df.drop(columns=["result_bin"])

    # Unerwünschte Spalten entfernen
    for col in REMOVED_FEATURES:
        if col in X.columns:
            logging.info("Entferne Spalte '%s' aus den Trainingsdaten.", col)
            X.drop(columns=[col], inplace=True)

    # Bekannte Features: Erstelle Schnittmengen basierend auf den erwarteten Spalten
    known_numeric = [col for col in KNOWN_NUMERIC if col in X.columns]
    known_categorical = [col for col in KNOWN_CATEGORICAL if col in X.columns]
    known_time = [col for col in KNOWN_TIME if col in X.columns]

    # Alle bisher bekannten Spalten (inkl. Pflichtspalten und zu entfernende) definieren
    known_all = set(KNOWN_NUMERIC + KNOWN_CATEGORICAL + KNOWN_TIME + REMOVED_FEATURES + REQUIRED_FEATURES)
    # Alle übrigen (unbekannten) Spalten
    unknown_features = [col for col in X.columns if col not in known_all]

    unknown_numeric = []
    unknown_categorical = []

    # Unbekannte numerische Spalten prüfen (Korrelation mit result_bin)
    for col in unknown_features:
        if pd.api.types.is_numeric_dtype(X[col]):
            temp = pd.concat([X[col], pd.Series(y, index=X.index)], axis=1).dropna()
            if not temp.empty:
                corr = temp[col].corr(temp.iloc[:, 1])
                if abs(corr) >= 0.1:
                    unknown_numeric.append(col)
                    logging.info("Unbekannte numerische Spalte '%s' aufgenommen (Korrelation=%.3f).", col, corr)
                else:
                    logging.info("Unbekannte numerische Spalte '%s' (Korrelation=%.3f) wird verworfen.", col, corr)
            else:
                logging.info("Unbekannte numerische Spalte '%s' enthält keine gültigen Werte.", col)
        # Unbekannte kategoriale Spalten prüfen (Chi‑Quadrat-Test)
        elif pd.api.types.is_object_dtype(X[col]):
            unique_count = X[col].nunique(dropna=True)
            if unique_count <= 30:
                contingency = pd.crosstab(X[col].fillna("missing"), y)
                try:
                    chi2, p, dof, ex = chi2_contingency(contingency)
                    if p < 0.05:
                        unknown_categorical.append(col)
                        logging.info("Unbekannte kategoriale Spalte '%s' aufgenommen (p=%.3f).", col, p)
                    else:
                        logging.info("Unbekannte kategoriale Spalte '%s' (p=%.3f) wird verworfen.", col, p)
                except Exception as e:
                    logging.warning("Chi‑Quadrat-Test für Spalte '%s' schlug fehl: %s", col, e)
            else:
                logging.info("Unbekannte Spalte '%s' wird als Freitext erkannt und ignoriert.", col)

    # Endgültige Feature-Gruppierung
    final_numeric = known_numeric + unknown_numeric
    final_categorical = known_categorical + unknown_categorical
    final_time = known_time  # nur die bekannten Zeitspalten

    # Nach der Transformation mit TimeFeaturesExtractor entstehen neue numerische Spalten:
    new_time_features = []
    if "build_date" in final_time:
        new_time_features += ["build_date_year", "build_date_month", "build_date_day", "build_date_weekday"]
    if "build_time" in final_time:
        new_time_features += ["build_time_sin", "build_time_cos"]
    if "build_hour" in final_time:
        new_time_features += ["build_hour_sin", "build_hour_cos"]

    final_numeric += new_time_features

    logging.info("Finale numerische Features: %s", final_numeric)
    logging.info("Finale kategoriale Features: %s", final_categorical)
    logging.info("Finale Zeitbasierte Features (werden transformiert): %s", final_time)

    # Modell auswählen
    model = get_model_by_name(args.model_name)

    # Trainingspipeline erstellen
    pipeline = build_training_pipeline(model, final_numeric, final_categorical, final_time)

    # Training und Modell speichern
    try:
        pipeline.fit(X, y)
        output_dir = os.path.dirname(args.output_model)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        joblib.dump(pipeline, args.output_model)
        logging.info("Modell '%s' erfolgreich trainiert und in '%s' gespeichert.", args.model_name, args.output_model)
    except Exception as e:
        logging.error("Fehler beim Training oder Speichern des Modells: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
