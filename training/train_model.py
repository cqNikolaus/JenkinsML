#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import joblib
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Wir passen das Label auf "result" an, da deine CSV-Spalte so heißt
LABEL_COLUMN = "result"

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrahiert aus angegebenen Datums-Spalten Komponenten wie Jahr, Monat, Tag und Stunde.
    Die ursprünglichen Datums-Spalten werden dabei entfernt.
    Nur relevant, wenn wir explizit 'date_columns' verwenden.
    """
    def __init__(self, date_cols: List[str] = None):
        self.date_cols = date_cols if date_cols else []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.date_cols:
            if col in X.columns:
                try:
                    # Dayfirst=True, damit 18.11.2024 als 18. November interpretiert wird
                    X[col] = pd.to_datetime(X[col], errors='coerce', dayfirst=True)
                    X[f"{col}_year"] = X[col].dt.year
                    X[f"{col}_month"] = X[col].dt.month
                    X[f"{col}_day"] = X[col].dt.day
                    X[f"{col}_hour"] = X[col].dt.hour
                    X.drop(columns=[col], inplace=True)
                except Exception as e:
                    logging.warning(
                        "Fehler bei der Datumsverarbeitung der Spalte '%s': %s",
                        col, e
                    )
        return X

def detect_column_types(df: pd.DataFrame, max_cat_unique: int = 30) -> Tuple[List[str], List[str], List[str]]:
    """
    Ermittelt numerische, kategoriale und Datums-Spalten im DataFrame.
    Kategoriale Spalten werden anhand der Anzahl eindeutiger Werte bestimmt.
    """
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    date_cols = list(df.select_dtypes(include=["datetime"]).columns)
    categorical_cols = []

    # Erkennen von kategorialen Spalten (String + Anzahl eindeutiger Werte)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique(dropna=True) <= max_cat_unique:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, date_cols

def get_model_by_name(model_name: str):
    """
    Gibt ein ML-Modell basierend auf dem übergebenen Namen zurück.
    Bei unbekannten oder nicht verfügbaren Modellen wird RandomForest als Standard genutzt.
    """
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
        logging.warning(
            "Unbekanntes oder nicht verfügbares Modell '%s', verwende RandomForest.",
            model_name
        )
        return RandomForestClassifier(n_estimators=100, random_state=42)

def parse_arguments() -> argparse.Namespace:
    """
    Parst die CLI-Argumente und gibt das Argument-Objekt zurück.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Pfad zur Trainings-CSV-Datei.")
    parser.add_argument("--model_name", type=str, default="random_forest",
                        help="Wählbares Modell (random_forest, gradient_boosting, logistic_regression, xgboost).")
    parser.add_argument("--output_model", type=str, default="model.pkl",
                        help="Zieldatei für das trainierte Modell.")
    parser.add_argument("--max_cat_unique", type=int, default=30,
                        help="Max. Anzahl eindeutiger Werte, damit String-Spalte als kategorisch gilt.")
    parser.add_argument("--date_columns", type=str, default="",
                        help="Kommagetrennte Liste von Spalten, die als Datum geparst werden sollen (Format unbekannt).")
    return parser.parse_args()

def load_data(input_csv: str) -> pd.DataFrame:
    """
    Lädt die CSV-Daten in einen DataFrame und entfernt ggf. erste oder letzte Zeile,
    falls diese offensichtlich keine gültigen Daten enthalten.
    """
    try:
        first_row = pd.read_csv(input_csv, nrows=1).columns.tolist()
        if LABEL_COLUMN not in first_row:
            logging.warning("Die erste Zeile scheint Metadaten zu sein. Ignoriere sie...")
            data = pd.read_csv(input_csv, skiprows=1)
        else:
            data = pd.read_csv(input_csv)

        # Letzte Zeile prüfen, ob Zielspalte gültig ist
        last_row = data.iloc[-1]
        if pd.isna(last_row[LABEL_COLUMN]):
            logging.warning("Die letzte Zeile scheint ungültige Werte zu enthalten. Entferne sie.")
            data = data.iloc[:-1]

        logging.info("Daten aus '%s' erfolgreich geladen.", input_csv)
        logging.info("Erkannte Spalten: %s", list(data.columns))
        return data
    except Exception as e:
        logging.error("Fehler beim Laden der Datei '%s': %s", input_csv, e)
        sys.exit(1)

def build_preprocessor(X: pd.DataFrame, args: argparse.Namespace) -> ColumnTransformer:
    """
    Baut den Preprocessor, der numerische, kategoriale und Datums-Spalten verarbeitet.
    """
    # Falls bestimmte Spalten explizit als Datum geparst werden sollen
    date_cols_arg = [col.strip() for col in args.date_columns.split(",") if col.strip()]
    if date_cols_arg:
        for col in date_cols_arg:
            if col in X.columns:
                # Dayfirst=True für Formate wie 18.11.2024
                X[col] = pd.to_datetime(X[col], errors='ignore', dayfirst=True)

    numeric_cols, categorical_cols, detected_date_cols = detect_column_types(
        X, max_cat_unique=args.max_cat_unique
    )

    # Zusammenführen: explizit angegebene und automatisch erkannte Datumsspalten
    date_cols = list(set(date_cols_arg + detected_date_cols))

    # Date-Pipeline: wandelt Datums-Spalten in (year, month, day, hour) auf
    date_pipeline = Pipeline([
        ("date_extractor", DateFeatureExtractor(date_cols=date_cols))
    ])

    # Numerische Pipeline: imputen, dann skalieren
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Kategoriale Pipeline: fehlende Werte füllen, dann One-Hot
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("date", date_pipeline, date_cols),
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder='drop')

    return preprocessor

def build_training_pipeline(X: pd.DataFrame, args: argparse.Namespace) -> Pipeline:
    """
    Erstellt die vollständige Trainings-Pipeline (Vorverarbeitung + Modell).
    """
    preprocessor = build_preprocessor(X, args)
    model = get_model_by_name(args.model_name)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", model)
    ])

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_arguments()

    # Daten laden
    df = load_data(args.input_csv)

    # Prüfen, ob die Zielspalte existiert
    if LABEL_COLUMN not in df.columns:
        logging.error("Fehler: Spalte '%s' nicht vorhanden.", LABEL_COLUMN)
        sys.exit(1)

    # Label (y)
    y = df[LABEL_COLUMN].copy()

    # Features (X)
    X = df.drop(columns=[LABEL_COLUMN])

    # Unerwünschte / eindeutig unbrauchbare Spalten entfernen
    # Du kannst hier beliebig erweitern, wenn du mehr entfernen willst
    UNWANTED_COLS = [
        "build_number",  # fortlaufende Nummer
        "build_url",     # eindeutige URL pro Build
        "parameters",    # komplexes JSON, ohne Parsen nicht hilfreich
        "build_date",    # da already build_* Zeitfeatures vorhanden
        "build_time"     # da already build_* Zeitfeatures vorhanden
    ]
    for col in UNWANTED_COLS:
        if col in X.columns:
            logging.info("Entferne Spalte '%s' aus den Trainingsdaten.", col)
            X.drop(columns=[col], inplace=True)

    # Trainingspipeline erstellen
    training_pipeline = build_training_pipeline(X, args)

    # Training + Modell speichern
    try:
        training_pipeline.fit(X, y)
        output_dir = os.path.dirname(args.output_model)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        joblib.dump(training_pipeline, args.output_model)
        logging.info(
            "Modell '%s' erfolgreich trainiert und in '%s' gespeichert.",
            args.model_name, args.output_model
        )
    except Exception as e:
        logging.error("Fehler beim Training oder Speichern des Modells: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
