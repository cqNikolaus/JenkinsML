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

# Konstante für die Zielspalte
LABEL_COLUMN = "result_bin"


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrahiert Jahr, Monat, Tag und Stunde aus angegebenen Datums-Spalten.
    Ursprüngliche Datums-Spalten werden entfernt.
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
                    X[col] = pd.to_datetime(X[col], errors='coerce')
                    X[f"{col}_year"] = X[col].dt.year
                    X[f"{col}_month"] = X[col].dt.month
                    X[f"{col}_day"] = X[col].dt.day
                    X[f"{col}_hour"] = X[col].dt.hour
                    X.drop(columns=[col], inplace=True)
                except Exception as e:
                    logging.warning("Fehler bei der Datumsverarbeitung der Spalte '%s': %s", col, e)
        return X


def detect_column_types(df: pd.DataFrame, max_cat_unique: int = 30) -> Tuple[List[str], List[str], List[str]]:
    """
    Ermittelt numerische, kategoriale und Datums-Spalten im DataFrame.
    Kategoriale Spalten werden anhand der Anzahl eindeutiger Werte bestimmt.
    """
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    date_cols = list(df.select_dtypes(include=["datetime"]).columns)
    categorical_cols = []

    # Erkennen von kategorialen Spalten anhand des Objekttyps und eindeutiger Werte
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
        logging.warning("Unbekanntes oder nicht verfügbares Modell '%s', verwende RandomForest.", model_name)
        return RandomForestClassifier(n_estimators=100, random_state=42)


def parse_arguments() -> argparse.Namespace:
    """
    Parst die CLI-Argumente und gibt das Argument-Objekt zurück.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Pfad zur Trainings-CSV-Datei.")
    parser.add_argument("--model_name", type=str, default="random_forest",
                        help="Wählbares Modell (random_forest, gradient_boosting, logistic_regression, xgboost).")
    parser.add_argument("--output_model", type=str, default="model.pkl",
                        help="Zieldatei für das trainierte Modell.")
    parser.add_argument("--max_cat_unique", type=int, default=30,
                        help="Maximale Anzahl eindeutiger Werte, damit String-Spalte als kategorisch gilt.")
    parser.add_argument("--date_columns", type=str, default="",
                        help="Kommagetrennte Liste von Spalten, die als Datum geparst werden sollen.")
    return parser.parse_args()


def load_data(input_csv: str) -> pd.DataFrame:
    """
    Lädt die CSV-Daten in einen DataFrame und erkennt automatisch, ob eine Kopfzeile ignoriert werden muss.
    """
    try:
        # 1️⃣ CSV normal einlesen
        df = pd.read_csv(input_csv)

        # 2️⃣ Falls nur eine einzige Spalte existiert, ist die erste Zeile keine echte Header-Zeile → Erste Zeile ignorieren
        if len(df.columns) == 1:
            logging.warning("Die erste Zeile scheint Metadaten zu sein. Ignoriere sie...")
            df = pd.read_csv(input_csv, skiprows=1)  # Erste Zeile überspringen

        # 3️⃣ Sicherstellen, dass 'result_bin' existiert
        if "result_bin" not in df.columns:
            logging.error("Fehler: 'result_bin' Spalte nicht gefunden! Verfügbar: %s", list(df.columns))
            sys.exit(1)

        logging.info("Daten aus '%s' erfolgreich geladen.", input_csv)
        logging.info("Erkannte Spalten: %s", list(df.columns))  # Debugging
        return df
    except Exception as e:
        logging.error("Fehler beim Laden der Datei '%s': %s", input_csv, e)
        sys.exit(1)



def build_preprocessor(X: pd.DataFrame, args: argparse.Namespace) -> ColumnTransformer:
    """
    Baut den Preprocessor, der numerische, kategoriale und Datums-Spalten verarbeitet.
    """
    # Vorab gegebene Datums-Spalten aus Argumenten verarbeiten
    date_cols_arg = [col.strip() for col in args.date_columns.split(",") if col.strip()]
    if date_cols_arg:
        for col in date_cols_arg:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors='ignore')

    numeric_cols, categorical_cols, detected_date_cols = detect_column_types(X, max_cat_unique=args.max_cat_unique)
    # Zusammenführen der explizit angegebenen und automatisch detektierten Datums-Spalten
    date_cols = list(set(date_cols_arg + detected_date_cols))

    date_pipeline = Pipeline([
        ("date_extractor", DateFeatureExtractor(date_cols=date_cols))
    ])

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
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
    Erstellt die vollständige Trainings-Pipeline inklusive Preprocessing und Modell.
    """
    preprocessor = build_preprocessor(X, args)
    model = get_model_by_name(args.model_name)

    training_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", model)
    ])
    return training_pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_arguments()

    df = load_data(args.input_csv)

    if LABEL_COLUMN not in df.columns:
        logging.error("Fehler: Spalte '%s' nicht vorhanden.", LABEL_COLUMN)
        sys.exit(1)

    y = df[LABEL_COLUMN].copy()
    X = df.drop(columns=[LABEL_COLUMN])

    training_pipeline = build_training_pipeline(X, args)

    try:
        training_pipeline.fit(X, y)
        joblib.dump(training_pipeline, args.output_model)
        logging.info("Modell '%s' wurde erfolgreich trainiert und in '%s' gespeichert.", args.model_name, args.output_model)
    except Exception as e:
        logging.error("Fehler beim Training oder Speichern des Modells: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
