#!/usr/bin/env python3
"""
1. Einleitung:
   Dieses Skript prognostiziert die Erfolgswahrscheinlichkeit des nächsten Pipeline-Builds.
   Es lädt ein trainiertes Modell (model.pkl), welches mittels joblib gespeichert wurde,
   und berechnet basierend auf den übergebenen CSV-Eingabedaten die durchschnittliche Erfolgswahrscheinlichkeit.
   Das Ergebnis wird anschließend in der Konsole ausgegeben.

2. Benötigte Module importieren:
   - argparse: Für das Einlesen von Kommandozeilenargumenten.
   - joblib: Zum Laden des trainierten Modells.
   - pandas: Zum Einlesen und Verarbeiten der CSV-Daten.
   - sys: Für den kontrollierten Programmabbruch bei Fehlern.
   - train_model: Damit die im Modell referenzierten Klassen (z. B. TimeFeaturesExtractor) verfügbar sind.
"""

import argparse
import joblib
import pandas as pd
import sys
from train_model import TimeFeaturesExtractor  # Wichtig für das Unpickling des Modells

def main():
    # Argument-Parsing: Kommandozeilenargumente einlesen
    parser = argparse.ArgumentParser(
        description="Vorhersage der Erfolgswahrscheinlichkeit des nächsten Pipeline-Builds"
    )
    parser.add_argument("--model", required=True, help="Pfad zur Modell-Datei (model.pkl)")
    parser.add_argument("--data", required=True, help="Pfad zur CSV-Datei mit Eingabedaten")
    args = parser.parse_args()

    # Modell laden
    try:
        model = joblib.load(args.model)
    except FileNotFoundError:
        sys.exit(f"Fehler: Die Modell-Datei '{args.model}' wurde nicht gefunden.")
    except Exception as e:
        sys.exit(f"Fehler beim Laden des Modells: {e}")

    # CSV-Daten laden
    try:
        data = pd.read_csv(args.data)
    except FileNotFoundError:
        sys.exit(f"Fehler: Die CSV-Datei '{args.data}' wurde nicht gefunden.")
    except Exception as e:
        sys.exit(f"Fehler beim Laden der CSV-Daten: {e}")

    # --- Spalten-Ausrichtung ---
    # Extrahiere aus dem im Modell enthaltenen Preprocessor die erwarteten numerischen und kategorialen Spalten.
    try:
        preprocessor = model.named_steps["preprocessor"]
        # Annahme: Der erste Transformer ist für numerische und der zweite für kategoriale Features.
        expected_numeric = list(preprocessor.transformers_[0][2])
        expected_categorical = list(preprocessor.transformers_[1][2])
        expected_columns = expected_numeric + expected_categorical
    except Exception as e:
        sys.exit(f"Fehler beim Extrahieren der erwarteten Spalten: {e}")

    # Fehlende erwartete Spalten mit NaN ergänzen, damit die Pipeline sie verarbeiten kann.
    for col in expected_columns:
        if col not in data.columns:
            data[col] = pd.NA

    # --- Vorhersage ---
    try:
        # predict_proba liefert für jeden Eintrag Wahrscheinlichkeiten zurück.
        predictions = model.predict_proba(data)
        # Annahme: Spalte mit Index 1 entspricht der Erfolgswahrscheinlichkeit.
        success_probability = predictions[:, 1].mean()  # Durchschnitt über alle Einträge
    except Exception as e:
        sys.exit(f"Fehler während der Vorhersage: {e}")

    # Ergebnis in der Konsole ausgeben
    print(f"Fehlschlagswahrscheinlichkeit: {round(success_probability * 100)}%")

if __name__ == "__main__":
    main()
