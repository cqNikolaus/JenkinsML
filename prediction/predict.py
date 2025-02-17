#!/usr/bin/env python3
"""
1. Einleitung:
   Dieses Skript prognostiziert die Erfolgswahrscheinlichkeit des nächsten Pipeline-Builds.
   Es lädt ein trainiertes Modell (model.pkl) und berechnet basierend auf den übergebenen CSV-Eingabedaten
   die durchschnittliche Erfolgswahrscheinlichkeit. Das Ergebnis wird anschließend in der Konsole ausgegeben.

2. Benötigte Module importieren:
   - argparse: Für das Einlesen von Kommandozeilenargumenten.
   - pickle: Zum Laden des trainierten Modells.
   - pandas: Zum Einlesen und Verarbeiten der CSV-Daten.
   - sys: Für den kontrollierten Programmabbruch bei Fehlern.
"""

import argparse
import pickle
import pandas as pd
import sys

def main():
    # 3. Argumenten-Parsing: Kommandozeilenargumente einlesen
    parser = argparse.ArgumentParser(
        description="Vorhersage der Erfolgswahrscheinlichkeit des nächsten Pipeline-Builds"
    )
    parser.add_argument("--model", required=True, help="Pfad zur Modell-Datei (model.pkl)")
    parser.add_argument("--data", required=True, help="Pfad zur CSV-Datei mit Eingabedaten")
    args = parser.parse_args()

    # 4. Modell laden: Trainiertes Modell aus der angegebenen Datei laden
    try:
        with open(args.model, "rb") as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        sys.exit(f"Fehler: Die Modell-Datei '{args.model}' wurde nicht gefunden.")
    except Exception as e:
        sys.exit(f"Fehler beim Laden des Modells: {e}")

    # 5. CSV-Daten laden: Eingabedaten aus der CSV-Datei einlesen
    try:
        data = pd.read_csv(args.data)
    except FileNotFoundError:
        sys.exit(f"Fehler: Die CSV-Datei '{args.data}' wurde nicht gefunden.")
    except Exception as e:
        sys.exit(f"Fehler beim Laden der CSV-Daten: {e}")

    # 6. Vorhersage: Erfolgswahrscheinlichkeit anhand der Eingabedaten berechnen
    try:
        # Annahme: Das Modell besitzt die Methode predict_proba, die für jeden Eintrag Wahrscheinlichkeiten zurückgibt.
        predictions = model.predict_proba(data)
        # Annahme: Die Spalte mit Index 1 entspricht der Wahrscheinlichkeit eines erfolgreichen Builds.
        success_probability = predictions[:, 1].mean()  # Durchschnittswahrscheinlichkeit über alle Eingaben
    except Exception as e:
        sys.exit(f"Fehler während der Vorhersage: {e}")

    # 7. Ausgabe: Ergebnis klar in der Konsole ausgeben
    print(f"Erfolgswahrscheinlichkeit: {round(success_probability * 100)}%")

if __name__ == "__main__":
    main()
