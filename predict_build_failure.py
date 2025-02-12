import os
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Jenkins Zugangsdaten und Basis-URL
JENKINS_URL = "https://jenkins-clemens01-0.comquent.academy/"
JOB_NAME = "jenkins-setup"
USERNAME = "admin"
API_TOKEN = os.getenv('JENKINS_TOKEN')


def fetch_jenkins_data(job_name, max_builds=50):
    """
    Ruft für einen angegebenen Job die letzten N Builds ab und
    gibt eine Liste mit relevanten Informationen zurück.
    """
    build_data = []

    for build_number in range(1, max_builds + 1):
        url = f"{JENKINS_URL}/job/{job_name}/{build_number}/api/json"
        response = requests.get(url, auth=(USERNAME, API_TOKEN))

        if response.status_code != 200:
            # Build existiert nicht oder kann nicht abgerufen werden, überspringen
            continue

        data = response.json()

        # Extrahiere relevante Felder
        build_result = data.get("result", "UNKNOWN")
        duration = data.get("duration", 0)  # Dauer in Millisekunden

        # Anzahl Commits in diesem Build aus 'changeSet' holen
        change_set = data.get("changeSet", {})
        commits_count = len(change_set.get("items", []))

        # Binarer Build-Status: 1 = FAILURE, 0 = SUCCESS
        if build_result == "FAILURE":
            result_bin = 1
        else:
            result_bin = 0

        build_data.append({
            "build_number": build_number,
            "result": build_result,
            "result_bin": result_bin,
            "duration_sec": duration / 1000.0,
            "commits_count": commits_count
        })

    return build_data


# 1. Daten abrufen
raw_build_data = fetch_jenkins_data(JOB_NAME, max_builds=50)

# 2. In DataFrame umwandeln und ungeeignete Zeilen herausfiltern
df = pd.DataFrame(raw_build_data)
df = df[df["result"] != "UNKNOWN"]  # nur valide Resultate

# 3. Feature-Set & Label definieren
X = df[["duration_sec", "commits_count"]]
y = df["result_bin"]

# 4. Trainings-/Testsplit (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# 5. Logistische Regression trainieren
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Bewertung (Accuracy)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 7. Beispiel: Prognose für einen neuen Build
#    Angenommen der nächste Build läuft 300 Sekunden und hat 5 Commits
test_build_data = [[300.0, 5]]  # [duration_sec, commits_count]
prob_failure = model.predict_proba(test_build_data)[0][1]  # index 1 = Wahrscheinlichkeit für FAIL
print(f"Wahrscheinlichkeit für Fehlschlag: {prob_failure * 100:.2f}%")
