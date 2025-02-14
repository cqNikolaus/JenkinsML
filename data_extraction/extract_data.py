import os
import sys
import requests
import pandas as pd
import json
import re

class JenkinsClient:
    """
    Ein Client zum Abrufen und Verarbeiten von Jenkins-Build-Daten.
    """
    def __init__(self, base_url, username, api_token, job_name, max_builds=50):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.api_token = api_token
        self.job_name = job_name
        self.max_builds = max_builds

    def get_build_api_url(self, build_number):
        """Erzeugt die URL zum Abrufen der Build-JSON-Daten."""
        return f"{self.base_url}/job/{self.job_name}/{build_number}/api/json"

    def get_console_log_url(self, build_number):
        """Erzeugt die URL zum Abrufen des Konsolen-Logs."""
        return f"{self.base_url}/job/{self.job_name}/{build_number}/consoleText"

    def fetch_json(self, url):
        """Holt JSON-Daten von einer URL."""
        try:
            response = requests.get(url, auth=(self.username, self.api_token), timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen von {url}: {e}", file=sys.stderr)
            return None

    def fetch_build_log(self, build_number):
        """Lädt den Konsolen-Log für den angegebenen Build."""
        url = self.get_console_log_url(build_number)
        try:
            response = requests.get(url, auth=(self.username, self.api_token))
            if response.status_code == 200:
                return response.text
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen des Konsolen-Logs für Build {build_number}: {e}", file=sys.stderr)
        return ""

    @staticmethod
    def count_error_keywords(log_text):
        """Zählt Schlüsselwörter (ERROR, Exception) im Log-Text."""
        return len(re.findall(r'(?i)(error|exception)', log_text))

    def parse_build_data(self, build_number, data):
        """
        Parst die JSON-Daten eines Builds und extrahiert die für ML relevanten Felder.
        Die unerwünschten Felder (timestamp, display_name, full_display_name, building, queue_id)
        werden nicht übernommen.
        """
        build_result = data.get("result", "UNKNOWN")
        duration_ms = data.get("duration", 0)
        timestamp_ms = data.get("timestamp", 0)
        estimated_duration_ms = data.get("estimatedDuration", 0)

        # built_on: Falls nicht gesetzt, Standardwert "built_in"
        built_on = data.get("builtOn", "") or "built_in"

        # build_url wird manuell erzeugt
        build_url = f"{self.base_url}/job/{self.job_name}/{build_number}/"

        # Aufteilen des Timestamps in Datum und Uhrzeit
        build_date = ""
        build_time_str = ""
        if timestamp_ms:
            dt = pd.to_datetime(timestamp_ms, unit="ms")
            build_date = dt.strftime("%Y-%m-%d")
            build_time_str = dt.strftime("%H:%M:%S")

        # Ermittlung der Commits aus "changeSet" oder "changeSets"
        commits_count = 0
        all_commits = []
        if "changeSet" in data and data["changeSet"]:
            all_commits = data["changeSet"].get("items", [])
            commits_count = len(all_commits)
        elif "changeSets" in data:
            for cs in data.get("changeSets", []):
                items = cs.get("items", [])
                all_commits.extend(items)
            commits_count = len(all_commits)

        # Berechnung der Anzahl der Commit-Autoren und Gesamtlänge der Commit-Messages
        commit_authors = set()
        total_commit_msg_length = 0
        for item in all_commits:
            author = item.get("author", {}).get("fullName", "")
            if author:
                commit_authors.add(author)
            total_commit_msg_length += len(item.get("msg", ""))
        commit_authors_count = len(commit_authors)

        # Ermittlung des change_set_kind
        if "changeSet" in data and data["changeSet"]:
            change_set_kind = data["changeSet"].get("kind", "")
        elif "changeSets" in data:
            kinds = [cs.get("kind", "") for cs in data.get("changeSets", []) if cs.get("kind", "")]
            change_set_kind = ", ".join(kinds)
        else:
            change_set_kind = ""

        culprits_count = len(data.get("culprits", []))
        executor_info = data.get("executor", {})
        executor_name = executor_info.get("name", "") if isinstance(executor_info, dict) else ""

        # Trigger-Typen aus den Aktionen extrahieren
        trigger_types = []
        for action in data.get("actions", []):
            if isinstance(action, dict) and "causes" in action:
                for cause in action.get("causes", []):
                    if "shortDescription" in cause:
                        trigger_types.append(cause["shortDescription"])
        trigger_types_str = ", ".join(trigger_types)

        # Parameter aus den Aktionen extrahieren
        parameters = {}
        for action in data.get("actions", []):
            if isinstance(action, dict) and "parameters" in action:
                for param in action.get("parameters", []):
                    parameters[param.get("name", "")] = param.get("value", "")
        parameters_str = json.dumps(parameters)

        result_bin = 1 if build_result == "FAILURE" else 0

        # Log-Text abrufen und Fehler zählen
        log_text = self.fetch_build_log(build_number)
        error_count = self.count_error_keywords(log_text)

        # Zusammenstellen der finalen Daten
        return {
            "build_number": build_number,
            "result": build_result,
            "result_bin": result_bin,
            "duration_sec": duration_ms / 1000.0,
            "commits_count": commits_count,
            "estimated_duration_sec": estimated_duration_ms / 1000.0,
            "built_on": built_on,
            "build_url": build_url,
            "parameters": parameters_str,
            "commit_authors_count": commit_authors_count,
            "total_commit_msg_length": total_commit_msg_length,
            "change_set_kind": change_set_kind,
            "culprits_count": culprits_count,
            "executor_name": executor_name,
            "trigger_types": trigger_types_str,
            "error_count": error_count,
            "build_date": build_date,   # Neues Datums-Feld
            "build_time": build_time_str  # Neues Uhrzeit-Feld
        }

    def fetch_all_builds(self):
        """Holt alle Builds bis zu max_builds und gibt eine Liste von Daten-Dictionaries zurück."""
        builds = []
        for build_number in range(1, self.max_builds + 1):
            url = self.get_build_api_url(build_number)
            try:
                response = requests.get(url, auth=(self.username, self.api_token), timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Fehler beim Abrufen von Build {build_number}: {e}", file=sys.stderr)
                break

            data = response.json()
            build_data = self.parse_build_data(build_number, data)
            builds.append(build_data)
        return builds

def main():
    base_url = "https://jenkins-clemens01-0.comquent.academy/"
    username = "admin"
    api_token = os.getenv("JENKINS_TOKEN")
    job_name = os.getenv("JOB_NAME")
    max_builds = 5000

    client = JenkinsClient(base_url, username, api_token, job_name, max_builds)
    builds_data = client.fetch_all_builds()

    if not builds_data:
        sys.exit("Keine Build-Daten verfügbar oder API-Aufruf fehlgeschlagen.")

    df = pd.DataFrame(builds_data)
    # Filtert Builds mit unbekanntem Ergebnis heraus
    df = df[df["result"] != "UNKNOWN"]
    df.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    main()
