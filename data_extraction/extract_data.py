import os
import sys
import requests
import pandas as pd
import json
import re

# Jenkins-Server Konfiguration
JENKINS_URL = "https://jenkins-clemens01-0.comquent.academy/"
USERNAME = "admin"
API_TOKEN = os.getenv("JENKINS_TOKEN")
JOB_NAME = os.getenv("JOB_NAME")
MAX_BUILDS = 5000

def fetch_build_log(job_name, build_number):
    """
    Ruft den Konsolen-Log eines Builds ab.
    """
    log_url = f"{JENKINS_URL}/job/{job_name}/{build_number}/consoleText"
    response = requests.get(log_url, auth=(USERNAME, API_TOKEN))
    if response.status_code == 200:
        return response.text
    return ""

def count_error_keywords(log_text):
    """
    Zählt, wie oft Schlüsselwörter (ERROR, Exception) im Log vorkommen.
    """
    return len(re.findall(r'(?i)(error|exception)', log_text))

def fetch_jenkins_data(job_name, max_builds=50):
    """
    Ruft Daten zu den letzten Builds aus Jenkins ab und extrahiert relevante Informationen.
    """
    build_data = []
    for build_number in range(1, max_builds + 1):
        url = f"{JENKINS_URL}/job/{job_name}/{build_number}/api/json"
        try:
            response = requests.get(url, auth=(USERNAME, API_TOKEN), timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen von Build {build_number}: {e}", file=sys.stderr)
            break

        data = response.json()
        build_result = data.get("result", "UNKNOWN")
        duration_ms = data.get("duration", 0)
        timestamp_ms = data.get("timestamp", 0)
        estimated_duration_ms = data.get("estimatedDuration", 0)
        built_on = data.get("builtOn", "")
        if not built_on:
            built_on = "built_in"
        display_name = data.get("displayName", "")
        full_display_name = data.get("fullDisplayName", "")
        build_url = f"{JENKINS_URL}/job/{job_name}/{build_number}/"
        queue_id = data.get("queueId", 0)
        building = int(data.get("building", False))

        build_time = ""
        if timestamp_ms:
            build_time = pd.to_datetime(timestamp_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S")

        commits_count = 0
        if "changeSet" in data and data["changeSet"]:
            commits_count = len(data["changeSet"].get("items", []))
        elif "changeSets" in data:
            commits_count = sum(len(cs.get("items", [])) for cs in data.get("changeSets", []))

        commit_authors = set()
        total_commit_msg_length = 0
        for item in change_set.get("items", []):
            author = item.get("author", {}).get("fullName", "")
            if author:
                commit_authors.add(author)
            total_commit_msg_length += len(item.get("msg", ""))
        commit_authors_count = len(commit_authors)
        change_set_kind = change_set.get("kind", "")

        culprits_count = len(data.get("culprits", []))
        executor_info = data.get("executor", {})
        executor_name = executor_info.get("name", "") if isinstance(executor_info, dict) else ""

        trigger_types = []
        for action in data.get("actions", []):
            if isinstance(action, dict) and "causes" in action:
                for cause in action.get("causes", []):
                    if "shortDescription" in cause:
                        trigger_types.append(cause["shortDescription"])
        trigger_types_str = ", ".join(trigger_types)

        parameters = {}
        for action in data.get("actions", []):
            if isinstance(action, dict) and "parameters" in action:
                for param in action.get("parameters", []):
                    parameters[param.get("name", "")] = param.get("value", "")
        parameters_str = json.dumps(parameters)

        result_bin = 1 if build_result == "FAILURE" else 0

        log_text = fetch_build_log(job_name, build_number)
        error_count = count_error_keywords(log_text)

        build_data.append({
            "build_number": build_number,
            "result": build_result,
            "result_bin": result_bin,
            "duration_sec": duration_ms / 1000.0,
            "commits_count": commits_count,
            "timestamp_ms": timestamp_ms,
            "build_time": build_time,
            "estimated_duration_sec": estimated_duration_ms / 1000.0,
            "built_on": built_on,
            "display_name": display_name,
            "full_display_name": full_display_name,
            "build_url": build_url,
            "building": building,
            "queue_id": queue_id,
            "parameters": parameters_str,
            "commit_authors_count": commit_authors_count,
            "total_commit_msg_length": total_commit_msg_length,
            "change_set_kind": change_set_kind,
            "culprits_count": culprits_count,
            "executor_name": executor_name,
            "trigger_types": trigger_types_str,
            "error_count": error_count
        })

    return build_data

def main():
    """
    Abruf der Jenkins-Build-Daten und Ausgabe als CSV über stdout.
    """
    data_list = fetch_jenkins_data(JOB_NAME, MAX_BUILDS)
    if not data_list:
        sys.exit("Keine Build-Daten verfügbar oder API-Aufruf fehlgeschlagen.")
    df = pd.DataFrame(data_list)
    df = df[df["result"] != "UNKNOWN"]
    df.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    main()
