import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE_URL = os.environ.get("APP_URL", "https://sentiment-analysis-icmn.onrender.com")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "monitoring/history.jsonl")

PROBES = [
    {"id": "clearly_positive", "text": "This movie was an absolute masterpiece. The acting was superb and the story stayed with me for days."},
    {"id": "clearly_negative", "text": "Terrible film. Total waste of time. Boring, predictable, and poorly acted from start to finish."},
    {"id": "mixed", "text": "Not bad, had some good moments but also dragged in parts."},
]


def _request(method, path, payload=None, timeout=90):
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def _load_last_record():
    if not os.path.exists(HISTORY_PATH):
        return None
    with open(HISTORY_PATH, "r") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def run_check():
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    try:
        status, health_body = _request("GET", "/health")
        record["health_status"] = status
        record["health_ok"] = status == 200 and health_body.get("status") == "ok"
    except (urllib.error.URLError, TimeoutError) as e:
        record["health_status"] = None
        record["health_ok"] = False
        record["health_error"] = str(e)

    previous = _load_last_record()
    previous_probes = {p["id"]: p for p in (previous or {}).get("probes", [])}

    probe_results = []
    flips = []

    if record["health_ok"]:
        for probe in PROBES:
            try:
                _, body = _request("POST", "/predict", {"text": probe["text"]})
                result = {"id": probe["id"], "label": body.get("label"), "confidence": body.get("confidence")}
            except (urllib.error.URLError, TimeoutError) as e:
                result = {"id": probe["id"], "label": None, "error": str(e)}

            prev = previous_probes.get(probe["id"])
            if prev and prev.get("label") and result.get("label") and prev["label"] != result["label"]:
                flips.append({"id": probe["id"], "was": prev["label"], "now": result["label"]})

            probe_results.append(result)

    record["probes"] = probe_results
    record["flips"] = flips
    return record


def main():
    record = run_check()

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(json.dumps(record, indent=2))

    if record["flips"]:
        print(f"::warning::{len(record['flips'])} probe(s) changed label since the last check: {record['flips']}")

    if not record["health_ok"]:
        print("::error::Health check failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
