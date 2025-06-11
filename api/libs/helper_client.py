import requests
import docker
import time
import numpy as np

HELPER_IMAGE = "fl-helper"           # Docker image name
HELPER_PORT = 8500                   # Port exposed by the helper
HELPER_HOST = "localhost"            # Assumes helper runs locally

def run_helper_container(port: int):
    client = docker.from_env()
    container = client.containers.run(
        HELPER_IMAGE,
        ports={f"{HELPER_PORT}/tcp": port},
        detach=True,
        remove=True
    )

    for _ in range(10):
        try:
            r = requests.get(f"http://localhost:{port}/docs")
            if r.status_code == 200:
                return container
        except Exception:
            time.sleep(1)
    container.stop()
    raise RuntimeError("Helper on port %d failed to start." % port)

def send_updates_to_helper(updates, use_split_learning, port: int):
    payload = {
        "updates": updates,
        "use_split_learning": use_split_learning
    }
    response = requests.post(f"http://localhost:{port}/aggregate", json=payload)
    response.raise_for_status()
    return response.json()["aggregated"]

def aggregate_via_helper(group_updates, use_split_learning=False, port=8500):
    container = run_helper_container(port)
    try:
        return send_updates_to_helper(group_updates, use_split_learning, port)
    finally:
        container.stop()