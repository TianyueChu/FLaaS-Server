import requests
import docker
import time
import json
import numpy as np

HELPER_IMAGE = "fl-helper"           # Docker image name
HELPER_PORT = 8500                   # Port exposed by the helper
HELPER_HOST = "localhost"            # Assumes helper runs locally

def run_helper_container(port: int):
    client = docker.from_env()
    container = client.containers.run(
        HELPER_IMAGE,
        ports={f"{HELPER_PORT}/tcp": port},
        detach=True
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

def send_updates_to_helper(updates, use_split_learning, port: int, round_id=None, helper_id=None, timeout=180):
    url = f"http://{HELPER_HOST}:{port}/aggregate"

    # Build payload and serialize ONCE to count exact bytes sent
    payload = {
        "updates": updates,
        "use_split_learning": bool(use_split_learning),
        # optional identifiers so the helper can log them too
        "round": round_id,
        "helper_id": helper_id,
    }

    body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    uplink_bytes = len(body_bytes)

    headers = {
        "Content-Type": "application/json",
        # Disable compression so downlink counts are exact on the wire
        "Accept-Encoding": "identity",
        # Pass IDs in headers as well for helper-side logs
        "X-Round-Id": "" if round_id is None else str(round_id),
        "X-Helper-Id": "" if helper_id is None else str(helper_id),
    }

    t0 = time.perf_counter()
    with requests.post(url, data=body_bytes, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        # Read raw bytes (no decompression) to get true downlink size
        raw = r.raw.read(decode_content=False)
    latency = time.perf_counter() - t0

    downlink_bytes = len(raw)

    # Parse JSON from raw bytes
    data = json.loads(raw.decode("utf-8"))
    if "aggregated" not in data:
        raise RuntimeError("Invalid helper response: missing 'aggregated' key")
    aggregated = data["aggregated"]

    print(
        f"[NET] round={round_id} helper={helper_id} port={port} "
        f"uplink_bytes={uplink_bytes} downlink_bytes={downlink_bytes} "
        f"status=200 latency_s={latency:.3f}"
    )
    return aggregated


def aggregate_via_helper(group_updates, use_split_learning=False, port=8500, round_id=None, helper_id=None):
    container = run_helper_container(port)
    try:
        result = send_updates_to_helper(group_updates, use_split_learning, port,
                                        round_id=round_id, helper_id=helper_id)

        usage, limit = get_container_mem_usage_limit(container)
        print(f"[MEM] round={round_id} helper={helper_id} port={port} "
              f"usage={usage / 1024 / 1024:.1f} MiB limit={limit / 1024 / 1024:.1f} MiB "
              f"headroom={(limit - usage) / 1024 / 1024:.1f} MiB")

        return result
    except Exception as e:
        print("[ERROR] Helper crashed or failed to aggregate:")
        try:
            print(container.logs().decode(errors="replace"))
        except Exception as log_err:
            print(f"[WARNING] Could not read container logs: {log_err}")
        raise e
    finally:
        try:
            print("[DEBUG] Helper container logs:\n")
            print(container.logs().decode())
            # for showing the fastapi
            # time.sleep(1)
            container.stop()
        except Exception as stop_err:
            print(f"[WARNING] Failed to stop container: {stop_err}")

def get_container_mem_usage_limit(container):
    s = container.stats(stream=False)
    usage = int(s["memory_stats"].get("usage", 0))
    limit = int(s["memory_stats"].get("limit", 0))
    return usage, limit
