"""Campagne de charge reproductible : mêmes paliers pour chaque limite CPU de l'API.

Usage (depuis la racine du dépôt, pile Docker démarrée) :
    python load_tests/run_campaign.py --cpus 0.5 1 2 --users 10 30 60 --repeats 2
Résultats : reports/load/summary.csv (ou --summary) et exports Locust bruts dans reports/load/raw/.
"""

import argparse
import csv
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

REPORTS = Path("reports/load")
CONTAINERS = {"api": "bank-marketing-api-1", "postgres": "bank-marketing-postgres-1"}


def compose(*args: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", *args], env=env, check=check, capture_output=True, text=True)


def to_mib(value: str) -> float:
    number = float(value.rstrip("KMGiB"))
    unit = value[len(value.rstrip("KMGiB")):]
    return number * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "B": 1 / 1024 ** 2}[unit]


def sample_resources(samples: list, stop: threading.Event) -> None:
    """docker stats : CPU en % d'un cœur (100 % = 1 cœur), mémoire utilisée."""
    while not stop.is_set():
        output = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}};{{.CPUPerc}};{{.MemUsage}}", *CONTAINERS.values()],
            capture_output=True, text=True,
        ).stdout
        for line in output.splitlines():
            name, cpu, memory = line.split(";")
            samples.append({"time": time.monotonic(), "name": name,
                            "cpu": float(cpu.rstrip("%")), "mem": to_mib(memory.split("/")[0].strip())})


def run_locust(name: str, users: int, duration: int, ramp_seconds: int, env: dict) -> dict:
    # --no-deps : sinon Compose recrée l'API avec sa configuration par défaut.
    compose(
        "--profile", "load", "run", "--rm", "-T", "--no-deps", "locust", "-f", "locustfile.py", "--headless",
        "--host", "http://api:8000", "-u", str(users), "-r", str(max(1, users // ramp_seconds)),
        "-t", f"{duration + ramp_seconds}s", "--reset-stats", "--only-summary",
        "--csv", f"/mnt/reports/raw/{name}", env=env, check=False,
    )
    with (REPORTS / "raw" / f"{name}_stats.csv").open(encoding="utf-8") as stream:
        return {row["Name"]: row for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpus", nargs="+", default=["0.5", "1", "2"])
    parser.add_argument("--users", nargs="+", type=int, default=[10, 30, 60])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--duration", type=int, default=60, help="Durée stable mesurée, en secondes.")
    parser.add_argument("--ramp", type=int, default=10, help="Montée en charge, exclue des mesures.")
    parser.add_argument("--summary", default="summary.csv", help="Fichier de résultats dans reports/load/.")
    args = parser.parse_args()
    (REPORTS / "raw").mkdir(parents=True, exist_ok=True)
    rows = []
    for cpus in args.cpus:
        env = {**os.environ, "API_CPUS": cpus, "API_MEMORY": "1g", "API_PORT": os.environ.get("API_PORT", "8000")}
        compose("up", "-d", "--wait", "--force-recreate", "--no-deps", "api", env=env)
        # Vérifier la configuration réellement testée, pas seulement celle demandée.
        applied_cpus = int(subprocess.run(
            ["docker", "inspect", CONTAINERS["api"], "--format", "{{.HostConfig.NanoCpus}}"],
            capture_output=True, text=True, check=True,
        ).stdout) / 1e9
        if applied_cpus != float(cpus):
            raise SystemExit(f"Limite CPU appliquée {applied_cpus} au lieu de {cpus} : campagne arrêtée.")
        run_locust(f"warmup-cpu{cpus}", 10, 20, 5, env)  # échauffement non retenu
        for users in args.users:
            for repeat in range(1, args.repeats + 1):
                name = f"cpu{cpus}-u{users}-r{repeat}"
                samples, stop = [], threading.Event()
                sampler = threading.Thread(target=sample_resources, args=(samples, stop))
                started = time.monotonic()
                sampler.start()
                stats = run_locust(name, users, args.duration, args.ramp, env)
                stop.set()
                sampler.join()
                stable = [s for s in samples if s["time"] - started > args.ramp + 3]

                def resource(container, key, function):
                    values = [s[key] for s in stable if s["name"] == CONTAINERS[container]]
                    return round(function(values), 1) if values else ""

                predict, total = stats["/predict"], stats["Aggregated"]
                rows.append({
                    "api_cpus": cpus, "applied_cpus": applied_cpus, "users": users, "repeat": repeat,
                    "predict_rps": round(float(predict["Requests/s"]), 2), "total_rps": round(float(total["Requests/s"]), 2),
                    "predict_p50_ms": predict["50%"], "predict_p95_ms": predict["95%"], "predict_p99_ms": predict["99%"],
                    "requests": total["Request Count"], "failures": total["Failure Count"],
                    "failure_rate": round(int(total["Failure Count"]) / max(1, int(total["Request Count"])), 4),
                    "api_cpu_mean_pct": resource("api", "cpu", statistics.mean),
                    "api_cpu_max_pct": resource("api", "cpu", max),
                    "api_mem_max_mib": resource("api", "mem", max),
                    "postgres_cpu_mean_pct": resource("postgres", "cpu", statistics.mean),
                })
                print(rows[-1], flush=True)
    with (REPORTS / args.summary).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    compose("up", "-d", "--wait", "--force-recreate", "--no-deps", "api",
            env={**os.environ, "API_PORT": os.environ.get("API_PORT", "8000")})


if __name__ == "__main__":
    main()
