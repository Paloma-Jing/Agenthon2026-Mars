"""Verify the local environment on daily, monthly, and return forecasts.

Print per-stage progress and elapsed time; save command output for inspection.
"""

import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "track2-forecasting-public"
OUT = ROOT / "runs" / "environment_check"
UNITS = ["t2-F3-conundrum-joint-2005", "t2-F1-cpi-glidepath-2023", "t2-F1-ai-mom-2024"]


def run_stage(label, args, log_path):
    start = time.monotonic()
    print(f"START {label}", flush=True)
    error_path = log_path.with_suffix(log_path.suffix + ".stderr.log")
    with log_path.open("w") as log, error_path.open("w") as errors:
        process = subprocess.Popen(args, cwd=REPO, stdout=log, stderr=errors)
        while True:
            try:
                code = process.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                print(f"RUNNING {label}: {time.monotonic() - start:.0f}s; log={log_path}", flush=True)
                if time.monotonic() - start > 180:
                    process.kill()
                    process.wait()
                    raise SystemExit(f"TIMEOUT {label}; see {log_path} and {error_path}")
    if code:
        print(log_path.read_text(), flush=True)
        print(error_path.read_text(), flush=True)
        raise SystemExit(f"FAILED {label}: exit {code}")
    print(f"DONE {label}: {time.monotonic() - start:.1f}s", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for index, name in enumerate(UNITS, 1):
        unit = REPO / "units" / name
        card = tomllib.loads((unit / "card.toml").read_text())
        output = OUT / name
        output.mkdir(exist_ok=True)
        run_stage(f"{index}/{len(UNITS)} forecast {name}", [
            sys.executable, "-m", "qfbench2_track_forecasting.cli",
            "--panels", str(unit), "--text", str(unit / "text"),
            "--asof", card["provenance"]["data_cutoff"],
            "--out", str(output / "forecast.parquet"),
        ], OUT / f"{name}.forecast.log")
        score_log = OUT / f"{name}.check.json"
        run_stage(f"{index}/{len(UNITS)} gates {name}", [
            sys.executable, "scoring/scoring.py", "score",
            "--card", str(unit / "card.toml"),
            "--forecast", str(output / "forecast.parquet"),
        ], score_log)
        result = json.loads(score_log.read_text())
        if not result.get("admissible"):
            raise SystemExit(f"Gate check failed: {result}")
        results.append(result)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS: {len(results)} forecasts passed local gates. No accuracy scores are available.", flush=True)


if __name__ == "__main__":
    main()
