"""Phase 4: score each trial with Jev (TypeSafe's System One model).

Rules from CLAUDE.md that this module enforces:
  * All arithmetic happens in Python. Jev is documented as unable to count or judge
    whether numbers are close, so it is never sent a number: every measure is turned
    into a short phrase ("higher than most comparable attempts") in code.
  * Every raw response is written to results/jev_raw/ BEFORE anything is parsed, and a
    trial that is already cached is never sent again, so re-running the analysis never
    needs the API.
  * The model version is pinned (jev-1.13.0) and every response's `model` field is
    checked against it.
  * The API key comes from the environment or from .env, and is never printed.

What is sent, and only this: a wording-only state (no trial name, surgeon, repetition,
skill label, score or digit) and the seven questions in configs/jev.yaml.

Run:  uv run python src/jev_client.py
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from loaders import load_all_meta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "jev.yaml"
FEATURES_CSV = PROJECT_ROOT / "results" / "features.csv"
RAW_DIR = PROJECT_ROOT / "results" / "jev_raw"
SCORES_CSV = PROJECT_ROOT / "results" / "jev_scores.csv"
ENV_PATH = PROJECT_ROOT / ".env"


def load_jev_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_api_key(env_path: Path = ENV_PATH) -> None:
    """Make TYPESAFE_API_KEY available to the SDK without ever printing it.

    Uses the environment if it is already set; otherwise reads the .env file, which is
    gitignored. Returns nothing on purpose, so the value cannot leak into a log.
    """
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "TYPESAFE_API_KEY" and value.strip():
                os.environ["TYPESAFE_API_KEY"] = value.strip().strip("'\"")
                return
    raise RuntimeError("TYPESAFE_API_KEY not found in the environment or in .env")


# --- the state: words only, computed in code ------------------------------------


def percentile_within_task(value: float, task_values) -> float:
    """Share of the task's trials below `value`, counting ties as half.

    For [1, 2, 3, 4, 5] and value 3: two are below and one ties, so (2 + 0.5) / 5 = 0.5.
    """
    task_values = np.asarray(task_values, dtype=float)
    below = np.sum(task_values < value)
    tied = np.sum(task_values == value)
    return float((below + 0.5 * tied) / len(task_values))


def band_phrase(percentile: float, bands: list) -> str:
    """The phrase for a percentile: the first band whose upper bound it falls under."""
    for upper, phrase in bands:
        if percentile < upper:
            return phrase
    return bands[-1][1]


def state_for_trial(trial: str, task: str, features: pd.DataFrame, cfg: dict) -> dict:
    """The state Jev sees for one trial: the task and a phrase per measure.

    `features` is the feature matrix only (no labels, no subjects); `trial` is used
    only to look up its own row and is NEVER written into the state. Each measure is
    described relative to the other trials of the same task, which are label-free, so
    no label can reach the state.
    """
    same_task = features[features.index.map(lambda t: t.startswith(task + "_"))]
    row = features.loc[trial]

    def describe(column: str) -> str:
        return band_phrase(percentile_within_task(row[column], same_task[column]), cfg["percentile_bands"])

    def build(spec):
        if isinstance(spec, dict):
            return {label: build(sub) for label, sub in spec.items()}
        return describe(spec)

    return {
        "situation": " ".join(cfg["situation"].split()),
        "task": cfg["task_words"][task],
        **build(cfg["state_fields"]),
    }


def build_questions(cfg: dict) -> dict:
    """The Score questions from the config, as SDK objects."""
    from typesafe_sdk import Score
    return {
        name: Score(instructions=" ".join(q["instructions"].split()), criteria=list(q["criteria"]))
        for name, q in cfg["questions"].items()
    }


def prompt_hash(cfg: dict) -> str:
    """A fingerprint of everything Jev is shown, so each cached response says which
    exact prompt produced it. It changes if any wording, band or field is edited."""
    material = {k: cfg[k] for k in ("prompt_version", "model", "situation", "task_words",
                                    "percentile_bands", "state_fields", "questions")}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


# --- the cache ---------------------------------------------------------------------


def cache_path(trial: str, run: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{trial}__run{run}.json"


def score_trial(trial, run, state, questions, client, cfg, raw_dir: Path = RAW_DIR) -> dict:
    """One request for one trial, cached to disk as raw JSON before parsing.

    If the trial and run are already cached, the API is NOT called: the stored file is
    returned. Otherwise the raw response is written first, then returned. The stored
    `model` is checked against the pinned version, so a silent alias change is caught.
    """
    path = cache_path(trial, run, raw_dir)
    if path.exists():
        record = json.loads(path.read_text())
    else:
        result = client.system_one(state, questions)
        raw = result.raw_http_response.json()
        record = {
            "trial": trial,
            "run": run,
            "requested_model": cfg["model"],
            "prompt_version": cfg["prompt_version"],
            "prompt_hash": prompt_hash(cfg),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state_sent": state,
            "response": raw,
        }
        raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))    # written BEFORE any parsing

    # Checked for fresh AND cached records, so a response from the wrong model version
    # can never be used silently, even on a later re-run.
    answered_by = record["response"].get("model")
    if answered_by != cfg["model"]:
        raise RuntimeError(f"{trial}: answered by model {answered_by!r}, not the pinned {cfg['model']!r}")
    return record


def parse_record(record: dict) -> dict:
    """Flatten one cached record into one row: score, confidence and the level
    probabilities for each question."""
    row = {"trial": record["trial"], "run": record["run"], "model": record["response"]["model"],
           "prompt_version": record["prompt_version"], "prompt_hash": record["prompt_hash"],
           "input_tokens": record["response"]["usage"]["input_tokens"]}
    for name, answer in record["response"]["answers"].items():
        row[f"{name}__score"] = answer["score"]
        row[f"{name}__confidence"] = answer["confidence"]
        for level, p in answer["probabilities"].items():
            row[f"{name}__p{level}"] = p
    return row


def build_scores_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Parse every cached response into one table (no API call)."""
    rows = [parse_record(json.loads(p.read_text())) for p in sorted(raw_dir.glob("*__run*.json"))]
    return pd.DataFrame(rows).sort_values(["trial", "run"]).reset_index(drop=True)


# --- running it ----------------------------------------------------------------------


def make_client(cfg: dict):
    from typesafe_sdk import RetryPolicy, TypeSafeClient
    load_api_key()
    r = cfg["retry"]
    return TypeSafeClient(model=cfg["model"], retry=RetryPolicy(
        max_retries=r["max_retries"], backoff_initial=r["backoff_initial"], backoff_max=r["backoff_max"]))


def main() -> None:
    cfg = load_jev_config()
    features = pd.read_csv(FEATURES_CSV, index_col="trial")
    tasks = load_all_meta().set_index("trial")["task"]            # task only; never sent as a label
    questions = build_questions(cfg)
    client = make_client(cfg)

    trials = list(features.index)
    failures, called = [], 0
    for run in range(cfg["repeats"]):
        for trial in trials:
            already = cache_path(trial, run).exists()
            try:
                score_trial(trial, run, state_for_trial(trial, tasks[trial], features, cfg),
                            questions, client, cfg)
            except Exception as error:                       # keep going, but never hide it
                failures.append((trial, run, f"{type(error).__name__}: {error}"))
                continue
            called += 0 if already else 1
        print(f"run {run}: done ({len(trials)} trials)")
    print(f"\nnew API calls this session: {called}; failures: {len(failures)}")
    for f in failures:
        print("  FAILED", f)

    table = build_scores_table()
    table.to_csv(SCORES_CSV, index=False)
    print(f"parsed {len(table)} cached responses into {SCORES_CSV}")


if __name__ == "__main__":
    main()
