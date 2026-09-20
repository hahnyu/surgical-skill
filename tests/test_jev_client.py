"""Tests for src/jev_client.py. No test here calls the real API: a fake client stands in.

The important ones check what Jev is allowed to see (words only, no identifiers, no
labels, no numbers) and that the cache is written before parsing and honoured on re-run.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

import jev_client
from jev_client import (
    band_phrase,
    build_questions,
    build_scores_table,
    cache_path,
    load_jev_config,
    parse_record,
    percentile_within_task,
    prompt_hash,
    score_trial,
    state_for_trial,
)
from loaders import load_all_meta


@pytest.fixture(scope="module")
def cfg():
    return load_jev_config()


@pytest.fixture(scope="module")
def features():
    return pd.read_csv(jev_client.FEATURES_CSV, index_col="trial")


@pytest.fixture(scope="module")
def tasks():
    return load_all_meta().set_index("trial")["task"]


class FakeResult:
    def __init__(self, raw):
        self._raw = raw
        self.raw_http_response = self

    def json(self):
        return self._raw


class FakeClient:
    """Stands in for TypeSafeClient: records what it was sent and returns a canned answer."""

    def __init__(self, model="jev-1.13.0"):
        self.model, self.calls, self.sent = model, 0, None

    def system_one(self, state, questions):
        self.calls += 1
        self.sent = (state, questions)
        probs = {"0": 0.0, "1": 0.25, "2": 0.5, "3": 0.25, "4": 0.0}
        answers = {name: {"type": "score", "score": 2.0, "confidence": 0.5, "legend": {}, "probabilities": probs}
                   for name in questions}
        return FakeResult({"model": self.model, "answers": answers, "usage": {"input_tokens": 500, "output_tokens": 20}})


# --- percentiles and phrases -----------------------------------------------------------


def test_percentile_on_hand_worked_cases():
    assert percentile_within_task(3, [1, 2, 3, 4, 5]) == pytest.approx(0.5)      # 2 below + half of 1 tie
    assert percentile_within_task(1, [1, 2, 3, 4, 5]) == pytest.approx(0.1)      # 0 below + half of 1 tie
    assert percentile_within_task(5, [1, 2, 3, 4, 5]) == pytest.approx(0.9)
    assert percentile_within_task(2, [1, 2, 2, 2, 5]) == pytest.approx(0.5)      # 1 below + 1.5 for 3 ties


def test_band_phrases_at_the_boundaries(cfg):
    b = cfg["percentile_bands"]
    assert band_phrase(0.0, b) == "among the lowest of comparable attempts"
    assert band_phrase(0.0999, b) == "among the lowest of comparable attempts"
    assert band_phrase(0.10, b) == "lower than most comparable attempts"
    assert band_phrase(0.50, b) == "typical of comparable attempts"
    assert band_phrase(0.70, b) == "higher than most comparable attempts"
    assert band_phrase(0.90, b) == "among the highest of comparable attempts"
    assert band_phrase(1.0, b) == "among the highest of comparable attempts"


# --- what Jev is allowed to see ------------------------------------------------------------


def all_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_strings(v)
    else:
        yield str(obj)


@pytest.mark.parametrize("trial", ["Suturing_B001", "Knot_Tying_H005", "Needle_Passing_I003", "Suturing_E002"])
def test_the_state_is_words_only_with_no_identifier_label_or_number(cfg, features, tasks, trial):
    state = state_for_trial(trial, tasks[trial], features, cfg)
    text = " ".join(all_strings(state)).lower()
    assert not re.search(r"\d", text), "the state must contain no digits at all"
    for forbidden in ("novice", "expert", "intermediate", "skill", "grs", "osats", "score", "trial",
                      "subject", "surgeon b", "repetition"):
        assert forbidden not in text, f"'{forbidden}' must not appear in the state"
    # No trial name, subject letter or repetition in any form.
    assert trial.lower() not in text
    assert not re.search(r"\b[a-z]\d{3}\b", text)


def test_the_state_carries_the_task_and_one_phrase_per_measure(cfg, features, tasks):
    trial = "Suturing_B001"
    state = state_for_trial(trial, tasks[trial], features, cfg)
    assert state["task"] == "suturing"
    phrases = {p for _, p in cfg["percentile_bands"]}
    measures = [v for k, v in state.items() if k not in ("situation", "task")]
    flat = []
    for m in measures:
        flat += list(m.values()) if isinstance(m, dict) else [m]
    assert len(flat) == 16, "all 16 features must appear"
    assert set(flat) <= phrases


def test_the_state_is_short(cfg, features, tasks):
    """Jev's accuracy is documented to fall as the state fills with irrelevant detail."""
    trial = "Suturing_B001"
    text = json.dumps(state_for_trial(trial, tasks[trial], features, cfg))
    assert len(text) < 2500


def test_the_state_depends_on_the_features_only_not_on_any_label(cfg, features, tasks):
    """state_for_trial receives only the feature matrix, so a label cannot reach it. Confirm
    by passing a matrix with an extra label column: the state must not change."""
    trial = "Knot_Tying_D003"
    plain = state_for_trial(trial, tasks[trial], features, cfg)
    labelled = features.copy()
    labelled["grs"] = 99
    assert state_for_trial(trial, tasks[trial], labelled, cfg) == plain


def test_an_extreme_trial_is_described_as_extreme(cfg, features, tasks):
    """The trial with the longest Suturing time must be in the top band for total time."""
    suturing = features[[t.startswith("Suturing_") for t in features.index]]
    longest = suturing["task_time_s"].idxmax()
    state = state_for_trial(longest, "Suturing", features, cfg)
    assert state["total_time_to_complete"] == "among the highest of comparable attempts"
    shortest = suturing["task_time_s"].idxmin()
    assert state_for_trial(shortest, "Suturing", features, cfg)["total_time_to_complete"] == \
        "among the lowest of comparable attempts"


# --- the questions and the prompt fingerprint -----------------------------------------------


def test_there_are_seven_score_questions_with_five_levels_each(cfg):
    questions = build_questions(cfg)
    assert list(questions) == ["overall_skill", "respect_for_tissue", "suture_needle_handling",
                               "time_and_motion", "flow_of_operation", "overall_performance",
                               "quality_of_final_product"]
    for name, q in questions.items():
        assert q.type == "score"
        assert len(q.criteria) == 5, name
        assert not any(re.search(r"\d", c) for c in q.criteria), f"{name}: no numbers in level descriptions"


def test_prompt_hash_is_stable_and_changes_when_any_wording_changes(cfg):
    assert prompt_hash(cfg) == prompt_hash(cfg)
    edited = json.loads(json.dumps(cfg))
    edited["questions"]["overall_skill"]["criteria"][2] += " "
    assert prompt_hash(edited) != prompt_hash(cfg)
    edited = json.loads(json.dumps(cfg))
    edited["percentile_bands"][2][0] = 0.71
    assert prompt_hash(edited) != prompt_hash(cfg)


def test_the_model_is_pinned_to_a_version_not_an_alias(cfg):
    assert cfg["model"] == "jev-1.13.0"
    assert "latest" not in cfg["model"]


# --- the cache -----------------------------------------------------------------------------


def test_a_response_is_cached_before_parsing_and_never_requested_twice(cfg, features, tasks, tmp_path):
    client = FakeClient()
    questions = build_questions(cfg)
    state = state_for_trial("Suturing_B001", "Suturing", features, cfg)

    first = score_trial("Suturing_B001", 0, state, questions, client, cfg, raw_dir=tmp_path)
    assert client.calls == 1
    assert cache_path("Suturing_B001", 0, tmp_path).exists()
    stored = json.loads(cache_path("Suturing_B001", 0, tmp_path).read_text())
    assert stored["prompt_hash"] == prompt_hash(cfg) and stored["requested_model"] == "jev-1.13.0"
    assert stored["state_sent"] == state, "the exact state sent is kept, for the ethics statement"

    again = score_trial("Suturing_B001", 0, state, questions, client, cfg, raw_dir=tmp_path)
    assert client.calls == 1, "a cached trial must not be sent again"
    assert again == first
    score_trial("Suturing_B001", 1, state, questions, client, cfg, raw_dir=tmp_path)
    assert client.calls == 2, "a different run is a different request"


def test_the_raw_response_is_on_disk_even_if_the_model_check_then_fails(cfg, features, tmp_path):
    client = FakeClient(model="jev-9.9.9")
    state = state_for_trial("Suturing_B001", "Suturing", features, cfg)
    with pytest.raises(RuntimeError, match="not the pinned"):
        score_trial("Suturing_B001", 0, state, build_questions(cfg), client, cfg, raw_dir=tmp_path)
    assert cache_path("Suturing_B001", 0, tmp_path).exists(), "evidence is kept"
    # And a later re-run still refuses to use it silently.
    with pytest.raises(RuntimeError, match="not the pinned"):
        score_trial("Suturing_B001", 0, state, build_questions(cfg), FakeClient(), cfg, raw_dir=tmp_path)


def test_parsing_a_cached_response(cfg, features, tmp_path):
    state = state_for_trial("Suturing_B001", "Suturing", features, cfg)
    record = score_trial("Suturing_B001", 0, state, build_questions(cfg), FakeClient(), cfg, raw_dir=tmp_path)
    row = parse_record(record)
    assert row["trial"] == "Suturing_B001" and row["run"] == 0 and row["model"] == "jev-1.13.0"
    assert row["overall_skill__score"] == 2.0 and row["overall_skill__confidence"] == 0.5
    assert row["overall_skill__p2"] == 0.5
    assert sum(row[f"overall_skill__p{i}"] for i in range(5)) == pytest.approx(1.0)


def test_the_scores_table_is_built_from_disk_without_the_api(cfg, features, tmp_path):
    client = FakeClient()
    questions = build_questions(cfg)
    for trial in ("Suturing_B001", "Suturing_B002"):
        for run in (0, 1):
            score_trial(trial, run, state_for_trial(trial, "Suturing", features, cfg), questions, client, cfg, tmp_path)
    table = build_scores_table(tmp_path)
    assert len(table) == 4 and set(table["run"]) == {0, 1}
    assert client.calls == 4


def test_the_api_key_is_never_printed(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("TYPESAFE_API_KEY=secret-value-123\n")
    jev_client.load_api_key(env)
    out = capsys.readouterr()
    assert "secret-value-123" not in out.out + out.err
    import os
    assert os.environ["TYPESAFE_API_KEY"] == "secret-value-123"


def test_a_missing_key_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not found"):
        jev_client.load_api_key(tmp_path / "missing.env")
