"""Tests for the GRS band rule in src/plot_features.py.

The bands decide which trials are called low, mid and high, so a silent mistake
here would change every figure. Work the expected answers out by hand.
"""

import pandas as pd
import pytest

from plot_features import grs_bands


def bands_of(scores):
    return list(grs_bands(pd.Series(scores)))


def test_nine_evenly_spaced_scores_split_three_three_three():
    # Scores 6..14. The 33.3rd percentile is 6 + 8/3 = 8.67 and the 66.7th is
    # 6 + 16/3 = 11.33, so: low = 6,7,8; mid = 9,10,11; high = 12,13,14.
    assert bands_of(range(6, 15)) == ["low"] * 3 + ["mid"] * 3 + ["high"] * 3


def test_bands_are_ordered_low_below_mid_below_high():
    result = grs_bands(pd.Series(range(6, 15)))
    assert result.ordered
    assert list(result.categories) == ["low", "mid", "high"]
    assert result.min() == "low" and result.max() == "high"


def test_the_same_score_always_lands_in_the_same_band():
    # Ties share a band; nobody is put in a different band by order or luck.
    scores = [10, 10, 10, 10, 20, 20, 30, 30, 30]
    result = dict(zip(scores, bands_of(scores)))
    for score, band in zip(scores, bands_of(scores)):
        assert result[score] == band


def test_band_follows_the_score_not_the_position():
    forward = bands_of([6, 9, 12, 15, 18, 21])
    backward = bands_of([21, 18, 15, 12, 9, 6])
    assert forward == backward[::-1]


def test_a_higher_score_is_never_in_a_lower_band():
    scores = [7, 13, 8, 24, 19, 11, 16, 30, 9, 21, 14, 12]
    order = {"low": 0, "mid": 1, "high": 2}
    labelled = sorted(zip(scores, bands_of(scores)))
    ranks = [order[b] for _, b in labelled]
    assert ranks == sorted(ranks)


def test_the_cuts_come_from_the_scores_passed_in_so_tasks_are_independent():
    # The same score 18 is "low" among high scores and "high" among low ones.
    among_high = bands_of([18, 22, 25, 28, 30, 27])
    among_low = bands_of([18, 8, 9, 7, 10, 6])
    assert among_high[0] == "low"
    assert among_low[0] == "high"


def test_the_index_is_kept():
    scores = pd.Series([12, 8, 20], index=["a", "b", "c"])
    result = grs_bands(scores)
    assert len(result) == 3
