import numpy as np
import pytest

from ramscore.e5 import (METRICS, check_reproduced, conformal_lambda, ece, expansion_mass, fit_temperature, level_probs,
                         tally)


def test_check_reproduced_allows_near_ties_only():
    logits = np.array([[5.0, 4.9999, 0, 0, 0], [0, 3.0, 0, 0, 0]])
    assert check_reproduced(logits, np.array([0, 1]), 'exact') == 0
    assert check_reproduced(logits, np.array([1, 1]), 'near tie') == 1
    with pytest.raises(SystemExit):
        check_reproduced(logits, np.array([0, 2]), 'real difference')


def test_level_probs_merge_grades_3_and_5_and_sum_to_one():
    logits = np.array([[2.0, 1.0, 0.0, -1.0, -1.0]])
    p = level_probs(logits)
    raw = np.exp(logits) / np.exp(logits).sum()
    assert p.shape == (1, 4)
    assert p.sum() == pytest.approx(1)
    assert p[0, 3] == pytest.approx(raw[0, 3] + raw[0, 4])


def test_level_probs_applies_offset_before_temperature():
    logits = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    offset = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    assert level_probs(logits, 2.0, offset) == pytest.approx(level_probs((logits + offset) / 2))


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(20000, 5)) * 1.5
    probs = np.exp(true) / np.exp(true).sum(1, keepdims=True)
    y5 = np.array([rng.choice(5, p=q) for q in probs])
    y = np.minimum(y5, 3)                                     # levels: grades 3 and 5 merge
    assert fit_temperature(true * 3, y) == pytest.approx(3, rel=0.05)


def test_expansion_grows_from_the_mode_to_the_larger_neighbour():
    p = np.array([[0.1, 0.2, 0.6, 0.1]])                       # mode 2, then 1, then 0 or 3 (tie goes left)
    assert expansion_mass(p)[0] == pytest.approx([0.8, 0.6, 0.0, 0.9])


def test_sets_are_contiguous_nested_and_contain_the_mode():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4), size=500)
    prev = expansion_mass(p)
    smaller = None
    for lam in (0.3, 0.6, 0.9, 0.99):
        member = prev <= lam
        assert member[np.arange(500), p.argmax(1)].all()
        for row in member:
            on = np.flatnonzero(row)
            assert (np.diff(on) == 1).all()
        if smaller is not None:
            assert (member | ~smaller).all()
        smaller = member


def test_conformal_lambda_rank_and_weights():
    scores = np.linspace(0.01, 1, 100)
    # ceil(101 * 0.9) = 91st smallest
    assert conformal_lambda(scores, 0.1) == pytest.approx(0.91)
    assert conformal_lambda(scores, 0.1, np.ones(100)) == pytest.approx(0.91)
    w = np.zeros(100)
    w[:10] = 1                                                 # n = 10, rank ceil(11 * 0.9) = 10
    assert conformal_lambda(scores, 0.1, w) == pytest.approx(0.10)
    assert conformal_lambda(np.array([0.5, 0.6]), 0.1) == np.inf   # too few points: the full set


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(4) * 0.5, size=20000)
    y = np.array([rng.choice(4, p=q) for q in p])
    prev = expansion_mass(p)
    scores = prev[np.arange(len(y)), y]
    lam = conformal_lambda(scores[:5000], 0.1)
    covered = (prev[5000:] <= lam)[np.arange(15000), y[5000:]]
    assert covered.mean() == pytest.approx(0.9, abs=0.015)


def test_tally_applies_the_referral_rule():
    member = np.array([[1, 0, 0, 0],    # {0}: not referred
                       [0, 1, 0, 0],    # {1}: not referred
                       [0, 0, 1, 0],    # {2}: referred, contains grade >= 2
                       [1, 1, 0, 0]],   # {0, 1}: referred, more than one grade
                      dtype=bool)
    y = np.array([0, 2, 2, 1])
    pred = np.array([0, 1, 2, 0])
    m = dict(zip(METRICS, tally(member, y, pred)))
    assert m['coverage'] == pytest.approx(3 / 4)
    assert m['set_size'] == pytest.approx(5 / 4)
    assert m['referral'] == pytest.approx(1 / 2)
    assert m['sens_ge2'] == pytest.approx(1 / 2)
    assert m['prev_ge2'] == pytest.approx(1 / 2)
    assert m['prev_referred'] == pytest.approx(1 / 2)
    assert m['prev_not_referred'] == pytest.approx(1 / 2)
    assert m['enrichment'] == pytest.approx(1)
    assert m['high_error_flagged'] == pytest.approx(0)
    assert m['coverage_l2'] == pytest.approx(1 / 2)


def test_tally_weights_count_a_joint_that_many_times():
    member = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=bool)
    y, pred = np.array([0, 3]), np.array([0, 2])
    m = dict(zip(METRICS, tally(member, y, pred, np.array([3.0, 1.0]))))
    assert m['coverage'] == pytest.approx(3 / 4)
    assert np.isnan(m['coverage_l1'])                          # no level-1 joint: undefined, not 0


def test_ece_is_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(3)
    prob = rng.random(50000)
    assert ece(prob, rng.random(50000) < prob) < 0.01
    assert ece(np.full(1000, 0.9), np.zeros(1000, dtype=bool)) == pytest.approx(0.9)
