import zipfile

import numpy as np
import pandas as pd
import pytest

from ramscore.compare import RunPicks
from ramscore.icc import case_weights, hand_totals, icc_a1, icc_c1
from ramscore.sensitivity import Scorer, contrast, hand_error_stats, icc_c1_weighted, read_metadata


def predictions(rng, keep, patients=20):
    """Four hands per patient with 16 joints each."""
    rows = []
    for p in range(patients):
        for h in range(4):
            gt = rng.choice([0, 1, 2, 3, 5], size=16, p=[0.6, 0.2, 0.1, 0.05, 0.05])
            pred = np.where(rng.random(16) < keep, gt, rng.choice([0, 1, 2, 3, 5], size=16))
            rows += [{'case': f'c{p}_{h}', 'patient': f'P{p:02d}', 'joint': f'j{i}', 'gt': int(g), 'pred': int(x)}
                     for i, (g, x) in enumerate(zip(gt, pred))]
    return pd.DataFrame(rows)


def test_weighted_consistency_matches_plain_form():
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=40), rng.normal(size=40)
    assert icc_c1_weighted(x1, x2)[0] == pytest.approx(icc_c1(x1, x2))


def test_weighted_consistency_equals_repeating_rows():
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=25), rng.normal(size=25)
    w = rng.integers(0, 4, size=25)
    assert icc_c1_weighted(x1, x2, w)[0] == pytest.approx(icc_c1(np.repeat(x1, w), np.repeat(x2, w)))


def test_consistency_ignores_an_offset_and_absolute_agreement_does_not():
    x = np.random.default_rng(2).normal(size=60)
    assert icc_c1_weighted(x, x + 5)[0] == pytest.approx(1.0)
    assert icc_a1(x, x + 5)[0] < 0.5


def test_hand_error_recovers_a_known_bias():
    rng = np.random.default_rng(3)
    totals = hand_totals(predictions(rng, keep=1.0))
    shifted = totals.assign(pred=totals['gt'] + 2)
    w = case_weights(totals['patient'].to_numpy(), 500, rng)
    picks = RunPicks(500, 0)([(0, 0), (0, 1)])
    stats = hand_error_stats([shifted, shifted], w, picks)
    assert stats['bias'] == pytest.approx(2.0) and stats['abs_error'] == pytest.approx(2.0)
    assert stats['bias_ci'] == pytest.approx([2.0, 2.0]) and stats['loa'] == pytest.approx([2.0, 2.0])


def test_scorer_point_matches_full_sample_and_contrast_detects_better_arm():
    good = [predictions(np.random.default_rng(s), keep=0.9) for s in (10, 11)]
    poor = [predictions(np.random.default_rng(s), keep=0.3) for s in (10, 11)]
    scorer = Scorer(good[0], 2000)
    levels = [(0, 0), (0, 1)]
    a, b = scorer.score(good, levels), scorer.score(poor, levels)
    expected = np.mean([icc_a1(hand_totals(df)['gt'], hand_totals(df)['pred'])[0] for df in good])
    assert a['icc'][0] == pytest.approx(expected)
    result = contrast(a, b, 'bacc4')
    assert result['diff'] > 0.3 and result['ci'][0] > 0 and result['p'] < 0.01


def test_read_metadata_handles_inline_and_numeric_cells(tmp_path):
    path = tmp_path / 'm.xlsx'
    sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
             '<row r="1"><c r="A1" t="inlineStr"><is><t>Mapped Image Stem</t></is></c><c r="B1" t="inlineStr"><is><t>isRA</t></is></c></row>'
             '<row r="2"><c r="A2" t="inlineStr"><is><t>JP_HMCRD_P0001_20210203_6791</t></is></c><c r="B2"><v>0</v></c></row>'
             '</sheetData></worksheet>')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/worksheets/sheet1.xml', sheet)
    frame = read_metadata(path)
    assert list(frame.columns) == ['Mapped Image Stem', 'isRA']
    assert frame.iloc[0]['isRA'] == '0' and frame.iloc[0]['Mapped Image Stem'].startswith('JP_HMCRD')
