import re

import pytest

from ramscore.verify_tables import NUMBER, Row, check_table, matches, render, table_rows


def token(text):
    return NUMBER.search(text)


@pytest.mark.parametrize('printed, value, ok', [
    ('0.346', 0.3459, True),
    ('0.346', 0.3470, False),
    ('+0.053', 0.0532, True),
    ('-0.007', -0.0074, True),
    ('-0.007', 0.0074, False),
    ('3.9', 3.94, True),
    ('3.9', 3.96, False),
    ('<0.001', 0.0008, True),
    ('<0.001', 0.002, False),
])
def test_matches_uses_printed_precision(printed, value, ok):
    assert matches(token(printed), value) is ok


@pytest.mark.parametrize('printed, value, expected', [
    ('+0.020', 0.0216, '+0.022'),
    ('+0.005', -0.0012, '-0.001'),
    ('0.006', 0.0008, '0.001'),
    ('0.001', 0.0002, '<0.001'),
    ('1.13', 1.114, '1.11'),
    ('-0.045', -0.037, '-0.037'),
    ('+0.010', 0.0004, '+0.000'),      # a signed difference stays a number, it is not a p-value
])
def test_render_keeps_style(printed, value, expected):
    assert render(token(printed), value) == expected


LINES = """text
**Table 9.** Caption.

| Cell | Value | CI |
|---|---:|---:|
| `A` row | 0.100 | [+0.010, +0.200] |
| **B** row | 0.300 | [+0.020, +0.400] |

after""".split('\n')


def test_table_rows_finds_body_only():
    rows = table_rows(LINES, '**Table 9.**')
    assert [i for i, _ in rows] == [5, 6]
    assert table_rows(LINES, '**Table 1.**') is None


def test_check_table_reports_and_fixes_only_stale_numbers():
    lines = list(LINES)
    expected = [Row('A row', [0.1, 0.01, 0.2]), Row('B row', [0.3, 0.05, 0.4])]
    problems, count = check_table(lines, '**Table 9.**', expected)
    assert count == 2 and len(problems) == 1 and '+0.020 -> +0.050' in problems[0][1]
    assert lines == LINES                                   # no change without fix
    check_table(lines, '**Table 9.**', expected, fix=True)
    assert lines[6] == '| **B** row | 0.300 | [+0.050, +0.400] |' and lines[5] == LINES[5]
    assert check_table(lines, '**Table 9.**', expected)[0] == []


def test_check_table_flags_layout_changes():
    wrong_count = check_table(list(LINES), '**Table 9.**', [Row('A row', [0.1, 0.01, 0.2])])[0]
    assert re.search(r'2 rows in manuscript, 1 expected', wrong_count[0][1])
    too_many = check_table(list(LINES), '**Table 9.**', [Row('A row', [0.1, 0.1, 0.1, 0.1]), Row('B row', [0.3, 0.02, 0.4])])[0]
    assert '3 numbers, 4 expected' in too_many[0][1]
    wrong_label = check_table(list(LINES), '**Table 9.**', [Row('Z row', [0.1, 0.01, 0.2]), Row('B row', [0.3, 0.02, 0.4])])[0]
    assert "row 'Z row' not found" in wrong_label[0][1]
