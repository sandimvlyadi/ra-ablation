"""Check the manuscript result tables cell by cell against the analysis outputs in runs/.

Each table row is reduced to its decimal numbers, in reading order, and compared with the values built from the
analysis outputs at the precision printed in the manuscript. Integers in labels and run counts are not compared.
Prose is not checked: only the listed tables are.

--fix rewrites a stale number in place, keeping its sign style, its decimals and the rest of the row. A row with
too few numbers is only reported, because its layout changed.

Usage: uv run --group analysis python -m ramscore.verify_tables [--fix] [--runs runs] [--papers papers]
Exit status 1 when a row is stale or cannot be compared.
"""
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

NUMBER = re.compile(r'(?P<lt><\s*)?(?P<sign>[+-]?)(?P<int>\d*)\.(?P<frac>\d+)')
NUMBER_COMMA = re.compile(r'(?P<lt><\s*)?(?P<sign>[+-]?)(?P<int>\d*),(?P<frac>\d+)')     # Indonesian documents


@dataclass
class Row:
    label: str          # substring the manuscript row must contain
    values: list        # expected numbers in reading order, matched to the last numbers of the row


def tokens(line, comma=False):
    """Decimal numbers of one table row, in reading order."""
    return list((NUMBER_COMMA if comma else NUMBER).finditer(line))


def matches(token, value):
    """Whether a printed number equals value at its printed precision."""
    decimals = len(token['frac'])
    printed = float(f"{token['sign']}{token['int'] or 0}.{token['frac']}")
    if token['lt']:
        return value < printed
    return abs(value - printed) <= 0.5 * 10 ** -decimals + 1e-9


def render(token, value, comma=False):
    """value in the printed style of token; an unsigned value below the printed precision becomes <0.001."""
    decimals = len(token['frac'])
    text = f'{abs(value):.{decimals}f}'
    if float(text) == 0 and 0 < value and not token['sign'] and decimals == 3:
        return '<0,001' if comma else '<0.001'
    if not token['int']:
        text = text.lstrip('0') or '0'
    sign = '-' if value < 0 and float(text) != 0 else ('+' if token['sign'] == '+' else '')
    return sign + (text.replace('.', ',') if comma else text)


def table_rows(lines, caption):
    """(line index, text) of the body rows of the table after the caption; None if there is no such caption."""
    start = next((i for i, line in enumerate(lines) if line.startswith(caption)), None)
    if start is None:
        return None
    rows, in_table = [], False
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.startswith('|'):
            if in_table:
                rows.append((i, line))
            elif set(line) <= set('|-: '):
                in_table = True
        elif in_table:
            break
    return rows


def check_table(lines, caption, expected, fix=False, comma=False):
    """(problems, rows compared); stale numbers in lines are replaced when fix is set."""
    rows = table_rows(lines, caption)
    if rows is None:
        return [(caption, 'table not found')], 0
    if len(rows) != len(expected):
        return [(caption, f'{len(rows)} rows in manuscript, {len(expected)} expected')], len(rows)
    problems = []
    for (index, line), row in zip(rows, expected):
        if row.label not in line.replace('`', '').replace('*', ''):
            problems.append((caption, f'row {row.label!r} not found at: {line[:70]}'))
            continue
        found = tokens(line, comma)
        if len(found) < len(row.values):
            problems.append((caption, f'{row.label}: {len(found)} numbers, {len(row.values)} expected'))
            continue
        found = found[len(found) - len(row.values):]
        stale = [(t, v) for t, v in zip(found, row.values) if not matches(t, v)]
        if not stale:
            continue
        problems.append((caption, f'{row.label}: ' + ', '.join(f"{t[0].strip()} -> {render(t, v, comma)}" for t, v in stale)))
        if fix:
            for t, v in sorted(stale, key=lambda s: -s[0].start()):
                line = line[:t.start()] + render(t, v, comma) + line[t.end():]
            lines[index] = line
    return problems, len(rows)


class Sources:
    """The analysis outputs the tables are built from."""

    def __init__(self, runs):
        self.runs = Path(runs)
        self.e2 = self._json('e2_analysis/contrasts.json')
        self.e6 = self._json('e6_analysis/e6.json')
        self.icc = self._json('icc_analysis/icc.json')['contrasts']
        self.abl = self._json('e6abl/analysis/e6abl.json')
        self.direct = self._json('e4/analysis/direct.json')
        self.fewshot = self._json('e4/analysis/fewshot.json')
        self.anchor = self._json('e4/analysis/anchor.json')
        self.e5 = self._json('e5/analysis/e5.json')['cells']
        self.icc_cells = {(r['experiment'], r['cell']): r for r in self._csv('icc_analysis/cells.csv')}
        self.e2_runs = self._csv('e2_analysis/runs.csv')
        self.e3_runs = self._csv('e3_analysis/runs.csv')
        self.sensitivity = self._json('sensitivity_analysis/sensitivity.json')

    def _json(self, path):
        return json.loads((self.runs / path).read_text())

    def _csv(self, path):
        with open(self.runs / path) as handle:
            return list(csv.DictReader(handle))

    def probe(self, name):
        return self._json(f'e1_probe/{name}.json')

    def compare_e1(self, backbone, kind, frac):
        return self._json(f'compare_e1/{backbone}_{kind}_f{frac}.json')

    def run_mean(self, table, metric, **where):
        picked = [float(r[metric]) for r in table if all(r[k] == v for k, v in where.items())]
        if not picked:
            raise KeyError(f'no runs for {where}')
        return sum(picked) / len(picked)

    def icc_cell(self, experiment, cell):
        row = self.icc_cells[(experiment, cell)]
        return float(row['icc_a1']), json.loads(row['ci']), float(row['icc_c1'])

    def icc_contrast(self, family, contrast):
        return next(c for c in self.icc if c['family'] == family and c['contrast'] == contrast)


def p1_table2(s):
    arms = [
        ('A0 Random ResNet34', 'r34', 'random_resnet34', None),
        ('A1 ImageNet ResNet34', 'r34', 'imagenet_resnet34', 'base'),
        ('A4 DINO ResNet34', 'r34', 'e1_dino_resnet34_ep100', 'vs'),
        ('A5 DINO+ChronoCon ResNet34', 'r34', 'e1_dinochrono_resnet34_ep100', 'vs'),
        ('A0 Random ViT-S/16', 'vits', 'random_vit_small_patch16_224', None),
        ('A1 ImageNet ViT-S/16', 'vits', 'imagenet_vit_small_patch16_224', 'base'),
        ('A3 MAE ViT-S/16', 'vits', 'e1_mae_vits_ep400', 'vs'),
        ('A4 DINO ViT-S/16', 'vits', 'e1_dino_vits_ep100', 'vs'),
        ('A5 DINO+ChronoCon ViT-S/16', 'vits', 'e1_dinochrono_vits_ep100', 'vs'),
    ]
    rows = []
    for label, bb, name, role in arms:
        probe = s.probe(name)
        lin, knn = probe['probes']['f1.0']['linear'], probe['probes']['f1.0']['knn']
        lin10 = s.compare_e1(bb, 'linear', '0.1')
        knn10 = s.compare_e1(bb, 'knn', '0.1')
        values = [lin['bacc4']]
        if role:
            values += lin['ci95']['bacc4']
        values += [knn['bacc4'], lin10['point'][name]['bacc4'], knn10['point'][name]['bacc4'], lin['qwk'],
                   probe['collapse']['effective_rank']]
        if role == 'vs':
            base = 'imagenet_resnet34' if bb == 'r34' else 'imagenet_vit_small_patch16_224'
            c = lin10['vs_baseline'][f'{name}-{base}']['bacc4']
            values += [c['diff'], *c['ci_hier'], c['p_holm']]
        rows.append(Row(label, values))
    return rows


def p1_table3(s):
    rows = []
    fractions = {'a5': ['0.1', '1.0']}
    for label, backbone, a in [('ResNet34 A0', 'resnet34', 'a0'), ('ResNet34 A1', 'resnet34', 'a1'),
                               ('ResNet34 A4', 'resnet34', 'a4'), ('ResNet34 A5', 'resnet34', 'a5'),
                               ('ViT-S A0', 'vits', 'a0'), ('ViT-S A1', 'vits', 'a1'), ('ViT-S A3', 'vits', 'a3'),
                               ('ViT-S A4', 'vits', 'a4'), ('ViT-S A5', 'vits', 'a5')]:
        values = []
        for frac in fractions.get(a, ['0.01', '0.05', '0.1', '0.25', '1.0']):
            for b in ('b0', 'b4', 'b6'):
                values.append(s.run_mean(s.e2_runs, 'bacc4', A=a, backbone=backbone, B=b, frac=frac))
        rows.append(Row(label, values))
    return rows


def _e2(s, backbone, b, contrast, frac='0.1'):
    return next(c for c in s.e2 if c['backbone'] == backbone and c['B'] == b and c['contrast'] == contrast
                and c['frac'] == frac)


def p1_table4(s):
    rows = []
    for label, backbone, contrast in [('DINO R34 - Scratch R34', 'resnet34', 'a4-a0'),
                                      ('DINO R34 - ImageNet R34', 'resnet34', 'a4-a1'),
                                      ('DINO ViT-S - Scratch ViT-S', 'vits', 'a4-a0'),
                                      ('DINO ViT-S - ImageNet ViT-S', 'vits', 'a4-a1'),
                                      ('MAE ViT-S - Scratch ViT-S', 'vits', 'a3-a0'),
                                      ('MAE ViT-S - ImageNet ViT-S', 'vits', 'a3-a1')]:
        values = []
        for b in ('b0', 'b4'):
            c = _e2(s, backbone, b, contrast)
            values += [c['bacc4_diff'], *c['bacc4_ci_hier'], c['p_holm']]
        rows.append(Row(label, values))
    return rows


def p1_table5(s):
    rows = []
    for label, cell in [('10%, B0', '0.1_b0'), ('10%, B4', '0.1_b4'), ('10%, B6', '0.1_b6'),
                        ('100%, B0', '1.0_b0'), ('100%, B4', '1.0_b4'), ('100%, B6', '1.0_b6')]:
        summary, contrasts = s.e6[cell]['summary'], {c['contrast']: c for c in s.e6[cell]['contrasts']}
        values = []
        for arm in ('mae_llrd', 'mae_e2', 'imagenet_llrd', 'imagenet_e2', 'dino_llrd'):
            values += [summary[arm]['bacc4'], summary[arm]['qwk']]
        for name in ('mae_llrd - mae_e2', 'mae_llrd - imagenet_llrd'):
            c = contrasts[name]
            values += [c['bacc4']['diff'], *c['bacc4']['ci'], c['bacc4'].get('p_holm', c['bacc4']['p'])]
        rows.append(Row(label, values))
    return rows


def p1_table6(s):
    rows = []
    for label, b in [('B0 Ordinal', 'b0'), ('B1 Weight', 'b1'), ('B2 Balanced', 'b2'), ('B3 CORN', 'b3'),
                     ('B4 CORN', 'b4'), ('B5 Hierarchical', 'b5'), ('B6 Decoupled', 'b6')]:
        values = []
        for frac in ('0.1', '1.0'):
            for a in ('a0', 'a1', 'a4'):
                for metric in ('bacc4', 'qwk', 'recall_ge2'):
                    values.append(s.run_mean(s.e3_runs, metric, A=a, B=b, frac=frac))
        rows.append(Row(label, values))
    return rows


def p1_icc(s):
    rows = []
    for label, a in [('ResNet34 A0', 'a0'), ('ResNet34 A1', 'a1'), ('ResNet34 A4', 'a4')]:
        values = []
        for frac in ('0.1', '1.0'):
            for b in ('b0', 'b1', 'b2', 'b4', 'b6'):
                icc, ci, _ = s.icc_cell('e3', f'{a}_{b}_{frac}')
                values += [icc, *ci]
        rows.append(Row(label, values))
    for label, arm in [('ViT-S A1 ImageNet-LLRD', 'imagenet_llrd'), ('ViT-S A3 MAE-LLRD', 'mae_llrd'),
                       ('ViT-S A4 DINO-LLRD', 'dino_llrd')]:
        values = []
        for frac in ('0.1', '1.0'):
            for b in ('b0', 'b4', 'b6'):
                summary = s.e6[f'{frac}_{b}']['summary'][arm]
                values += [summary['icc'], *summary['icc_ci']]
        rows.append(Row(label, values))
    return rows


def p1_ablation(s):
    rows = []
    for key, title in (('mask', 'Bone-mask channel'), ('levels', '3-class training'), ('pool', '+RAM-W600 pool')):
        for c in s.abl[key]:
            values = [c['bacc4']['base'], c['bacc4']['variant'], c['bacc4']['diff'], *c['bacc4']['ci'],
                      c['bacc4']['p_holm'], c['qwk']['diff'], c['qwk']['p']]
            if 'icc' in c:
                values += [c['icc']['diff'], c['icc']['p']]
            rows.append(Row(title, values))
    return rows


def p1_rq2(s):
    rows = []
    for r in s.sensitivity['rq2']:
        values = []
        for m in ('bacc4', 'qwk', 'icc'):
            values += [r[m]['diff'], *r[m]['ci'], r[m]['p_holm']]
        rows.append(Row(r['contrast'].upper(), values))
    return rows


def p1_frozen(s):
    rows = []
    for r in s.sensitivity['frozen']:
        if r['head'] == 'b4':
            values = []
            for m in ('bacc4', 'qwk', 'icc'):
                values += [r[f'frozen_{m}'], r[f'fine_{m}'], r[m]['diff'], *r[m]['ci']]
            rows.append(Row(r['encoder'].upper(), values))
    return rows


def p1_hand(s):
    rows = []
    for r in s.sensitivity['hand']:
        if r['head'] in ('b1', 'b2', 'b4'):
            a, low = r['all'], r['low']
            rows.append(Row(r['head'].upper(), [a['bias'], *a['bias_ci'], *a['loa'], a['abs_error'], *a['abs_error_ci'],
                                                low['icc'], *low['icc_ci'], low['abs_error'], *low['abs_error_ci']]))
    return rows


def p1_centre(s):
    rows = []
    for r in s.sensitivity['centre_metrics']:
        if r['centre'] in ('HMCRD', 'SARC'):
            values = []
            for m in ('bacc4', 'qwk', 'icc'):
                values += [r[m], *r[f'{m}_ci']]
            rows.append(Row(r['centre'], values))
    return rows


def p2_table2(s):
    d, rows = s.direct, []
    for label, k in [('K1: DINO', 'K1'), ('K2: Scratch', 'K2'), ('K3: DINO', 'K3'), ('K4: ImageNet', 'K4')]:
        p, ci = d['point'][k], d['ci'][k]
        icc_in, _, _ = s.icc_cell('e4_internal', k)
        icc_ex, icc_ci, _ = s.icc_cell('e4_external', f'{k}_all_0')
        rows.append(Row(label, [p['internal']['bacc4'], *ci['internal']['bacc4'], p['external']['bacc4'],
                                *ci['external']['bacc4'], p['drop']['bacc4'], *ci['drop']['bacc4'],
                                p['internal']['qwk'], p['external']['qwk'], *ci['external']['qwk'],
                                icc_in, icc_ex, *icc_ci]))
    for c in d['contrasts'][:2]:
        icc = s.icc_contrast('E4 H3 drop', c['contrast'])
        rows.append(Row(c['contrast'], [c['bacc4_diff'], *c['bacc4_ci_hier'], c['bacc4_p_holm'], c['qwk_diff'],
                                        *c['qwk_ci_hier'], c['qwk_p_holm'], icc['diff'], *icc['ci'], icc['p_holm']]))
    return rows


def p2_table3(s):
    f, rows = s.fewshot, []
    for label, k in [('K1: DINO', 'K1'), ('K2: Scratch', 'K2'), ('K3: DINO', 'K3'), ('K4: ImageNet', 'K4')]:
        values = []
        for n in (0, 5, 20):
            p = f['point'][f'{k}_k{n}']
            icc, _, _ = s.icc_cell('e4_external', f'{k}_fs_{n}')
            values.append(p['bacc4'])
            if n == 20:
                values += f['ci'][f'{k}_k{n}']['bacc4']
            values += [p['qwk'], icc]
        rows.append(Row(label, values))
    for label, name in [('Secondary K3 - K2', 'K3 - K2'), ('Secondary K3 - K4', 'K3 - K4')]:
        values = []
        for n in (0, 5, 20):
            c = next(c for c in f['contrasts'] if c['k'] == n and c['contrast'] == name)
            values.append(c['bacc4_diff'])
            if c['bacc4_p_holm'] < 0.05:        # the manuscript prints a Holm p only for a significant BACC4 cell
                values.append(c['bacc4_p_holm'])
            values += [c['qwk_diff'], c['qwk_p_holm']]
        rows.append(Row(label, values))
    return rows


def p2_table4(s):
    t, a = s.anchor['test'], s.anchor['all']
    rows = [Row('Human Specialists', [t['point']['reader']['bacc4'], *t['ci']['reader']['bacc4'],
                                      t['point']['reader']['qwk'], *t['ci']['reader']['qwk']])]
    for k in ('K1', 'K2', 'K3', 'K4'):
        d, f = f'{k} direct', f'{k} few-shot 20'
        values = [t['point'][d]['bacc4'], t['point'][f]['bacc4'], t['diffs'][d]['bacc4']['diff'],
                  t['diffs'][f]['bacc4']['diff']]
        if k == 'K4':
            values += t['diffs'][f]['bacc4']['ci']
        values += [t['point'][d]['qwk'], t['point'][f]['qwk'], t['diffs'][d]['qwk']['diff'], t['diffs'][f]['qwk']['diff']]
        if k == 'K4':
            values += t['diffs'][f]['qwk']['ci']
        rows.append(Row(k, values))
    rows.append(Row('Human Specialists', [a['point']['reader']['bacc4'], *a['ci']['reader']['bacc4'],
                                          a['point']['reader']['qwk'], *a['ci']['reader']['qwk']]))
    d = 'K4 direct'
    rows.append(Row('K4', [a['point'][d]['bacc4'], *a['ci'][d]['bacc4'], a['diffs'][d]['bacc4']['diff'],
                           *a['diffs'][d]['bacc4']['ci'], a['point'][d]['qwk'], *a['ci'][d]['qwk'],
                           a['diffs'][d]['qwk']['diff'], *a['diffs'][d]['qwk']['ci']]))
    return rows


def p2_table5(s):
    rows = []
    for k in ('K1', 'K2', 'K3', 'K4'):
        c = s.e5[f'{k}_internal_k0_raw']
        cal = c['calibration']
        rows.append(Row(k, [cal['T'], cal['lambda'], cal['nll_before'], cal['nll_after'],
                            c['coverage']['mean'], *c['coverage']['ci'],
                            c['coverage_l0']['mean'], c['coverage_l1']['mean'], c['coverage_l2']['mean'],
                            c['coverage_l3']['mean'], c['set_size']['mean'], c['referral']['mean'],
                            *c['referral']['ci'], c['sens_ge2']['mean'], c['enrichment']['mean'],
                            *c['enrichment']['ci'], c['prev_not_referred']['mean']]))
    return rows


def p2_table6(s):
    rows = []
    for label, regime in [('Raw B2: RAM-H1200 val -> RAM-H1200 test', 'internal_k0_raw'),
                          ('Log-prior', 'internal_k0_prior'),
                          ('Raw B2: RAM-H1200 val -> RheumaVIT test', 'external_k0_raw'),
                          ('Raw B2: RheumaVIT 5-pt', 'local_k5_raw'),
                          ('Raw B2: RheumaVIT 20-pt', 'local_k20_raw'),
                          ('Log-prior', 'external_k0_prior')]:
        cells = [s.e5[f'{k}_{regime}'] for k in ('K1', 'K2', 'K3', 'K4')]
        values = [c['coverage']['mean'] for c in cells] + [c['set_size']['mean'] for c in cells]
        values += [c['referral']['mean'] for c in cells]
        k4 = cells[3]
        values += [k4['sens_ge2']['mean'], k4['enrichment']['mean'], *k4['enrichment']['ci'],
                   k4['prev_not_referred']['mean']]
        rows.append(Row(label, values))
    return rows


def id_table41(s):
    return [Row(r.label.replace('/16', ''), r.values) for r in p1_table2(s)]


def id_table42(s):
    labels = ('A4 DINO R34', 'A4 DINO R34', 'A4 DINO ViT-S', 'A4 DINO ViT-S', 'A3 MAE ViT-S', 'A3 MAE ViT-S')
    return [Row(label, r.values) for label, r in zip(labels, p1_table4(s))]


def id_table43(s):
    return p1_table5(s)


def id_table44(s):
    rows = []
    for label, b in [('B0 Ordinal', 'b0'), ('B1 Weight', 'b1'), ('B2 Balanced', 'b2'), ('B3 CORN', 'b3'),
                     ('B4 CORN', 'b4'), ('B5 Hierarchical', 'b5'), ('B6 Decoupled', 'b6')]:
        values = []
        for frac in ('0.1', '1.0'):
            for a in ('a0', 'a1', 'a4'):
                values += [s.run_mean(s.e3_runs, 'bacc4', A=a, B=b, frac=frac), s.run_mean(s.e3_runs, 'qwk', A=a, B=b, frac=frac),
                           s.icc_cell('e3', f'{a}_{b}_{frac}')[0]]
                if frac == '0.1':
                    values.append(s.run_mean(s.e3_runs, 'recall_ge2', A=a, B=b, frac=frac))
        rows.append(Row(label, values))
    return rows


def id_table45(s):
    d, f, t, rows = s.direct, s.fewshot, s.anchor['test'], []
    for k in ('K1', 'K2', 'K3', 'K4'):
        p, ci = d['point'][k], d['ci'][k]
        gap = t['diffs'][f'{k} few-shot 20']['bacc4']
        rows.append(Row(k, [p['internal']['bacc4'], p['external']['bacc4'], p['drop']['bacc4'], *ci['drop']['bacc4'], p['external']['qwk'],
                            f['point'][f'{k}_k5']['qwk'], f['point'][f'{k}_k20']['qwk'], f['point'][f'{k}_k5']['bacc4'],
                            f['point'][f'{k}_k20']['bacc4'], s.icc_cell('e4_external', f'{k}_all_0')[0],
                            s.icc_cell('e4_external', f'{k}_fs_20')[0], gap['diff'], *gap['ci']]))
    rows.append(Row('Pembaca', []))
    return rows


def id_table46(s):
    rows = []
    for regime, label in (('raw', 'Mentah'), ('prior', 'Log-Prior')):
        for k in ('K1', 'K2', 'K3', 'K4'):
            c, e = s.e5[f'{k}_internal_k0_{regime}'], s.e5[f'{k}_external_k0_{regime}']
            cal = c['calibration']
            values = [cal['T'], cal['lambda'], c['coverage']['mean'], *c['coverage']['ci'], c['set_size']['mean'], c['referral']['mean'],
                      c['sens_ge2']['mean'], c['enrichment']['mean'], *c['enrichment']['ci']]
            if regime == 'raw':
                values.append(c['coverage_l2']['mean'])
            values += [cal['nll_after'], cal['ece_after_l0'], e['coverage']['mean'], e['referral']['mean']]
            rows.append(Row(f'{k} ', values))
    return rows


TABLES = {
    'paper1/manuscript_en.md': [('**Table 2.**', p1_table2), ('**Table 3.**', p1_table3), ('**Table 4.**', p1_table4),
                                ('**Table 5.**', p1_table5), ('**Table 6.**', p1_table6), ('**Table 7.**', p1_rq2),
                                ('**Table 8.**', p1_icc), ('**Table 9.**', p1_hand), ('**Table 10.**', p1_ablation),
                                ('**Table 11.**', p1_frozen), ('**Table 12.**', p1_centre)],
    'paper2/manuscript_en.md': [('**Table 2.**', p2_table2), ('**Table 3.**', p2_table3), ('**Table 4.**', p2_table4),
                                ('**Table 5.**', p2_table5), ('**Table 6.**', p2_table6)],
    'proposal/proposal_tesis_id.md': [('**Tabel 4.1.**', id_table41), ('**Tabel 4.2.**', id_table42), ('**Tabel 4.3.**', id_table43),
                                      ('**Tabel 4.4.**', id_table44), ('**Tabel 4.5.**', id_table45), ('**Tabel 4.6.**', id_table46)],
}
COMMA_DECIMAL = {'proposal/proposal_tesis_id.md'}


def verify(papers, runs, fix=False):
    """{document: problems} and the number of rows compared."""
    sources, report, total = Sources(runs), {}, 0
    for document, tables in TABLES.items():
        path = Path(papers) / document
        lines = path.read_text().split('\n')
        problems = []
        for caption, build in tables:
            found, count = check_table(lines, caption, build(sources), fix=fix, comma=document in COMMA_DECIMAL)
            problems += [(f'{document} {c}', reason) for c, reason in found]
            total += count
        report[document] = problems
        if fix:
            path.write_text('\n'.join(lines))
    return report, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--papers', default='papers')
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--fix', action='store_true', help='rewrite stale numbers in place')
    args = parser.parse_args()
    report, total = verify(args.papers, args.runs, fix=args.fix)
    problems = [p for found in report.values() for p in found]
    for where, reason in problems:
        print(f'{where} {reason}')
    print(f'{total} rows compared, {len(problems)} stale or not comparable' + (' (fixed where the layout allowed)' if args.fix else ''))
    return 1 if problems and not args.fix else 0


if __name__ == '__main__':
    sys.exit(main())
