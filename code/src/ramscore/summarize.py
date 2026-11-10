"""Mean and SD of test metrics across runs (e.g. seeds).

Usage: python -m ramscore.summarize 'runs/e0_resnet34_s*'
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ('bacc5', 'bacc4', 'qwk', 'macro_f1', 'accuracy', 'mae', 'within1', 'recall_ge2', 'binary_sensitivity')


def main():
    runs = sorted(Path(p).parent for pattern in sys.argv[1:] for p in glob.glob(f'{pattern}/metrics.json'))
    if not runs:
        sys.exit('No metrics.json found')
    tests = [json.loads((r / 'metrics.json').read_text())['test'] for r in runs]
    print(f'{len(runs)} runs: ' + ', '.join(r.name for r in runs))
    for k in KEYS:
        v = np.array([t[k] for t in tests], dtype=float)
        print(f'  {k:20s} {np.nanmean(v):.4f} ± {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.4f}   '
              + ' '.join(f'{x:.4f}' for x in v))


if __name__ == '__main__':
    main()
