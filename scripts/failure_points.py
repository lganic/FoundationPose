"""Baseline against refined at the places FoundationPose is weakest, on the held-out subset.

A failure is a pose whose worst surface point is off by more than 10% of the object's diameter, the
same rule as analyze_baseline.py. Fixed and broken count how many targets crossed that line.

  python scripts/failure_points.py [--subset test_all]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from drpose.data import REPO, subset_keys
from drpose.stats import METRICS, cluster_ids, paired_bootstrap

TAB = REPO / 'results/tables'
CONDITIONS = ['clean', 'occlusion_25', 'occlusion_50', 'occlusion_75', 'motion_blur_11', 'motion_blur_21', 'rgb_noise_25', 'rgb_noise_50',
              'depth_noise_10', 'depth_noise_20', 'depth_holes_40', 'depth_holes_60', 'mask_dilate_7', 'mask_erode_7', 'mask_box']


def load(tag, condition, keys):
  p = REPO / 'results/raw/eval' / tag / 'ycbv' / f'{condition}.csv'
  if not p.exists():
    return None
  d = pd.read_csv(p)
  d = d[d['key'].isin(keys)].sort_values('key').reset_index(drop=True)
  return d if len(d) == len(keys) else None


def failed(d):
  return (d['mssd'] / d['diameter'] > 0.10).to_numpy()


def counts(a, b):
  fa, fb = failed(a), failed(b)
  return dict(n=len(a), fp_failures=int(fa.sum()), fp_failure_pct=round(100 * fa.mean(), 1), fixed=int((fa & ~fb).sum()), broken=int((~fa & fb).sum()))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--subset', default='test_subset', help='test_all looks at clean frames only, with far more low-visibility targets')
  args = p.parse_args()
  keys = subset_keys(args.subset)
  suffix = '' if args.subset == 'test_subset' else f'_{args.subset}'
  rows = []
  for c in CONDITIONS:
    a, b = load('fp_cnos', c, keys), load('dr_final', c, keys)
    if a is None or b is None:
      continue
    rows.append(dict(condition=c, **counts(a, b)))
  by_condition = pd.DataFrame(rows)
  by_condition.to_csv(TAB / f'failures_by_condition{suffix}.csv', index=False)
  print(by_condition.to_string(index=False))

  a, b = load('fp_cnos', 'clean', keys), load('dr_final', 'clean', keys)
  edges, labels = [0, .5, .7, .9, 1.0], ['under 50%', '50-70%', '70-90%', '90-100%']
  a['bin'] = pd.cut(a['visib_fract'], edges, labels=labels, include_lowest=True)
  rows = []
  for name in labels:
    i = (a['bin'] == name).to_numpy()
    if i.sum() < 10:
      continue
    x, y = a[i].drop(columns='bin'), b[i]
    s = paired_bootstrap(x, y, METRICS['adds_auc'], n_boot=1000)
    m = paired_bootstrap(x, y, METRICS['ar_mspd'], n_boot=1000)
    rows.append(dict(visible=name, **counts(x, y), fp_adds=round(METRICS['adds_auc'](x), 2), d_adds=round(s['diff'], 2), lo=round(s['lo'], 2), hi=round(s['hi'], 2),
                     d_mspd=round(m['diff'], 2), mspd_lo=round(m['lo'], 2), mspd_hi=round(m['hi'], 2), clusters=cluster_ids(x).nunique()))
  by_visibility = pd.DataFrame(rows)
  by_visibility.to_csv(TAB / f'failures_by_visibility{suffix}.csv', index=False)
  print()
  print(by_visibility.to_string(index=False))


if __name__ == '__main__':
  main()
