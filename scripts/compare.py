"""Compares methods on the same targets, with paired cluster-bootstrap intervals.

Comparing two result files directly is unsafe: they can cover different targets, and a baseline
file can contain test targets while a refiner has only seen dev. So this always restricts both
sides to one named subset and refuses to run unless every method covers all of it.

  python scripts/compare.py --base fp_cnos --methods dr_v0 icp fp_iters --subset dev_subset
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from drpose.data import REPO, subset_keys
from drpose.metrics import auc
from drpose.stats import METRICS, paired_bootstrap


def load(tag, dataset, condition, keys):
  df = pd.read_csv(REPO / 'results' / 'raw' / 'eval' / tag / dataset / f'{condition}.csv')
  df = df[df['key'].isin(keys)] if keys is not None else df
  if keys is not None and len(df) != len(keys):
    raise SystemExit(f'{tag}/{condition} covers {len(df)} of {len(keys)} targets in the subset')
  return df


def compare(base, methods, dataset, condition, keys, metrics, n_boot=1000):
  a = load(base, dataset, condition, keys)
  rows = []
  for m in [base] + methods:
    b = a if m == base else load(m, dataset, condition, keys)
    r = dict(method=m, n=len(b), te_mm=b['te'].median(), re_deg=b['re'].median(), mspd_px=b['mspd'].median(), seconds=b['seconds'].median())
    for name in metrics:
      r[name] = METRICS[name](b)
      if m != base:
        p = paired_bootstrap(a, b, METRICS[name], n_boot=n_boot)
        r[f'd_{name}'], r[f'd_{name}_lo'], r[f'd_{name}_hi'] = p['diff'], p['lo'], p['hi']
    if m != base:
      better = (b.sort_values('key')['mssd'].to_numpy() < a.sort_values('key')['mssd'].to_numpy() - 0.1).mean()
      worse = (b.sort_values('key')['mssd'].to_numpy() > a.sort_values('key')['mssd'].to_numpy() + 0.1).mean()
      r['better_%'], r['worse_%'] = 100 * better, 100 * worse
    rows.append(r)
  return pd.DataFrame(rows)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--base', default='fp_cnos')
  p.add_argument('--methods', nargs='+', required=True)
  p.add_argument('--subset', required=True, help="'dev_subset', 'test_subset', 'test_all', or 'all'")
  p.add_argument('--dataset', default='ycbv')
  p.add_argument('--condition', default='clean')
  p.add_argument('--metrics', nargs='+', default=['adds_auc', 'add_auc', 'ar_mssd', 'ar_mspd'])
  a = p.parse_args()
  keys = None if a.subset == 'all' else subset_keys(a.subset)
  out = compare(a.base, a.methods, a.dataset, a.condition, keys, a.metrics)
  pd.set_option('display.width', 220, 'display.max_columns', 40)
  cols = ['method', 'n', 'te_mm', 're_deg', 'mspd_px', 'seconds'] + a.metrics
  print(out[cols].round(2).to_string(index=False))
  print()
  for _, r in out.iloc[1:].iterrows():
    parts = [f"{m} {r[f'd_{m}']:+.2f} [{r[f'd_{m}_lo']:+.2f},{r[f'd_{m}_hi']:+.2f}]" for m in a.metrics]
    print(f"{r['method']:<16} vs {a.base}: " + '  '.join(parts) + f"   better {r['better_%']:.0f}% worse {r['worse_%']:.0f}%")


if __name__ == '__main__':
  main()
