"""Compares poses by how well their outlines sit on RGB edges, and against the human choices.

Exploratory: these questions were asked after the labels came in.

  python scripts/analyze_edge_fit.py [--dataset lmo]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from drpose.data import REPO
from drpose.stats import cluster_ids

BASE, REF = 'fp_cnos', 'dr_final'


def cluster_mean(d, values, n_boot=4000, seed=0):
  """Mean with a 95% interval that resamples whole scene-object groups."""
  ok = np.isfinite(values)
  c = (cluster_ids(d) if 'dataset' in d else d.scene.astype(str) + '_' + d.obj.astype(str)).to_numpy()[ok]
  v = values[ok]
  ids, inv = np.unique(c, return_inverse=True)
  sums, counts = np.bincount(inv, v), np.bincount(inv)
  idx = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))
  boot = sums[idx].sum(1) / counts[idx].sum(1)
  return v.mean(), *np.percentile(boot, [2.5, 97.5]), len(v)


def line(name, d, values, unit):
  m, lo, hi, n = cluster_mean(d, values)
  print(f'  {name:<42} {m:+.3f} {unit} [{lo:+.3f}, {hi:+.3f}]  n={n}')
  return dict(quantity=name, mean=round(m, 3), lo=round(lo, 3), hi=round(hi, 3), n=n, unit=unit)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', default='ycbv')
  args = p.parse_args()
  d = pd.read_csv(REPO / f'results/tables/edge_fit_{args.dataset}.csv').assign(dataset=args.dataset)
  out = []
  print('Distance from outline to nearest RGB edge, px (negative: the first pose sits closer to the edges)')
  for col in ['fit', 'fit_loose', 'fit_seen']:
    out.append(line(f'refined - FoundationPose ({col})', d, (d[f'{REF}_{col}'] - d[f'{BASE}_{col}']).to_numpy(), 'px'))
    out.append(line(f'GT - FoundationPose ({col})', d, (d[f'gt_{col}'] - d[f'{BASE}_{col}']).to_numpy(), 'px'))
  moved = (d[f'{REF}_fit'] - d[f'{BASE}_fit'])
  print(f'  refined closer on {(moved < 0).sum()}, farther on {(moved > 0).sum()}, of {len(d)}')
  print(f'  mean fit: FoundationPose {d[f"{BASE}_fit"].mean():.3f}, refined {d[f"{REF}_fit"].mean():.3f}, GT {d["gt_fit"].mean():.3f}')

  print('\nImage motion of the object center, px (+u right, +v down), and depth, mm')
  for a, b, name in [(REF, BASE, 'refined - FoundationPose'), ('gt', BASE, 'GT - FoundationPose'), ('gt', REF, 'GT - refined')]:
    out.append(line(f'{name}: du', d, (d[f'{a}_u'] - d[f'{b}_u']).to_numpy(), 'px'))
    out.append(line(f'{name}: dv', d, (d[f'{a}_v'] - d[f'{b}_v']).to_numpy(), 'px'))
    out.append(line(f'{name}: dz', d, (d[f'{a}_z'] - d[f'{b}_z']).to_numpy(), 'mm'))

  print('\nTranslation that would best seat the outline on RGB edges, px')
  for m, name in [(BASE, 'FoundationPose'), (REF, 'refined'), ('gt', 'GT')]:
    out.append(line(f'{name}: dx', d, d[f'{m}_dx'].to_numpy(float), 'px'))
    out.append(line(f'{name}: dy', d, d[f'{m}_dy'].to_numpy(float), 'px'))

  print('\nPer scene, median image motion of refined from FoundationPose, px')
  g = d.assign(du=d[f'{REF}_u'] - d[f'{BASE}_u'], dv=d[f'{REF}_v'] - d[f'{BASE}_v']).groupby('scene')
  print(g[['du', 'dv']].median().round(2).assign(n=g.size()).T.to_string())

  summary = 'edge_fit_summary.csv' if args.dataset == 'ycbv' else f'edge_fit_summary_{args.dataset}.csv'
  pd.DataFrame(out).to_csv(REPO / 'results/tables' / summary, index=False)
  if args.dataset != 'ycbv':   # the labels only ever covered YCB-V
    return

  r = pd.read_csv(REPO / 'results/labeling/ratings_unblinded.csv')
  r = r[(r.kind == 'refined_vs_base') & (r.winner != 'tie')].merge(d, left_on='target', right_on='key', suffixes=('', '_d'))
  print(f'\nDoes edge fit predict the person? {len(r)} decided pairs')
  for col in ['fit', 'fit_loose', 'fit_seen']:
    gain = (r[f'{BASE}_{col}'] - r[f'{REF}_{col}']).to_numpy()   # positive: refined sits closer to the edges
    human = np.where(r.winner == 'refined', 1, -1)
    keep = np.isfinite(gain) & (gain != 0)
    agree = int((np.sign(gain[keep]) == human[keep]).sum())
    print(f'  {col:<10} agrees {agree}/{keep.sum()} = {agree / keep.sum():.3f}, p={binomtest(agree, int(keep.sum()), 0.5).pvalue:.4f}; edges favor refined on {(gain[keep] > 0).sum()}')


if __name__ == '__main__':
  main()
