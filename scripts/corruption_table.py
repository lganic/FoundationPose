"""One row per corruption and method: the paired change from FoundationPose on the same corrupted frames.

The columns from GT are the planned comparison. edge_fit is the exploratory image check added after
the human labels: how far the outline moved from the edges of the clean photo, in px, where positive
is worse. A condition is skipped until every method covers the whole test subset.

  python scripts/corruption_table.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from drpose.corruptions import CONDITIONS
from drpose.data import REPO, load_splits
from scripts.compare import compare
from scripts.analyze_edge_fit import cluster_mean

METHODS = ['dr_final', 'icp', 'fp_iters']
METRICS = ['adds_auc', 'add_auc', 'ar_mssd', 'ar_mspd']
TAB = REPO / 'results/tables'


def main():
  keys = set(load_splits()['test_subset'])
  rows, skipped = [], []
  for c in CONDITIONS:
    try:
      t = compare('fp_cnos', METHODS, 'ycbv', c, keys, METRICS)
    except (SystemExit, FileNotFoundError) as e:
      skipped.append(c)
      continue
    base = t.iloc[0]
    for _, r in t.iloc[1:].iterrows():
      row = dict(condition=c, method=r['method'], fp_adds_auc=round(base['adds_auc'], 2), fp_ar_mspd=round(base['ar_mspd'], 2))
      for m in METRICS:
        row[f'd_{m}'], row[f'd_{m}_lo'], row[f'd_{m}_hi'] = (round(r[f'd_{m}{s}'], 2) for s in ('', '_lo', '_hi'))
      row['better_pct'], row['worse_pct'] = round(r['better_%'], 1), round(r['worse_%'], 1)
      rows.append(row)
    edge = TAB / ('edge_fit_ycbv.csv' if c == 'clean' else f'edge_fit_ycbv_{c}.csv')
    if edge.exists():
      d = pd.read_csv(edge)
      m, lo, hi, n = cluster_mean(d, (d['dr_final_fit'] - d['fp_cnos_fit']).to_numpy())
      rows[-len(METHODS)].update(edge_fit=round(m, 3), edge_fit_lo=round(lo, 3), edge_fit_hi=round(hi, 3), edge_farther=int((d['dr_final_fit'] > d['fp_cnos_fit']).sum()), edge_closer=int((d['dr_final_fit'] < d['fp_cnos_fit']).sum()))
  d = pd.DataFrame(rows)
  d.to_csv(TAB / 'corruption_refine.csv', index=False)
  pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_rows', 100)
  show = d[d.method == 'dr_final'].copy()
  show['ADD-S'] = show.apply(lambda r: f"{r.d_adds_auc:+.2f} [{r.d_adds_auc_lo:+.2f},{r.d_adds_auc_hi:+.2f}]", axis=1)
  show['MSPD'] = show.apply(lambda r: f"{r.d_ar_mspd:+.2f} [{r.d_ar_mspd_lo:+.2f},{r.d_ar_mspd_hi:+.2f}]", axis=1)
  cols = ['condition', 'fp_adds_auc', 'ADD-S', 'MSPD', 'better_pct', 'worse_pct'] + [c for c in ['edge_fit', 'edge_fit_lo', 'edge_fit_hi', 'edge_farther', 'edge_closer'] if c in show]
  print(show[cols].to_string(index=False))
  print('\nControls, ADD-S and MSSD change:')
  for _, r in d[d.method != 'dr_final'].iterrows():
    print(f"  {r.condition:<16} {r.method:<9} ADD-S {r.d_adds_auc:+.2f} [{r.d_adds_auc_lo:+.2f},{r.d_adds_auc_hi:+.2f}]  MSSD {r.d_ar_mssd:+.2f} [{r.d_ar_mssd_lo:+.2f},{r.d_ar_mssd_hi:+.2f}]")
  if skipped:
    print('\nnot ready:', ' '.join(skipped))


if __name__ == '__main__':
  main()
