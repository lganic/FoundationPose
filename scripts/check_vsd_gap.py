"""Does the change in VSD follow the gap between sensor depth and ground truth?

Exploratory. VSD compares depth surfaces, and it moved more than any other metric on both datasets,
up on YCB-V and down on LM-O. If the refiner pays for agreeing with a sensor that disagrees with
ground truth, VSD should fall as that gap grows, within each dataset. Bins are the ones
transfer_breakdown.py fixed before looking.

  python scripts/check_vsd_gap.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from drpose.data import REPO
from drpose.stats import METRICS, paired_bootstrap

TAB = REPO / 'results/tables'
BINS, NAMES = [0, 1, 3, 6, 30], ['under 1 mm', '1-3 mm', '3-6 mm', 'over 6 mm']


def main():
  rows = []
  for ds in ('ycbv', 'lmo'):
    t = pd.read_csv(TAB / f'transfer_targets_{ds}.csv').set_index('key')
    a, b = (pd.read_csv(REPO / 'results/raw/eval' / tag / ds / 'clean.csv') for tag in ('fp_cnos', 'dr_final'))
    a, b = (d[d['key'].isin(t.index)].sort_values('key').reset_index(drop=True) for d in (a, b))
    gap = pd.cut(t.loc[a['key'], 'depth_offset'].abs().to_numpy(), BINS, labels=NAMES, include_lowest=True)
    for name in NAMES:
      i = (gap == name)
      if i.sum() < 10:
        continue
      x, y = a[i].reset_index(drop=True), b[i].reset_index(drop=True)
      row = dict(dataset=ds, gap=name, n=int(i.sum()), fp_vsd=round(METRICS['ar_vsd'](x), 1))
      for m in ('ar_vsd', 'ar_mssd', 'ar_mspd'):
        r = paired_bootstrap(x, y, METRICS[m], n_boot=1000)
        row.update({f'd_{m}': round(r['diff'], 2), f'{m}_lo': round(r['lo'], 2), f'{m}_hi': round(r['hi'], 2)})
      row['clusters'] = r['n_clusters']
      rows.append(row)
  d = pd.DataFrame(rows)
  d.to_csv(TAB / 'vsd_by_depth_gap.csv', index=False)
  print(d.to_string(index=False))


if __name__ == '__main__':
  main()
