"""Is missing depth lopsided around objects, in a way that could push a depth-only refiner sideways?

A depth cost can only be paid where depth exists. If the sensor's shadow leaves holes beside one side
of each object, an outline that overhangs that side pays nothing, while an overhang on the other side
lands on background depth and pays. Exploratory, like check_edge_fit.py.

  python scripts/check_depth_shadow.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import REPO, SceneCache, load_splits, load_targets, target_key

BAND = 6   # px outside the outline


def main():
  import numpy as np
  import pandas as pd
  from scipy.stats import spearmanr
  from drpose.cpu_render import silhouette

  keep = set(load_splits()['test_subset'])
  targets = sorted((t for t in load_targets('ycbv') if target_key(t) in keep), key=lambda t: t['obj'])

  def read(tag):
    with open(REPO / 'results/raw' / tag / 'ycbv/clean.jsonl') as f:
      return {r['key']: np.array(r['pose']) for r in map(json.loads, f) if r['pose'] is not None}

  base, ref = read('fp_cnos'), read('dr_final')
  cache, rows = SceneCache(), []
  for t in targets:
    k = target_key(t)
    if k not in base or k not in ref:
      continue
    _, depth, _, K = cache.load(t, 'cnos')
    m = cache.mesh(t['dataset'], t['scene'], t['obj'])
    sil = silhouette(np.asarray(m.vertices), np.asarray(m.faces), base[k], K, *depth.shape)
    hole = depth < 0.1
    row = dict(key=k, scene=t['scene'], obj=t['obj'])
    for name, axis, sign in [('left', 1, -1), ('right', 1, 1), ('up', 0, -1), ('down', 0, 1)]:
      band = np.zeros_like(sil)
      for s in range(1, BAND + 1):
        band |= np.roll(sil, sign * s, axis=axis)
      band &= ~sil
      row[name] = hole[band].mean() if band.any() else np.nan
    u = lambda p: K[0, 0] * p[0, 3] / p[2, 3] + K[0, 2]
    v = lambda p: K[1, 1] * p[1, 3] / p[2, 3] + K[1, 2]
    row['du'], row['dv'] = u(ref[k]) - u(base[k]), v(ref[k]) - v(base[k])
    rows.append(row)
  d = pd.DataFrame(rows).dropna()
  print(f'{len(d)} targets. Share of pixels with no depth in a {BAND} px band just outside the outline:')
  print((100 * d[['left', 'right', 'up', 'down']].mean()).round(1).to_string())
  print('by scene, right minus left, percentage points:')
  print((100 * (d.right - d.left).groupby(d.scene).mean()).round(1).to_frame('right-left').T.to_string())
  rho, p = spearmanr(d.right - d.left, d.du)
  print(f'rank correlation of (right - left holes) with rightward motion du: {rho:+.3f}, p={p:.4f}')
  rho, p = spearmanr(d.down - d.up, d.dv)
  print(f'rank correlation of (down - up holes) with downward motion dv: {rho:+.3f}, p={p:.4f}')
  hi = d[(d.right - d.left) > (d.right - d.left).median()]
  lo = d[(d.right - d.left) <= (d.right - d.left).median()]
  print(f'median du: more lopsided half {hi.du.median():+.2f} px, less lopsided half {lo.du.median():+.2f} px')
  d.to_csv(REPO / 'results/tables/depth_shadow_ycbv.csv', index=False)


if __name__ == '__main__':
  main()
