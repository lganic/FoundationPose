"""Which way does refinement move objects, measured against the table they stand on?

Exploratory, written after refined poses turned out to move up in the image and toward the camera on
two datasets with different sensors. Fits the support plane from depth around each object and splits
the motion of the object's center into three parts: along the plane's normal (lift), along the
viewing ray, and what is left. Ground truth's offset from FoundationPose is split the same way.

  python scripts/check_motion_direction.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from drpose.data import REPO, SceneCache, load_splits, load_targets, target_key
from scripts.analyze_edge_fit import cluster_mean

BASE, REF = 'fp_cnos', 'dr_final'
TOL = 0.004   # m, a point within this of the plane supports it


def support_plane(depth, mask, K, rng, tries=150):
  """Unit normal facing the camera for the dominant plane in a ring around the object, or None."""
  import cv2
  m = mask.astype(np.uint8)
  ring = (cv2.dilate(m, np.ones((61, 61), np.uint8)) > 0) & (cv2.dilate(m, np.ones((9, 9), np.uint8)) == 0) & (depth > 0.1)
  v, u = np.nonzero(ring)
  if len(u) < 500:
    return None
  z = depth[v, u]
  P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
  best, best_n = None, 0
  for _ in range(tries):
    a, b, c = P[rng.choice(len(P), 3, replace=False)]
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-9:
      continue
    n /= np.linalg.norm(n)
    hits = (np.abs((P - a) @ n) < TOL).sum()
    if hits > best_n:
      best, best_n = n, hits
  if best is None or best_n < 0.5 * len(P):
    return None
  return -best if best @ P.mean(0) > 0 else best


def main():
  rng = np.random.default_rng(0)
  cache, summary = SceneCache(), []
  for dataset in ('ycbv', 'lmo'):
    keep = set(load_splits()['test_subset']) if dataset == 'ycbv' else None
    targets = [t for t in load_targets(dataset) if keep is None or target_key(t) in keep]
    targets.sort(key=lambda t: (t['obj'], t['scene'], t['im']))
    poses = {}
    for tag in (BASE, REF):
      with open(REPO / 'results/raw' / tag / dataset / 'clean.jsonl') as f:
        poses[tag] = {r['key']: np.array(r['pose']) for r in map(json.loads, f) if r['pose'] is not None}
    rows = []
    for t in targets:
      k = target_key(t)
      if k not in poses[BASE] or k not in poses[REF]:
        continue
      _, depth, mask, K = cache.load(t, 'cnos')
      if mask is None:
        continue
      n = support_plane(depth, mask, K, rng)
      if n is None:
        continue
      v = np.asarray(cache.mesh(dataset, t['scene'], t['obj']).vertices)
      center = 0.5 * (v.max(0) + v.min(0))
      place = lambda P: P[:3, :3] @ center + P[:3, 3]
      a, b, g = place(poses[BASE][k]), place(poses[REF][k]), place(t['gt_pose'])
      ray = -a / np.linalg.norm(a)   # toward the camera
      row = dict(key=k, dataset=dataset, scene=t['scene'], im=t['im'], obj=t['obj'], tilt=float(np.degrees(np.arccos(np.clip(n @ ray, -1, 1)))))
      for name, d in (('ref', b - a), ('gt', g - a)):
        row[f'{name}_lift'], row[f'{name}_ray'] = d @ n * 1e3, d @ ray * 1e3
        side = np.cross(n, ray)
        side /= max(np.linalg.norm(side), 1e-9)
        row[f'{name}_side'] = d @ side * 1e3
      rows.append(row)
    d = pd.DataFrame(rows)
    print(f'\n{dataset}: plane found for {len(d)} of {len(targets)} targets, median tilt of normal from viewing ray {d.tilt.median():.1f} deg')
    close = (d[['gt_lift', 'gt_ray', 'gt_side']].abs().max(axis=1) < 20).to_numpy()   # keeps GT's offset from being swamped by gross failures
    for col, sub in [('ref_lift', None), ('ref_ray', None), ('ref_side', None), ('gt_lift', close), ('gt_ray', close), ('gt_side', close)]:
      x = d if sub is None else d[sub]
      m, lo, hi, n = cluster_mean(x, x[col].to_numpy())
      print(f'  {col:<9} mean {m:+.3f} mm [{lo:+.3f}, {hi:+.3f}]  median {x[col].median():+.3f}  positive on {100 * (x[col] > 0).mean():.0f}%  n={n}')
      summary.append(dict(dataset=dataset, quantity=col, mean_mm=round(m, 3), lo=round(lo, 3), hi=round(hi, 3), median_mm=round(float(x[col].median()), 3), positive_pct=round(100 * float((x[col] > 0).mean()), 1), n=n))
  pd.DataFrame(summary).to_csv(REPO / 'results/tables/motion_direction.csv', index=False)


if __name__ == '__main__':
  main()
