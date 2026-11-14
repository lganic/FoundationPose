"""Scores a results file against ground truth and writes one CSV row per target.

  python scripts/evaluate.py --tag fp_cnos --dataset ycbv --condition clean
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from drpose.data import REPO, load_targets, target_key
from drpose.metrics import Evaluator
from drpose.stats import summarize


def viewpoint(gt_pose):
  """Where the camera sits in the object's own frame: elevation and azimuth in degrees, distance in m."""
  R, t = gt_pose[:3, :3], gt_pose[:3, 3]
  c = -R.T @ t
  dist = np.linalg.norm(c)
  c = c / dist
  return float(np.degrees(np.arcsin(np.clip(c[2], -1, 1)))), float(np.degrees(np.arctan2(c[1], c[0]))), float(dist)


def evaluate(tag, dataset, condition, use_vsd=True):
  src = REPO / 'results' / 'raw' / tag / dataset / f'{condition}.jsonl'
  rows = {}
  with open(src) as f:
    for line in f:
      if line.strip():
        r = json.loads(line)
        rows[r['key']] = r
  targets = {target_key(t): t for t in load_targets(dataset)}
  ev = Evaluator(dataset, use_vsd=use_vsd)
  out = []
  for key in sorted(rows, key=lambda k: (targets[k]['scene'], targets[k]['im'], targets[k]['obj'])):
    t, r = targets[key], rows[key]
    pose = None if r['pose'] is None else np.array(r['pose'])
    elev, azim, dist = viewpoint(t['gt_pose'])
    out.append(dict(key=key, dataset=dataset, scene=t['scene'], im=t['im'], obj=t['obj'], visib_fract=t['visib_fract'], px_visib=t['px_visib'],
                    diameter=ev.diameter(t['obj']), elev=elev, azim=azim, dist=dist, seconds=r.get('seconds', np.nan), **ev.errors(t, pose)))
  df = pd.DataFrame(out)
  dst = REPO / 'results' / 'raw' / 'eval' / tag / dataset / f'{condition}.csv'
  dst.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(dst, index=False)
  return df


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--tag', default='fp_cnos')
  p.add_argument('--dataset', default='ycbv')
  p.add_argument('--condition', default='clean')
  p.add_argument('--no_vsd', action='store_true')
  a = p.parse_args()
  df = evaluate(a.tag, a.dataset, a.condition, use_vsd=not a.no_vsd)
  print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summarize(df).items()})


if __name__ == '__main__':
  main()
