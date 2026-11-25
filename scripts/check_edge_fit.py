"""Asks the picture, not ground truth, which pose fits: do outlines land on RGB edges?

Exploratory. Written after the human labels disagreed with the metrics, to find out why, so nothing
here counts as a held-out confirmation. Three measurements:

  fit:   mean distance from a pose's outline to the nearest image edge, capped so that hidden stretches
         of outline cost the same for every pose.
  shift: the small image translation that would best seat the outline on the edges. A pose that is
         right in the image needs none.
  reg:   the same translation search for the sensor's own depth edges against the RGB edges, with no
         pose involved. A steady non-zero answer means depth and color are not registered.

  python scripts/check_edge_fit.py --dataset ycbv --subset test_subset
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import REPO, SceneCache, load_splits, load_targets, target_key

CAP = 8.0      # px
SEARCH = 8     # px, each way
CANNY = [(50, 150), (30, 90)]


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', default='ycbv')
  p.add_argument('--subset', default='test_subset')
  p.add_argument('--methods', nargs='*', default=['fp_cnos', 'dr_final'])
  p.add_argument('--reg_only', action='store_true', help='skip the poses and measure only depth against color')
  p.add_argument('--no_reg', action='store_true', help='skip depth against color')
  p.add_argument('--condition', default='clean', help='the run the poses come from')
  p.add_argument('--max_frames', type=int, default=150)
  p.add_argument('--suffix', default='', help='keeps a side run from overwriting the main table')
  a = p.parse_args()

  import cv2
  import numpy as np
  import pandas as pd
  from drpose.cpu_render import silhouette

  targets = load_targets(a.dataset)
  if a.subset != 'all':
    keep = set(load_splits()[a.subset])
    targets = [t for t in targets if target_key(t) in keep]
  targets.sort(key=lambda t: (t['obj'], t['scene'], t['im']))

  cache = SceneCache()
  shifts = [(dx, dy) for dy in range(-SEARCH, SEARCH + 1) for dx in range(-SEARCH, SEARCH + 1)]

  def edge_distance(rgb, lo, hi):
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    return np.minimum(cv2.distanceTransform(255 - cv2.Canny(gray, lo, hi), cv2.DIST_L2, 5), CAP)

  def score(dist, us, vs, dx=0, dy=0):
    H, W = dist.shape
    return float(dist[np.clip(vs + dy, 0, H - 1), np.clip(us + dx, 0, W - 1)].mean())

  def best_shift(dist, us, vs):
    s = np.array([score(dist, us, vs, dx, dy) for dx, dy in shifts])
    return shifts[int(s.argmin())], s

  # Depth against color, per frame, with no pose involved.
  frames = {}
  for t in targets:
    frames.setdefault((t['scene'], t['im']), t)
  picks = [] if a.no_reg else list(frames.values())
  np.random.default_rng(0).shuffle(picks)
  picks = picks[:a.max_frames]
  rows, totals, by_side = [], {}, {}
  for t in picks:
    rgb, depth, _, K = cache.load(t, 'cnos')
    dist = edge_distance(rgb, *CANNY[0])
    # Keep the near side of each jump: the far side is the sensor's shadow. Each side of an object is
    # kept apart, because a misregistered sensor moves opposite sides the same way, while blur that
    # fattens objects moves them in opposite directions.
    side = {s: np.zeros(depth.shape, bool) for s in ('top', 'bottom', 'left', 'right')}
    for ax in (0, 1):
      d0 = depth[:-1, :] if ax == 0 else depth[:, :-1]
      d1 = depth[1:, :] if ax == 0 else depth[:, 1:]
      both = (d0 > 0.1) & (d1 > 0.1)
      jump = both & (np.abs(d0 - d1) > 0.02)
      if ax == 0:
        side['bottom'][:-1, :] |= jump & (d0 < d1)
        side['top'][1:, :] |= jump & (d1 < d0)
      else:
        side['right'][:, :-1] |= jump & (d0 < d1)
        side['left'][:, 1:] |= jump & (d1 < d0)
    near = side['top'] | side['bottom'] | side['left'] | side['right']
    vs, us = np.nonzero(near)
    if len(us) < 100:
      continue
    (dx, dy), s = best_shift(dist, us, vs)
    totals.setdefault(t['scene'], []).append(s)
    rows.append(dict(scene=t['scene'], im=t['im'], dx=dx, dy=dy, at_zero=s[shifts.index((0, 0))], at_best=s.min(), n=len(us)))
    for name, m in side.items():
      sv, su = np.nonzero(m)
      if len(su) < 20:
        continue
      line = [score(dist, su, sv, 0, o) if name in ('top', 'bottom') else score(dist, su, sv, o, 0) for o in range(-SEARCH, SEARCH + 1)]
      by_side.setdefault(name, []).append(line)
  tables = REPO / 'results/tables'
  tables.mkdir(parents=True, exist_ok=True)
  if rows:
    print('\nOne-dimensional best shift for each side of the objects, pooled over frames (px):')
    for name, lines in by_side.items():
      m = np.mean(lines, 0)
      print(f'  {name:>6}: {int(m.argmin()) - SEARCH:+d} along {"y" if name in ("top", "bottom") else "x"}, {len(lines)} frames, distance {m[SEARCH]:.3f} -> {m.min():.3f}')
    reg = pd.DataFrame(rows)
    print(f'\nDEPTH vs COLOR, {len(reg)} frames of {a.dataset}. Shift that moves depth edges onto color edges (px):')
    print(f'  per frame: dx mean {reg.dx.mean():+.2f} median {reg.dx.median():+.1f}, dy mean {reg.dy.mean():+.2f} median {reg.dy.median():+.1f}')
    print(f'  edge distance at zero shift {reg.at_zero.mean():.2f} px, at the best shift {reg.at_best.mean():.2f} px')
    for sc, ss in sorted(totals.items()):
      m = np.mean(ss, 0)
      print(f'  scene {sc:>3}: {len(ss):>3} frames, pooled best shift {shifts[int(m.argmin())]}, distance {m[shifts.index((0, 0))]:.3f} -> {m.min():.3f}')
    reg.to_csv(tables / f'depth_color_registration_{a.dataset}.csv', index=False)
  if a.reg_only:
    return

  # Outlines against color. Poses may come from a corrupted run, but the edges are always the clean photo's.
  poses = {}
  for tag in a.methods:
    with open(REPO / 'results/raw' / tag / a.dataset / f'{a.condition}.jsonl') as f:
      poses[tag] = {r['key']: np.array(r['pose']) for r in map(json.loads, f) if r['pose'] is not None}

  mesh, rows, current = None, [], None
  for t in targets:
    k = target_key(t)
    if not all(k in poses[m] for m in a.methods):
      continue
    if t['obj'] != current:
      m = cache.mesh(t['dataset'], t['scene'], t['obj'])
      mesh, current = (np.asarray(m.vertices), np.asarray(m.faces)), t['obj']
    rgb, depth, mask, K = cache.load(t, 'cnos')
    H, W = depth.shape
    dists = [edge_distance(rgb, lo, hi) for lo, hi in CANNY]
    near = None if mask is None else cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    row = dict(key=k, scene=t['scene'], im=t['im'], obj=t['obj'])
    for name, pose in [(m, poses[m][k]) for m in a.methods] + [('gt', t['gt_pose'])]:
      sil = silhouette(*mesh, pose, K, H, W).astype(np.uint8)
      contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      if not contours:
        continue
      pts = np.concatenate(contours)[:, 0]
      us, vs = pts[:, 0], pts[:, 1]
      row[f'{name}_fit'] = score(dists[0], us, vs)
      row[f'{name}_fit_loose'] = score(dists[1], us, vs)
      if near is not None:
        seen = near[vs, us]
        if seen.sum() > 20:
          row[f'{name}_fit_seen'] = score(dists[0], us[seen], vs[seen])
          (dx, dy), _ = best_shift(dists[0], us[seen], vs[seen])
          row[f'{name}_dx'], row[f'{name}_dy'] = dx, dy
      row[f'{name}_u'] = float(K[0, 0] * pose[0, 3] / pose[2, 3] + K[0, 2])
      row[f'{name}_v'] = float(K[1, 1] * pose[1, 3] / pose[2, 3] + K[1, 2])
      row[f'{name}_z'] = float(pose[2, 3] * 1e3)
    rows.append(row)
  d = pd.DataFrame(rows)
  name = f'edge_fit_{a.dataset}{a.suffix}.csv' if a.condition == 'clean' else f'edge_fit_{a.dataset}_{a.condition}{a.suffix}.csv'
  d.to_csv(tables / name, index=False)
  print(f'\n{len(d)} targets written to results/tables/{name}')


if __name__ == '__main__':
  main()
