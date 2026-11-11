"""Why did the refiner's gain on YCB-V not carry over to LM-O?

Exploratory: these questions were asked after the LM-O result came in, so nothing here is a held-out
confirmation. LM-O's headline comparison in compare.py is the planned test; this only looks inside it.
Three explanations, each with a CPU-only check:

  masks:     the refiner fits depth through the detector's mask. Is the loss confined to poor masks?
  occlusion: is the loss confined to mostly hidden objects, as it was at low visibility on YCB-V?
  depth:     a refiner that fits sensor depth can only approach ground truth if the two agree. How far
             does the sensor sit from the ground-truth surface on each dataset?

  python scripts/transfer_breakdown.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from drpose.data import REPO, SceneCache, load_splits, load_targets, target_key
from drpose.stats import METRICS, cluster_ids, paired_bootstrap

TAB = REPO / 'results/tables'
BASE, REF = 'fp_cnos', 'dr_final'


def rows_for(dataset):
  """Both methods' scores, on the targets that dataset's comparison uses."""
  keep = set(load_splits()['test_subset']) if dataset == 'ycbv' else None
  out = []
  for tag in (BASE, REF):
    d = pd.read_csv(REPO / 'results/raw/eval' / tag / dataset / 'clean.csv')
    d = d if keep is None else d[d['key'].isin(keep)]
    out.append(d.sort_values('key').reset_index(drop=True))
  assert (out[0]['key'] == out[1]['key']).all()
  return out


def visible_vertices(mesh, pose, K, shape):
  """Camera-facing mesh vertices at a pose, as pixels and depths. The nearest vertex wins each pixel."""
  v = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
  n = np.asarray(mesh.vertex_normals) @ pose[:3, :3].T
  v = v[(v * n).sum(1) < 0]
  u = np.round(K[0, 0] * v[:, 0] / v[:, 2] + K[0, 2]).astype(int)
  w = np.round(K[1, 1] * v[:, 1] / v[:, 2] + K[1, 2]).astype(int)
  ok = (u >= 0) & (u < shape[1]) & (w >= 0) & (w < shape[0])
  u, w, z = u[ok], w[ok], v[ok, 2]
  order = np.argsort(z)
  u, w, z = u[order], w[order], z[order]
  _, first = np.unique(w * shape[1] + u, return_index=True)
  return u[first], w[first], z[first]


def per_target(dataset):
  """CNOS mask quality and sensor-minus-GT depth for each target, without a GPU."""
  keep = set(load_splits()['test_subset']) if dataset == 'ycbv' else None
  targets = [t for t in load_targets(dataset) if keep is None or target_key(t) in keep]
  targets.sort(key=lambda t: (t['obj'], t['scene'], t['im']))
  cache, rows = SceneCache(), []
  for t in targets:
    _, depth, truth, K = cache.load(t, 'gt')
    _, _, found, _ = cache.load(t, 'cnos')
    truth = truth.astype(bool)
    iou = np.nan if found is None else (truth & found.astype(bool)).sum() / max((truth | found.astype(bool)).sum(), 1)
    u, v, z = visible_vertices(cache.mesh(dataset, t['scene'], t['obj']), t['gt_pose'], K, depth.shape)
    seen = truth[v, u] & (depth[v, u] > 0.1)
    r = (depth[v, u] - z)[seen] * 1e3
    r = r[np.abs(r) < 30]   # past this it is another surface, not the sensor's error on this one
    rows.append(dict(key=target_key(t), iou=iou, n_depth=len(r), depth_offset=np.median(r) if len(r) >= 30 else np.nan))
  return pd.DataFrame(rows)


def split(a, b, groups, label):
  fail = lambda d: (d['mssd'] / d['diameter'] > .10).to_numpy()
  out = []
  for name in groups.cat.categories:
    i = (groups == name).to_numpy()
    if i.sum() < 10:
      continue
    x, y = a[i].reset_index(drop=True), b[i].reset_index(drop=True)
    s = paired_bootstrap(x, y, METRICS['adds_auc'], n_boot=1000)
    m = paired_bootstrap(x, y, METRICS['ar_mspd'], n_boot=1000)
    fx, fy = fail(x), fail(y)
    out.append(dict(split=label, group=name, n=len(x), clusters=cluster_ids(x).nunique(), fp_adds=round(METRICS['adds_auc'](x), 2), d_adds=round(s['diff'], 2), adds_lo=round(s['lo'], 2), adds_hi=round(s['hi'], 2),
                    d_mspd=round(m['diff'], 2), mspd_lo=round(m['lo'], 2), mspd_hi=round(m['hi'], 2), fp_fail_pct=round(100 * fx.mean(), 1), fixed=int((fx & ~fy).sum()), broken=int((~fx & fy).sum())))
  return out


def main():
  res, depth_rows = [], []
  for dataset in ('ycbv', 'lmo'):
    a, b = rows_for(dataset)
    info = per_target(dataset).set_index('key').loc[a['key']].reset_index()
    vis = pd.cut(a['visib_fract'], [0, .5, .7, .9, 1.0], labels=['under 50%', '50-70%', '70-90%', '90-100%'], include_lowest=True)
    iou = pd.cut(info['iou'], [0, .5, .8, 1.0], labels=['IoU under 0.5', 'IoU 0.5-0.8', 'IoU over 0.8'], include_lowest=True)
    # Set before looking: if agreement between sensor and ground truth is what matters, refinement
    # should help in the first bin and hurt in the last, inside one dataset and not only between two.
    gap = pd.cut(info['depth_offset'].abs(), [0, 1, 3, 6, 30], labels=['under 1 mm', '1-3 mm', '3-6 mm', 'over 6 mm'], include_lowest=True)
    for r in split(a, b, vis, 'visibility') + split(a, b, iou, 'mask') + split(a, b, gap, 'sensor-GT gap'):
      res.append(dict(dataset=dataset, **r))
    o = info['depth_offset'].dropna()
    depth_rows.append(dict(dataset=dataset, targets=len(o), mean_mm=round(o.mean(), 2), median_mm=round(o.median(), 2), p10_mm=round(o.quantile(.1), 2), p90_mm=round(o.quantile(.9), 2),
                           abs_median_mm=round(o.abs().median(), 2)))
    info.to_csv(TAB / f'transfer_targets_{dataset}.csv', index=False)
  res = pd.DataFrame(res)
  res.to_csv(TAB / 'transfer_breakdown.csv', index=False)
  pd.set_option('display.width', 200)
  pd.set_option('display.max_columns', 30)
  print(res.to_string(index=False))
  depth = pd.DataFrame(depth_rows)
  depth.to_csv(TAB / 'sensor_vs_gt_depth.csv', index=False)
  print('\nSensor depth minus ground-truth surface, median per target, mm (positive: the sensor reads farther away)')
  print(depth.to_string(index=False))


if __name__ == '__main__':
  main()
