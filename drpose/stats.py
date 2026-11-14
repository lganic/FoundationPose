"""Summary numbers with uncertainty.

Frames of one video are nearly the same picture, so treating them as independent would make the
confidence intervals far too narrow. Everything here resamples whole clusters of frames.
"""
import numpy as np

from .metrics import ar, auc


def cluster_ids(df):
  """YCB-V clusters by (scene, object). LM-O has one scene, so it uses (object, block of 100 frames)."""
  if df['dataset'].iloc[0] == 'lmo':
    return df['obj'].astype(str) + '_' + (df['im'] // 100).astype(str)
  return df['scene'].astype(str) + '_' + df['obj'].astype(str)


def summarize(df):
  """The headline metrics for one table of per-target errors, in percent."""
  a = ar(df)
  return dict(n=len(df), adds_auc=100 * auc(df['adds'] / 1e3), add_auc=100 * auc(df['add'] / 1e3),
              ar=100 * a['ar'], ar_vsd=100 * a['ar_vsd'], ar_mssd=100 * a['ar_mssd'], ar_mspd=100 * a['ar_mspd'],
              te_median=float(df['te'].median()), re_median=float(df['re'].median()), mspd_median=float(df['mspd'].median()))


def _resample_rows(ids, rng):
  groups = {}
  for i, c in enumerate(ids):
    groups.setdefault(c, []).append(i)
  keys = list(groups)
  picks = rng.integers(0, len(keys), len(keys))
  return np.concatenate([groups[keys[j]] for j in picks])


def paired_bootstrap(a, b, metric, n_boot=2000, seed=0):
  """The difference metric(b) - metric(a) on identical targets, with a 95% cluster-bootstrap interval.

  Pairing matters: a and b are resampled with the same rows, so shared difficulty cancels out.
  """
  a, b = a.sort_values('key').reset_index(drop=True), b.sort_values('key').reset_index(drop=True)
  assert (a['key'] == b['key']).all(), 'the comparison must be on the same targets'
  ids = cluster_ids(a).to_numpy()
  rng = np.random.default_rng(seed)
  diffs = []
  for _ in range(n_boot):
    rows = _resample_rows(ids, rng)
    diffs.append(metric(b.iloc[rows]) - metric(a.iloc[rows]))
  lo, hi = np.percentile(diffs, [2.5, 97.5])
  return dict(diff=float(metric(b) - metric(a)), lo=float(lo), hi=float(hi), n_clusters=len(set(ids)))


METRICS = {
  'adds_auc': lambda d: 100 * auc(d['adds'] / 1e3),
  'add_auc': lambda d: 100 * auc(d['add'] / 1e3),
  'ar_mssd': lambda d: 100 * ar(d)['ar_mssd'],
  'ar_mspd': lambda d: 100 * ar(d)['ar_mspd'],
  'ar_vsd': lambda d: 100 * ar(d)['ar_vsd'],
  'ar': lambda d: 100 * ar(d)['ar'],
}
