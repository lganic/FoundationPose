"""Metric checks with answers known in advance, plus a physical check of the renderer."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import load_targets, scene_dir
from drpose.metrics import ERROR_COLUMNS, Evaluator, MSPD_TH, ar, auc


@pytest.fixture(scope='module')
def ycbv():
  return Evaluator('ycbv')


@pytest.fixture(scope='module')
def targets():
  return load_targets('ycbv')


def pick(targets, ev, symmetric):
  for t in targets:
    if (len(ev.syms(t['obj'])) > 1) == symmetric and t['visib_fract'] > 0.8:
      return t
  raise AssertionError


def shifted(pose, dx=0.0, dy=0.0, dz=0.0):
  p = pose.copy()
  p[:3, 3] += [dx, dy, dz]
  return p


def test_auc_known_values():
  assert auc([0.0] * 10) == pytest.approx(1.0)
  assert auc([1.0] * 10) == pytest.approx(0.0)
  assert auc([0.05] * 10) == pytest.approx(0.5, abs=0.02)   # correct for the top half of the threshold range
  assert auc(np.linspace(0, 0.1, 10001)) == pytest.approx(0.5, abs=0.01)


def test_auc_counts_failures_against_the_method():
  assert auc([0.0, np.inf]) == pytest.approx(0.5, abs=0.01)


def test_gt_pose_scores_perfectly(ycbv, targets):
  t = pick(targets, ycbv, symmetric=False)
  e = ycbv.errors(t, t['gt_pose'])
  for c in ('add', 'adds', 'mssd', 'mspd', 'te', 're'):
    assert e[c] == pytest.approx(0, abs=1e-6)
  assert all(e[f'vsd{i}'] == pytest.approx(0, abs=1e-6) for i in range(10))


def test_known_translation_error(ycbv, targets):
  t = pick(targets, ycbv, symmetric=False)
  e = ycbv.errors(t, shifted(t['gt_pose'], dx=0.01))
  assert e['add'] == pytest.approx(10, abs=1e-6)
  assert e['mssd'] == pytest.approx(10, abs=1e-6)
  assert e['te'] == pytest.approx(10, abs=1e-6)
  assert e['re'] == pytest.approx(0, abs=1e-4)
  assert 0 < e['adds'] <= 10
  # A 10 mm sideways shift projects to about f * 0.01 / z pixels.
  z = t['gt_pose'][2, 3]
  assert e['mspd'] == pytest.approx(ycbv.K(t)[0, 0] * 0.01 / z, rel=0.25)


def test_symmetry_is_forgiven_by_mssd_but_not_add(ycbv, targets):
  t = pick(targets, ycbv, symmetric=True)
  sym = ycbv.syms(t['obj'])[-1]
  T = np.eye(4)
  T[:3, :3], T[:3, 3] = sym['R'], np.asarray(sym['t']).reshape(3) / 1e3
  e = ycbv.errors(t, t['gt_pose'] @ T)
  assert e['mssd'] == pytest.approx(0, abs=1e-3)
  assert e['mspd'] == pytest.approx(0, abs=1e-3)
  assert e['re'] == pytest.approx(0, abs=1e-3)
  assert e['add'] > 5    # the symmetry-blind metric still sees a difference


def test_missing_pose_is_the_worst_score(ycbv, targets):
  e = ycbv.errors(targets[0], None)
  assert set(e) == set(ERROR_COLUMNS)
  assert np.isinf(e['add']) and np.isinf(e['mssd']) and e['vsd0'] == 1.0


def test_ar_bounds_and_threshold_semantics():
  zeros = {c: [0.0] for c in ERROR_COLUMNS}
  assert ar(pd.DataFrame({**zeros, 'diameter': [100.0]}))['ar'] == pytest.approx(1.0)
  bad = {c: [1e9] for c in ERROR_COLUMNS}
  bad.update({f'vsd{i}': [1.0] for i in range(10)})
  assert ar(pd.DataFrame({**bad, 'diameter': [100.0]}))['ar'] == pytest.approx(0.0)
  # An MSPD of exactly 25 px passes the 30..50 thresholds only: the comparison is strict.
  one = {**zeros, 'mspd': [25.0], 'diameter': [100.0]}
  assert ar(pd.DataFrame(one))['ar_mspd'] == pytest.approx((MSPD_TH > 25).mean())


def test_ar_is_undefined_rather_than_low_when_vsd_was_skipped():
  zeros = {c: [0.0] for c in ERROR_COLUMNS}
  zeros.update({f'vsd{i}': [np.nan] for i in range(10)})
  out = ar(pd.DataFrame({**zeros, 'diameter': [100.0]}))
  assert np.isnan(out['ar']) and np.isnan(out['ar_vsd'])
  assert out['ar_mssd'] == pytest.approx(1.0) and out['ar_mspd'] == pytest.approx(1.0)


def test_rendering_at_gt_reproduces_bops_own_masks(ycbv, targets):
  """If intrinsics, units, mesh, pose convention or the y-axis were wrong, this would fail badly."""
  rng = np.random.default_rng(0)
  ious = []
  for i in rng.choice(len(targets), 40, replace=False):
    t = targets[i]
    depth = ycbv.renderer().render_object(t['obj'], t['gt_pose'][:3, :3], t['gt_pose'][:3, 3] * 1e3, *[ycbv.K(t)[a] for a in ((0, 0), (1, 1), (0, 2), (1, 2))])['depth']
    bop = cv2.imread(f"{scene_dir('ycbv', t['scene'])}/mask/{t['im']:06d}_{t['gt_index']:06d}.png", -1) > 0
    ren = depth > 0
    ious.append((ren & bop).sum() / (ren | bop).sum())
  assert min(ious) > 0.95, f'worst IoU {min(ious):.3f}'
  assert np.median(ious) > 0.98, f'median IoU {np.median(ious):.3f}'
