"""Every method must see identical corrupted frames, and each severity must do what its name says."""
import sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose import corruptions as c


def frame(seed=0):
  rng = np.random.default_rng(seed)
  rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
  depth = np.full((480, 640), 0.9)
  mask = np.zeros((480, 640), bool)
  mask[150:330, 220:420] = True
  depth[mask] = 0.7
  depth[:40] = 0     # pixels the sensor already lost
  return rgb, depth, mask


@pytest.mark.parametrize('cond', sorted(c.CONDITIONS))
def test_deterministic_and_leaves_inputs_alone(cond):
  rgb, depth, mask = frame()
  before = rgb.copy(), depth.copy(), mask.copy()
  a = c.apply(cond, rgb, depth, mask, 123)
  b = c.apply(cond, rgb, depth, mask, 123)
  for x, y in zip(a, b):
    assert np.array_equal(x, y)
  for x, y in zip(before, (rgb, depth, mask)):
    assert np.array_equal(x, y)
  assert a[0].dtype == np.uint8 and a[0].shape == rgb.shape and a[2].dtype == bool


# mask_box is left out: the box of this rectangular test mask is the mask itself. It has its own test below.
@pytest.mark.parametrize('cond', [k for k in c.CONDITIONS if k not in ('clean', 'mask_box')])
def test_each_condition_changes_only_what_it_claims(cond):
  rgb, depth, mask = frame()
  r, d, m = c.apply(cond, rgb, depth, mask, 7)
  kind = c.CONDITIONS[cond][0]
  changed = (not np.array_equal(r, rgb), not np.array_equal(d, depth), not np.array_equal(m, mask))
  expected = {'rgb_noise': (1, 0, 0), 'motion_blur': (1, 0, 0), 'depth_noise': (0, 1, 0), 'depth_holes': (0, 1, 0),
              'occlusion': (1, 1, 1), 'mask_erode': (0, 0, 1), 'mask_dilate': (0, 0, 1)}[kind]
  assert changed == tuple(bool(x) for x in expected)


@pytest.mark.parametrize('sigma', [10, 25, 50])
def test_rgb_noise_sigma(sigma):
  flat = np.full((480, 640, 3), 128, np.uint8)     # mid-gray, so clipping cannot shrink the noise
  out = c.apply(f'rgb_noise_{sigma}', flat, np.ones((480, 640)), np.ones((480, 640), bool), 1)[0]
  assert abs(out.astype(float).std() - sigma) < 0.05 * sigma


@pytest.mark.parametrize('sigma', [5, 10, 20])
def test_depth_noise_sigma_and_lost_pixels_stay_lost(sigma):
  rgb, depth, mask = frame()
  d = c.apply(f'depth_noise_{sigma}', rgb, depth, mask, 1)[1]
  assert np.all(d[:40] == 0)
  assert abs((d - depth)[40:].std() * 1e3 - sigma) < 0.05 * sigma


@pytest.mark.parametrize('p', [20, 40, 60])
def test_depth_holes_fraction(p):
  rgb, depth, mask = frame()
  d = c.apply(f'depth_holes_{p}', rgb, depth, mask, 1)[1]
  lost = ((depth > 0) & (d == 0)).sum() / (depth > 0).sum()
  assert abs(lost - p / 100) < 0.05


@pytest.mark.parametrize('p', [25, 50, 75])
@pytest.mark.parametrize('seed', range(8))
def test_occlusion_covers_the_requested_fraction_and_is_consistent(p, seed):
  rgb, depth, mask = frame()
  r, d, m = c.apply(f'occlusion_{p}', rgb, depth, mask, seed)
  assert not (m & ~mask).any()                                  # the visible mask can only shrink
  assert abs((1 - m.sum() / mask.sum()) - p / 100) < 0.02
  hidden = mask & ~m
  assert np.all(d[hidden] < 0.7) and np.all(d[hidden] >= 0.7 - 0.10 - 1e-9)   # the occluder is in front
  assert (r[hidden] != rgb[hidden]).any(axis=-1).mean() > 0.95                 # and it shows up in RGB
  assert np.array_equal(d[m], depth[m])                                        # the rest stays untouched


def test_blur_keeps_mean_brightness():
  rgb, depth, mask = frame()
  for k in (5, 11, 21):
    out = c.apply(f'motion_blur_{k}', rgb, depth, mask, 3)[0]
    assert abs(out.astype(float).mean() - rgb.astype(float).mean()) < 1.0
    assert out.astype(float).std() < rgb.astype(float).std()


def test_mask_box_is_the_tight_box():
  rgb, depth, mask = frame()
  m = c.apply('mask_box', rgb, depth, mask, 0)[2]
  assert np.array_equal(m, mask)     # the test mask is already a rectangle
  odd = np.zeros_like(mask)
  odd[100, 100] = odd[200, 300] = True
  m = c.apply('mask_box', rgb, depth, odd, 0)[2]
  assert m.sum() == 101 * 201 and m[100, 100] and m[200, 300]
