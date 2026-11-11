"""Checks the Windows shims against the real code they replace.

The references are optional, because neither is part of a normal install:
  P3D_REF    path to a checkout of facebookresearch/pytorch3d (its transforms are pure Python)
  MYCPP_REF  folder holding a compiled mycpp extension built from this repo's mycpp/ sources
"""
import importlib.util, os, sys
import numpy as np, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import compat_windows as cw


def load_p3d():
  root = os.environ.get('P3D_REF')
  if not root:
    pytest.skip('P3D_REF not set')
  sys.path.insert(0, root)   # pytorch3d.transforms is pure Python, so no compiled extension is needed
  import pytorch3d.transforms.so3 as so3
  import pytorch3d.transforms.rotation_conversions as rotation_conversions
  return {'so3': so3, 'rotation_conversions': rotation_conversions}


def load_mycpp():
  folder = os.environ.get('MYCPP_REF')
  if not folder:
    pytest.skip('MYCPP_REF not set')
  sys.path.insert(0, folder)
  spec = importlib.util.spec_from_file_location('mycpp', os.path.join(folder, 'mycpp.pyd'))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def random_rotations(n, rng):
  q = rng.normal(size=(n, 4))
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  w, x, y, z = q.T
  return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                   2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                   2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=1).reshape(n, 3, 3)


def make_poses(n, rng, spread=0.0):
  T = np.tile(np.eye(4), (n, 1, 1))
  T[:, :3, :3] = random_rotations(n, rng)
  T[:, :3, 3] = rng.normal(size=(n, 3)) * spread
  return T.astype(np.float32)


@pytest.mark.parametrize('scale', [1e-5, 1e-3, 1e-2, 0.1, 1.0, 3.0])
def test_so3_exp_map_matches_pytorch3d(scale):
  ref = load_p3d()['so3']
  v = torch.randn(512, 3, dtype=torch.float64) * scale
  assert torch.allclose(cw.so3_exp_map(v), ref.so3_exp_map(v), atol=1e-12)


def test_rotation_6d_matches_pytorch3d():
  ref = load_p3d()['rotation_conversions']
  d6 = torch.randn(512, 6, dtype=torch.float64)
  assert torch.allclose(cw.rotation_6d_to_matrix(d6), ref.rotation_6d_to_matrix(d6), atol=1e-12)


def test_so3_exp_map_is_a_rotation():
  R = cw.so3_exp_map(torch.randn(256, 3, dtype=torch.float64))
  assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(256, 3, 3), atol=1e-9)
  assert torch.allclose(torch.linalg.det(R), torch.ones(256, dtype=torch.float64), atol=1e-9)


@pytest.mark.parametrize('n_sym', [1, 2, 8, 72])
@pytest.mark.parametrize('angle,dist,spread', [(30, 99999, 0.0), (10, 99999, 0.0), (30, 0.05, 0.05), (60, 99999, 0.0)])
def test_cluster_poses_matches_compiled_mycpp(n_sym, angle, dist, spread):
  ref = load_mycpp()
  rng = np.random.default_rng(n_sym * 1000 + angle)
  poses = make_poses(300, rng, spread)
  syms = make_poses(n_sym, rng)
  syms[0] = np.eye(4, dtype=np.float32)
  want = np.asarray(ref.cluster_poses(angle, dist, list(poses), list(syms)))
  got = np.asarray(cw.mycpp.cluster_poses(angle, dist, poses, syms))
  assert want.shape == got.shape
  assert np.array_equal(want, got)
