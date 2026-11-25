"""Tests the refiner on synthetic scenes, where the true pose is known exactly.

Real ground truth is off by millimeters, so it cannot show whether the gradient and the optimizer
are correct. Here the "photo" is rendered with this same pipeline, so a correct refiner must
recover the pose almost perfectly, and any bug in the gradient shows up as a failure to converge.
"""
import sys
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import SceneCache, load_targets
from drpose.metrics import rotation_angle_deg
from drpose.refine import RefineConfig, Scene, crop_window, refine, sample_crop, so3_exp

K = np.array([[1066.8, 0, 312.99], [0, 1067.5, 241.31], [0, 0, 1.0]])
H, W = 480, 640


@pytest.fixture(scope='module')
def scene():
  import nvdiffrast.torch as dr
  from Utils import make_mesh_tensors
  cache = SceneCache()
  mesh = cache.mesh('ycbv', 48, 15)     # the power drill: textured and not symmetric
  pts = np.asarray(mesh.vertices)
  diam = np.linalg.norm(pts.max(0) - pts.min(0))
  return Scene(make_mesh_tensors(mesh), dr.RasterizeCudaContext(), diam)


def true_pose():
  rng = np.random.default_rng(1)
  w = torch.as_tensor(rng.normal(size=(1, 3)) * 0.8, dtype=torch.float32)
  P = np.eye(4)
  P[:3, :3] = so3_exp(w)[0].numpy()
  P[:3, 3] = [0.03, -0.02, 0.75]
  return P


def perturb(P, deg, mm, seed=0):
  rng = np.random.default_rng(seed)
  axis = rng.normal(size=3)
  axis /= np.linalg.norm(axis)
  w = torch.as_tensor((axis * np.radians(deg))[None], dtype=torch.float32)
  Q = P.copy()
  Q[:3, :3] = so3_exp(w)[0].numpy() @ P[:3, :3]
  d = rng.normal(size=3)
  Q[:3, 3] += d / np.linalg.norm(d) * mm / 1e3
  return Q


def synth(scene, P, background_depth=1.2):
  """Renders the full frame at P, as if the camera had taken it."""
  R = torch.as_tensor(P[None, :3, :3], device='cuda', dtype=torch.float32)
  t = torch.as_tensor(P[None, :3, 3], device='cuda', dtype=torch.float32)
  # The render is square, so draw into a 640x640 canvas and keep the top 480 rows.
  with torch.no_grad():
    cov, z, col, hard = scene.render(R, t, K, 640)
  hard = hard[0, :H].cpu().numpy()
  depth = np.where(hard, z[0, :H].cpu().numpy(), background_depth)
  rgb = (col[0, :H].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
  return rgb, depth, hard


def errors(P, Q):
  return rotation_angle_deg(P[:3, :3] @ Q[:3, :3].T), 1e3 * np.linalg.norm(P[:3, 3] - Q[:3, 3])


def test_so3_exp_matches_matrix_exponential_and_has_finite_gradient_at_zero():
  w = torch.randn(64, 3, dtype=torch.float64)
  x, y, z = w.unbind(-1)
  o = torch.zeros_like(x)
  Wm = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(-1, 3, 3)
  assert torch.allclose(so3_exp(w), torch.linalg.matrix_exp(Wm), atol=1e-10)
  w0 = torch.zeros(1, 3, requires_grad=True)
  so3_exp(w0)[0, 0, 1].backward()
  assert torch.isfinite(w0.grad).all() and w0.grad.abs().sum() > 0.5


def test_crop_intrinsics_project_points_to_the_right_place():
  P = true_pose()
  size = 192
  u0, v0, scale, Kc = crop_window(P, K, 0.1, H, W, size, 1.25)
  X = np.array([0.05, 0.02, 0.8])
  u, v = K[0, 0] * X[0] / X[2] + K[0, 2], K[1, 1] * X[1] / X[2] + K[1, 2]
  uc, vc = Kc[0, 0] * X[0] / X[2] + Kc[0, 2], Kc[1, 1] * X[1] / X[2] + Kc[1, 2]
  assert u0 + (uc + 0.5) / scale - 0.5 == pytest.approx(u)
  assert v0 + (vc + 0.5) / scale - 0.5 == pytest.approx(v)


def test_sample_crop_is_exact_at_scale_one():
  img = torch.arange(H * W, dtype=torch.float32, device='cuda').reshape(H, W, 1)
  out = sample_crop(img, 100, 50, 1.0, 64, 'nearest')
  assert torch.equal(out, img[50:114, 100:164])


def test_cropped_render_matches_full_render(scene):
  """The crop's intrinsics must place the object exactly where the full-frame render has it."""
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  cfg = RefineConfig()
  u0, v0, scale, Kc = crop_window(P, K, scene.radius, H, W, cfg.crop_size, cfg.crop_margin)
  R = torch.as_tensor(P[None, :3, :3], device='cuda', dtype=torch.float32)
  t = torch.as_tensor(P[None, :3, 3], device='cuda', dtype=torch.float32)
  with torch.no_grad():
    _, _, _, hard = scene.render(R, t, Kc, cfg.crop_size)
  want = sample_crop(torch.as_tensor(mask, device='cuda', dtype=torch.float32)[..., None], u0, v0, scale, cfg.crop_size, 'nearest')[..., 0] > 0.5
  iou = (hard[0] & want).sum() / (hard[0] | want).sum()
  assert float(iou) > 0.97


def test_renderer_agrees_with_real_bop_masks():
  """Independent of this renderer: BOP's masks were made by different software from the GT pose.

  The synthetic tests cannot catch a convention bug such as a flipped axis, because the fake photo
  comes from this same renderer and the bug cancels out. This one uses real data, so it can.
  """
  import cv2
  import nvdiffrast.torch as dr
  from Utils import make_mesh_tensors
  from drpose.data import scene_dir
  cache = SceneCache()
  glctx = dr.RasterizeCudaContext()
  cfg = RefineConfig()
  targets = load_targets('ycbv')
  rng = np.random.default_rng(1)
  ious, shifted_ious = [], []
  for i in rng.choice(len(targets), 25, replace=False):
    t = targets[i]
    mesh = cache.mesh('ycbv', t['scene'], t['obj'])
    pts = np.asarray(mesh.vertices)
    sc = Scene(make_mesh_tensors(mesh), glctx, np.linalg.norm(pts.max(0) - pts.min(0)))
    Kt = cache.reader('ycbv', t['scene']).get_K(0).copy()
    P = t['gt_pose']
    u0, v0, scale, Kc = crop_window(P, Kt, sc.radius, H, W, cfg.crop_size, cfg.crop_margin)
    R = torch.as_tensor(P[None, :3, :3], device='cuda', dtype=torch.float32)
    tt = torch.as_tensor(P[None, :3, 3], device='cuda', dtype=torch.float32)
    with torch.no_grad():
      _, _, _, hard = sc.render(R, tt, Kc, cfg.crop_size, need_rgb=False)
    bop = cv2.imread(f"{scene_dir('ycbv', t['scene'])}/mask/{t['im']:06d}_{t['gt_index']:06d}.png", -1) > 0
    want = sample_crop(torch.as_tensor(bop, device='cuda', dtype=torch.float32)[..., None], u0, v0, scale, cfg.crop_size, 'nearest')[..., 0] > 0.5
    ious.append(float((hard[0] & want).sum() / (hard[0] | want).sum().clamp_min(1)))
    flipped = torch.flip(hard[0], dims=[0])
    shifted_ious.append(float((flipped & want).sum() / (flipped | want).sum().clamp_min(1)))
  assert np.median(ious) > 0.95, f'median IoU {np.median(ious):.3f}'
  assert np.median(ious) > np.median(shifted_ious) + 0.05     # the check can tell a flip apart


def test_gradient_matches_finite_differences(scene):
  """Compares d(cost)/d(pose) from nvdiffrast with a numeric derivative of the same cost."""
  from drpose.refine import Observation, objective
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  Q = perturb(P, 2, 4)
  cfg = RefineConfig(w_rgb=0.0, w_sil=1.0, w_depth=0.0)     # silhouette: the term that depends on antialias
  obs = Observation(rgb, depth, mask, K, Q, scene.radius, cfg)
  R0 = torch.as_tensor(Q[None, :3, :3], device='cuda', dtype=torch.float32)
  t0 = torch.as_tensor(Q[None, :3, 3], device='cuda', dtype=torch.float32)

  def cost(d):
    cov, z, col, hard = scene.render(R0, t0 + d, obs.Kc, cfg.crop_size, need_rgb=False)
    return objective(obs, cov, z, col, hard, cfg)[0].sum()

  d = torch.zeros(1, 3, device='cuda', requires_grad=True)
  cost(d).backward()
  numeric = []
  eps = 1e-3
  with torch.no_grad():
    for i in range(3):
      e = torch.zeros(1, 3, device='cuda')
      e[0, i] = eps
      numeric.append(float((cost(e) - cost(-e)) / (2 * eps)))
  analytic = d.grad[0].cpu().numpy()
  numeric = np.array(numeric)
  cos = analytic @ numeric / (np.linalg.norm(analytic) * np.linalg.norm(numeric))
  assert cos > 0.9, f'analytic {analytic} numeric {numeric}'


@pytest.mark.parametrize('deg,mm', [(1, 2), (3, 5), (6, 10)])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_recovers_the_true_pose(scene, deg, mm, seed):
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  Q = perturb(P, deg, mm, seed)
  out, final, initial = refine(scene, rgb, depth, mask, K, Q[None], RefineConfig(steps=120))
  re, te = errors(P, out[0])
  r0, t0 = errors(P, Q)
  assert final[0] < initial[0]
  assert re < 0.25 * r0 + 0.1 and te < 0.25 * t0 + 0.3, f'start {r0:.2f} deg {t0:.2f} mm, end {re:.2f} deg {te:.2f} mm'


def test_stays_put_when_already_correct(scene):
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  out, _, _ = refine(scene, rgb, depth, mask, K, P[None], RefineConfig())
  re, te = errors(P, out[0])
  assert re < 0.2 and te < 0.5


@pytest.mark.parametrize('camera_offset_px', [0.0, 0.5])
def test_no_subpixel_pull_against_analytic_depth(camera_offset_px):
  """Refined poses moved about 0.4 px up on two datasets, which is what a pixel-center slip would do.

  Every other synthetic test here draws its photo with the refiner's own renderer, so such a slip
  cancels, and IoU against BOP's masks is too blunt to see half a pixel. This depth comes from
  intersecting rays with a sphere at OpenCV pixel centers, sharing no code with the renderer. The
  0.5 case moves the camera on purpose, to show the test can see the size of slip it is looking for.

  It averages over positions because one position can't settle it: how the outline falls on the
  pixel grid pulls a single run by up to 0.18 px, in either direction.
  """
  import nvdiffrast.torch as dr
  import trimesh
  from Utils import make_mesh_tensors
  radius = 0.05
  mesh = trimesh.creation.icosphere(subdivisions=5, radius=radius)
  scene = Scene(make_mesh_tensors(mesh), dr.RasterizeCudaContext(), 2 * radius)

  rng = np.random.default_rng(0)
  v, u = np.mgrid[0:H, 0:W].astype(np.float64)
  d = np.stack([(u - K[0, 2] - camera_offset_px) / K[0, 0], (v - K[1, 2] - camera_offset_px) / K[1, 1], np.ones_like(u)], -1)
  rgb = np.full((H, W, 3), 128, np.uint8)
  moved = []
  for _ in range(10):
    C = np.array([0.03, -0.02, 0.75]) + rng.uniform(-1, 1, 3) * (0.02, 0.02, 0.05)
    a, b, c = (d * d).sum(-1), (d @ C), C @ C - radius ** 2
    disc = b * b - a * c
    hit = disc > 0
    depth = np.where(hit, (b - np.sqrt(np.where(hit, disc, 0))) / a, 1.2)
    P = np.eye(4)
    P[:3, 3] = C
    out, _, _ = refine(scene, rgb, depth, hit, K, P[None], RefineConfig(w_sil=0, w_rgb=0, sigma_depth=0.003, steps=120))
    x = out[0][:3, 3]
    moved.append((K[0, 0] * (x[0] / x[2] - C[0] / C[2]), K[1, 1] * (x[1] / x[2] - C[1] / C[2])))
  du, dv = np.mean(moved, 0)
  assert du == pytest.approx(camera_offset_px, abs=0.1) and dv == pytest.approx(camera_offset_px, abs=0.1), f'du {du:+.3f} px, dv {dv:+.3f} px'


def test_occluder_does_not_pull_the_pose(scene):
  """A surface in front of half the object must not be mistaken for the object."""
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  vs, us = np.nonzero(mask)
  cut = np.median(us)
  occ = np.zeros_like(mask)
  occ[:, :int(cut)] = True
  depth = depth.copy()
  depth[occ] = P[2, 3] - 0.15
  rgb = rgb.copy()
  rgb[occ] = (90, 120, 60)
  visible = mask & ~occ
  Q = perturb(P, 2, 4, 3)
  out, _, _ = refine(scene, rgb, depth, visible, K, Q[None], RefineConfig(steps=120))
  re, te = errors(P, out[0])
  r0, t0 = errors(P, Q)
  assert te < t0 and re < r0, f'start {r0:.2f} deg {t0:.2f} mm, end {re:.2f} deg {te:.2f} mm'


def test_occlusion_gate_beats_having_no_gate(scene):
  """Without the gate, the silhouette term treats the occluder as a misfit and biases the pose.

  Asserting only that the error shrank was too weak: a mutation run showed that deleting the gate
  still passed that test, because a biased result still improves on the start.

  With these terms on, one seed's error varies by up to 0.4 mm between identical runs, so comparing
  one run with another is a coin flip on some seeds. This compares means across seeds, with a margin
  that a deleted gate cannot clear. The frozen depth-only settings don't need the gate: a reading
  past it is already past the cost cap, so gated and ungated runs match.
  """
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  vs, us = np.nonzero(mask)
  occ = np.zeros_like(mask)
  occ[:, :int(np.median(us))] = True
  depth, rgb = depth.copy(), rgb.copy()
  depth[occ] = P[2, 3] - 0.15
  rgb[occ] = (90, 120, 60)
  visible = mask & ~occ
  gated, ungated = [], []
  for seed in range(3, 9):
    Q = perturb(P, 2, 4, seed)
    a, _, _ = refine(scene, rgb, depth, visible, K, Q[None], RefineConfig(steps=120))
    b, _, _ = refine(scene, rgb, depth, visible, K, Q[None], RefineConfig(steps=120, occ_gate=1e9))
    gated.append(errors(P, a[0])[1])
    ungated.append(errors(P, b[0])[1])
  assert np.mean(gated) < 0.75 * np.mean(ungated), f'gated {np.mean(gated):.3f} mm, ungated {np.mean(ungated):.3f} mm'


@pytest.mark.parametrize('term', ['w_depth', 'w_sil', 'w_rgb'])
def test_every_term_charges_for_leaving_the_mask(scene, term):
  """Moving the object out of view must never be cheaper than sitting on it.

  The color term once scored zero with nothing left to compare, which made leaving the object
  the best possible move, and an ablation wrongly concluded that color information hurts.
  """
  from drpose.refine import Observation, objective
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  cfg = RefineConfig(w_depth=0.0, w_sil=0.0, w_rgb=0.0)
  setattr(cfg, term, 1.0)
  obs = Observation(rgb, depth, mask, K, P, scene.radius, cfg)
  costs = []
  for shift in (0.0, 0.5):
    R = torch.as_tensor(P[None, :3, :3], device='cuda', dtype=torch.float32)
    t = torch.as_tensor(P[None, :3, 3], device='cuda', dtype=torch.float32) + torch.tensor([shift, 0, 0], device='cuda')
    with torch.no_grad():
      cov, z, col, hard = scene.render(R, t, obs.Kc, cfg.crop_size)
      costs.append(float(objective(obs, cov, z, col, hard, cfg)[0][0]))
  assert costs[1] > costs[0] + 0.5


def test_uniform_color_gives_a_finite_pose(scene):
  """Zero variance used to send a NaN gradient through sqrt, and the NaN pose was then accepted."""
  P = true_pose()
  _, depth, mask = synth(scene, P)
  flat = np.full((H, W, 3), 120, np.uint8)
  Q = perturb(P, 2, 4)
  out, final, _ = refine(scene, flat, depth, mask, K, Q[None], RefineConfig(w_depth=0.0, w_sil=0.0, w_rgb=1.0, steps=40))
  assert np.isfinite(out).all() and np.isfinite(final).all()


def test_rejects_a_change_that_does_not_improve_the_objective(scene):
  """With no steps there is nothing to accept, so the input pose must come back unchanged."""
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  Q = perturb(P, 2, 4)
  out, final, initial = refine(scene, rgb, depth, mask, K, Q[None], RefineConfig(steps=0))
  assert np.array_equal(out[0], Q)
  assert final[0] == initial[0]


def test_hypotheses_are_refined_independently(scene):
  P = true_pose()
  rgb, depth, mask = synth(scene, P)
  Qs = np.stack([perturb(P, 3, 5, s) for s in range(4)])
  batch, _, _ = refine(scene, rgb, depth, mask, K, Qs, RefineConfig(steps=30))
  single, _, _ = refine(scene, rgb, depth, mask, K, Qs[1:2], RefineConfig(steps=30))
  # The crop is anchored on the first hypothesis, so the two runs differ a little but must agree closely.
  re, te = errors(batch[1], single[0])
  assert re < 0.5 and te < 1.0
