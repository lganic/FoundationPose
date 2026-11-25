"""One picture per refiner step, from a real YCB-V frame, for the method slide.

Uses the frozen settings and the same depth cleanup as run_refine.py. FoundationPose's own pose is
already within a few millimeters, too close to see move, so the start is FoundationPose's pose with
a deliberate offset. The pictures are an illustration of the loop, not a result.

  python scripts/make_method_figures.py
"""
import json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from drpose.data import REPO, SceneCache
from drpose.refine import Observation, RefineConfig, Scene, objective, so3_exp

KEY = 'ycbv/55/1603/14'                 # the mug on the human-check slide
OFFSET_MM = np.array([7.0, -5.0, 6.0])   # camera frame: right, up (negative v), away
OFFSET_DEG = 4.0
OUT = 480                                # px per picture
FIGS = REPO / 'results/figures'
BLUE, ORANGE, GRAY = (214, 120, 42), (52, 104, 235), (110, 110, 110)   # BGR


def rotation(axis, deg):
  axis = np.asarray(axis, float) / np.linalg.norm(axis)
  return cv2.Rodrigues(axis * np.radians(deg))[0]


def gray_photo(obs):
  g = (obs.rgb.cpu().numpy() @ [0.299, 0.587, 0.114])
  return np.repeat((0.35 + 0.5 * g)[..., None], 3, -1)


def up(img):
  return cv2.resize(img, (OUT, OUT), interpolation=cv2.INTER_NEAREST if img.dtype == bool else cv2.INTER_LINEAR)


def outline(img, hard, color, thickness=3):
  m = up(hard.astype(np.uint8) * 255)
  cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cv2.drawContours(img, cs, -1, color, thickness, cv2.LINE_AA)


def diverging(x, lim):
  """Blue for negative, orange for positive, through a neutral gray at zero. x in [-lim, lim]."""
  a = np.clip(x / lim, -1, 1)[..., None]
  mid, neg, pos = np.array([0.92, 0.92, 0.9]), np.array([0.84, 0.47, 0.16]), np.array([0.2, 0.41, 0.92])   # BGR
  return np.where(a < 0, mid + (neg - mid) * -a, mid + (pos - mid) * a)


def main():
  from Utils import bilateral_filter_depth, erode_depth, make_mesh_tensors
  import nvdiffrast.torch as dr
  from drpose.metrics import Evaluator

  ds, scene, im, obj = KEY.split('/')
  t = dict(dataset=ds, scene=int(scene), im=int(im), obj=int(obj))   # all SceneCache.load needs
  cache = SceneCache()
  rgb, depth, mask, K = cache.load(t, 'cnos')
  depth = bilateral_filter_depth(erode_depth(depth, radius=2, device='cuda'), radius=2, device='cuda')
  scene = Scene(make_mesh_tensors(cache.mesh('ycbv', t['scene'], t['obj'])), dr.RasterizeCudaContext(), Evaluator('ycbv', use_vsd=False).diameter(t['obj']) / 1e3)
  with open(REPO / 'results/raw/fp_cnos/ycbv/clean.jsonl') as f:
    fp = np.array(next(json.loads(l) for l in f if KEY in l)['pose'])

  c = scene.center.cpu().numpy()
  start = fp.copy()
  R_off = rotation([0.3, 1.0, 0.2], OFFSET_DEG)
  start[:3, :3] = R_off @ fp[:3, :3]
  start[:3, 3] = fp[:3, :3] @ c + fp[:3, 3] + OFFSET_MM / 1e3 - start[:3, :3] @ c

  cfg = RefineConfig(w_sil=0, w_rgb=0, sigma_depth=0.003, steps=30)
  obs = Observation(rgb, depth, mask, K, start, scene.radius, cfg)
  vis = Observation(rgb, depth, mask, K, start, scene.radius, replace(cfg, crop_size=384))
  dev = obs.depth.device

  # The refine() loop, kept step by step so the path can be drawn.
  R0 = torch.as_tensor(start[None, :3, :3], device=dev, dtype=torch.float32)
  t0 = torch.as_tensor(start[None, :3, 3], device=dev, dtype=torch.float32)
  cc = scene.center
  center0 = R0 @ cc + t0
  w = torch.zeros(1, 3, device=dev, requires_grad=True)
  d = torch.zeros(1, 3, device=dev, requires_grad=True)
  opt = torch.optim.Adam([{'params': [w], 'lr': cfg.lr_rot}, {'params': [d], 'lr': cfg.lr_trans}])
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=0.05 * cfg.lr_trans)
  compose = lambda w, d: (so3_exp(w) @ R0, None)
  def pose_of(w, d):
    R = so3_exp(w) @ R0
    return R, center0 + d - R @ cc
  path, costs, first_grad = [], [], None
  best = (float('inf'), None, None)
  for step in range(cfg.steps + 1):
    opt.zero_grad(set_to_none=True)
    R, tt = pose_of(w, d)
    cov, z, col, hard = scene.render(R, tt, obs.Kc, cfg.crop_size, False)
    total, _ = objective(obs, cov, z, col, hard, cfg)
    costs.append(float(total))
    path.append((R.detach().clone(), tt.detach().clone()))
    if float(total) < best[0]:
      best = (float(total), R.detach().clone(), tt.detach().clone())
    if step == cfg.steps:
      break
    total.sum().backward()
    if step == 0:
      first_grad = d.grad.detach().clone()[0]
    opt.step()
    sched.step()
  print(f'cost {costs[0]:.3f} -> {best[0]:.3f} over {cfg.steps} steps')

  def draw(R, tt, o=vis):
    with torch.no_grad():
      return [x.cpu().numpy()[0] if x is not None else None for x in scene.render(R, tt, o.Kc, 384, False)]

  _, z0, _, h0 = draw(*path[0])
  zo, m, valid = vis.depth.cpu().numpy(), vis.mask.cpu().numpy(), vis.valid.cpu().numpy()
  photo = gray_photo(vis)

  # 1. Render: the depth the renderer draws at the current guess.
  img = np.full((384, 384, 3), 0.96)
  zz = np.where(h0, z0, np.nan)
  lo, hi = np.nanpercentile(zz, 2), np.nanpercentile(zz, 98)
  shade = np.clip((zz - lo) / max(hi - lo, 1e-6), 0, 1)
  near, far = np.array([0.55, 0.30, 0.08]), np.array([0.97, 0.86, 0.72])   # dark blue near, pale far (BGR)
  img[h0] = (near + (far - near) * shade[..., None])[h0]
  render_img = (up(img) * 255).astype(np.uint8)
  cv2.imwrite(str(FIGS / 'method_1_render.png'), render_img)

  # 2. Compare: rendered depth minus sensor depth, inside the detector's mask.
  both = h0 & valid & m
  res = (z0 - zo) * 1e3
  img = photo.copy()
  img[both] = diverging(res, 15)[both]
  out = (up(img) * 255).astype(np.uint8)
  outline(out, m, GRAY, 2)
  cv2.imwrite(str(FIGS / 'method_2_compare.png'), out)

  # 3. Differentiate: the cost as the object slides left-right and up-down from the start, with the
  # gradient's sideways part as an arrow. It points downhill on this map, which is all a gradient says.
  span, n = 20.0, 51                      # mm each way, grid points per side
  s = np.linspace(-span, span, n) / 1e3
  shifts = torch.tensor([[x, y, 0.0] for y in s for x in s], device=dev, dtype=torch.float32)
  with torch.no_grad():
    Rg = R0.expand(len(shifts), 3, 3)
    tg = (center0 + shifts - Rg @ cc)
    cov, z, col, hard = scene.render(Rg, tg, obs.Kc, cfg.crop_size, False)
    cost = objective(obs, cov, z, col, hard, cfg)[0].cpu().numpy().reshape(n, n)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.colors import LinearSegmentedColormap
  cmap = LinearSegmentedColormap.from_list('valley', ['#1d4f91', '#5b93d8', '#c9dcf3', '#f4f7fb'])
  fig = plt.figure(figsize=(OUT / 200, OUT / 200), dpi=200)
  ax = fig.add_axes([0, 0, 1, 1])
  ext = [-span, span, span, -span]        # x right, y down, as in the image
  ax.imshow(cost, cmap=cmap, extent=ext, interpolation='bilinear')
  ax.contour(s * 1e3, s * 1e3, cost, levels=10, colors='white', linewidths=0.6, alpha=0.8)
  g = -first_grad.cpu().numpy()[:2]
  g = g / np.linalg.norm(g) * span * 0.55
  ax.annotate('', xy=(g[0], g[1]), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>,head_width=0.5,head_length=0.9', color='#14202b', lw=2.6))
  ax.plot([0], [0], 'o', ms=9, color='#eb6834', mec='white', mew=1.5)
  i = np.unravel_index(cost.argmin(), cost.shape)
  ax.set_xlim(-span, span)
  ax.set_ylim(span, -span)
  ax.axis('off')
  fig.savefig(FIGS / 'method_3_gradient.png', dpi=200)
  plt.close(fig)
  print(f'cost map: lowest point at x {s[i[1]] * 1e3:+.1f} mm, y {s[i[0]] * 1e3:+.1f} mm from the start; downhill arrow ({g[0]:+.2f}, {g[1]:+.2f})')
  Kc = vis.Kc
  p = lambda X: np.array([Kc[0, 0] * X[0] / X[2] + Kc[0, 2], Kc[1, 1] * X[1] / X[2] + Kc[1, 2]]) * OUT / 384

  # 4. Step: the outline at the start and after 30 steps, with the center's path between them.
  out = (up(photo) * 255).astype(np.uint8)
  outline(out, h0, ORANGE, 3)
  _, _, _, h1 = draw(best[1], best[2])
  outline(out, h1, BLUE, 3)
  pts = [p((R[0] @ cc + tt[0]).cpu().numpy()) for R, tt in path]
  for q0, q1 in zip(pts, pts[1:]):
    cv2.line(out, tuple(int(v) for v in q0), tuple(int(v) for v in q1), (40, 40, 40), 2, cv2.LINE_AA)
  cv2.circle(out, tuple(int(v) for v in pts[0]), 7, ORANGE, -1, cv2.LINE_AA)
  cv2.circle(out, tuple(int(v) for v in p((best[1][0] @ cc + best[2][0]).cpu().numpy())), 7, BLUE, -1, cv2.LINE_AA)
  cv2.imwrite(str(FIGS / 'method_4_step.png'), out)

  # How far the final pose is from FoundationPose's, to report honestly.
  fc = fp[:3, :3] @ c + fp[:3, 3]
  bc = (best[1][0] @ cc + best[2][0]).cpu().numpy()
  print(f'start offset {np.linalg.norm(OFFSET_MM):.1f} mm; final center is {np.linalg.norm(bc - fc) * 1e3:.1f} mm from FoundationPose')


if __name__ == '__main__':
  main()
