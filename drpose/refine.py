"""Differentiable render-and-compare pose refinement, built on nvdiffrast.

Given an initial pose, this renders the object, compares the render with the observed RGB, depth
and detector mask, and follows the gradient of that comparison through the rasterizer back to the
six pose parameters.

What makes the gradient useful:
  * dr.antialias gives silhouette edges a gradient with respect to vertex positions. Without it,
    a rasterized image is piecewise constant and has no pose gradient at the object's outline.
  * Pixels where the sensor sees something clearly in front of the render are excluded, so an
    occluder is explained as an occluder and does not pull the pose.
  * The depth cost is robust and saturates, so sensor outliers and bad mask pixels cannot dominate.
  * The objective is scored once more at the end, and the result is only accepted if it improved.
    That check uses no ground truth.

Nothing in here sees a ground-truth pose or a BOP mask: the caller passes a detector mask.
"""
from dataclasses import dataclass

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F


@dataclass
class RefineConfig:
  steps: int = 60
  lr_rot: float = 4e-3        # radians per Adam step
  lr_trans: float = 1.5e-3    # meters per Adam step
  crop_size: int = 192        # the crop is rendered at this resolution
  crop_margin: float = 1.25   # crop half-width, as a multiple of the object radius
  w_depth: float = 1.0
  w_sil: float = 1.0
  w_rgb: float = 0.5
  sigma_depth: float = 0.005  # meters; depth residuals around this size are treated as sensor noise
  occ_gate: float = 0.02      # meters; a sensor reading this far in front of the render is an occluder
  cost_cap: float = 3.0       # the most any single depth pixel can cost
  rgb_cap: float = 2.0        # what a mask pixel the render fails to cover costs in the color term
  w_prior: float = 0.0        # optional pull back toward the initial pose
  min_gain: float = 0.0       # the relative objective improvement needed to accept the result


def so3_exp(w):
  """Rodrigues, safe to differentiate at zero. w: (B,3) -> (B,3,3)."""
  theta2 = (w * w).sum(-1, keepdim=True)
  small = theta2 < 1e-10
  theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)
  theta = theta2_safe.sqrt()
  a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
  b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2_safe)
  x, y, z = w.unbind(-1)
  o = torch.zeros_like(x)
  W = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(-1, 3, 3)
  eye = torch.eye(3, device=w.device, dtype=w.dtype)
  return eye + a[..., None] * W + b[..., None] * (W @ W)


def crop_window(pose, K, radius, H, W, size, margin):
  """A square window around the object's bounding sphere, and the intrinsics of that crop.

  Crop pixel i looks at image coordinate u0 + (i + 0.5) / scale - 0.5, which keeps pixel centers
  exact under scaling. Returns (u0, v0, scale, K_crop).
  """
  c = pose[:3, 3]
  u, v = K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]
  half = max(K[0, 0], K[1, 1]) * radius * margin / max(c[2], 1e-3)
  scale = size / (2 * half)
  u0, v0 = u - half, v - half
  Kc = K.copy()
  Kc[0, 0], Kc[1, 1] = K[0, 0] * scale, K[1, 1] * scale
  Kc[0, 2], Kc[1, 2] = (K[0, 2] - u0 + 0.5) * scale - 0.5, (K[1, 2] - v0 + 0.5) * scale - 0.5
  return u0, v0, scale, Kc


def sample_crop(img, u0, v0, scale, size, mode):
  """Samples (H,W,C) tensor `img` on the crop grid. Points outside the image read as zero."""
  H, W = img.shape[:2]
  i = torch.arange(size, device=img.device, dtype=torch.float32)
  us = u0 + (i + 0.5) / scale - 0.5
  vs = v0 + (i + 0.5) / scale - 0.5
  gx = (us + 0.5) / W * 2 - 1
  gy = (vs + 0.5) / H * 2 - 1
  grid = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], -1)[None]
  x = img.permute(2, 0, 1)[None].float()
  return F.grid_sample(x, grid, mode=mode, padding_mode='zeros', align_corners=False)[0].permute(1, 2, 0)


class Scene:
  """One object's mesh tensors plus the rasterizer context."""

  def __init__(self, mesh_tensors, glctx, diameter_m):
    self.mt = mesh_tensors
    self.glctx = glctx
    self.pos = mesh_tensors['pos'].float().contiguous()
    self.tri = mesh_tensors['faces'].int().contiguous()
    self.radius = 0.5 * diameter_m
    self.center = 0.5 * (self.pos.max(0).values + self.pos.min(0).values)

  def render(self, R, t, Kc, size, need_rgb=True):
    """R (B,3,3), t (B,3). Returns the soft coverage (B,S,S), depth (B,S,S), color (B,S,S,3) and hard coverage."""
    B = R.shape[0]
    cam = self.pos[None] @ R.transpose(1, 2) + t[:, None]
    X, Y, Z = cam.unbind(-1)
    near, far = 0.01, 10.0
    # Pixel centers are at integers in OpenCV but at half-integers in NDC, hence the + 0.5.
    cx = 2 * (Kc[0, 0] * X + (Kc[0, 2] + 0.5) * Z) / size - Z
    cy = 2 * (Kc[1, 1] * Y + (Kc[1, 2] + 0.5) * Z) / size - Z
    cz = (far + near) / (far - near) * Z - 2 * far * near / (far - near)
    clip = torch.stack([cx, cy, cz, Z], -1).contiguous()
    rast, _ = dr.rasterize(self.glctx, clip, self.tri, resolution=[size, size])
    hard = (rast[..., 3] > 0)
    z, _ = dr.interpolate(Z[..., None].contiguous(), rast, self.tri)
    parts = [hard[..., None].float()]
    if need_rgb:
      if 'tex' in self.mt:
        uv, _ = dr.interpolate(self.mt['uv'][None].contiguous(), rast, self.mt['uv_idx'])
        col = dr.texture(self.mt['tex'].contiguous(), uv, filter_mode='linear')
      else:
        col, _ = dr.interpolate(self.mt['vertex_color'][None].contiguous(), rast, self.tri)
      parts.append(col * hard[..., None].float())
    aa = dr.antialias(torch.cat(parts, -1).contiguous(), rast, clip, self.tri)
    return aa[..., 0], z[..., 0], (aa[..., 1:4] if need_rgb else None), hard


class Observation:
  """The cropped image evidence for one target."""

  def __init__(self, rgb, depth, mask, K, pose0, radius, cfg, device='cuda'):
    H, W = depth.shape
    u0, v0, scale, self.Kc = crop_window(pose0, K, radius, H, W, cfg.crop_size, cfg.crop_margin)
    s = cfg.crop_size
    self.rgb = sample_crop(torch.as_tensor(rgb, device=device, dtype=torch.float32) / 255, u0, v0, scale, s, 'bilinear')
    self.depth = sample_crop(torch.as_tensor(depth, device=device, dtype=torch.float32)[..., None], u0, v0, scale, s, 'nearest')[..., 0]
    self.mask = sample_crop(torch.as_tensor(mask, device=device, dtype=torch.float32)[..., None], u0, v0, scale, s, 'nearest')[..., 0] > 0.5
    self.valid = self.depth > 1e-3


def objective(obs, cov, z, col, hard, cfg):
  """Per-hypothesis cost, shape (B,). Lower is better. Every term is normalized by the mask size."""
  m = obs.mask[None]
  zo = obs.depth[None]
  vo = obs.valid[None]
  m_f = m.float()
  n_mask = m_f.sum((1, 2)).clamp_min(1.0)

  occluded = vo & hard & (zo < z - cfg.occ_gate)              # the sensor sees something in front of the render
  usable = hard & vo & m & ~occluded

  # Depth: Charbonnier cost, capped so that outliers saturate. Mask pixels the render fails to
  # cover pay the cap, otherwise sliding off the object would be a way to lower the cost.
  r = (z - zo) / cfg.sigma_depth
  rho = ((1 + r * r).sqrt() - 1).clamp(max=cfg.cost_cap)
  uncovered = m & vo & ~hard
  n_depth = (m & vo).float().sum((1, 2)).clamp_min(1.0)
  depth_cost = ((rho * usable.float()).sum((1, 2)) + cfg.cost_cap * uncovered.float().sum((1, 2))) / n_depth

  # Silhouette: squared difference of soft coverage and mask, except where an occluder hides the object.
  sil = (cov - m_f) ** 2 * (~occluded).float()
  sil_cost = sil.sum((1, 2)) / n_mask

  # Color: compare after normalizing each channel over the shared region, which removes the global
  # brightness and contrast differences between a texture map and a real photo.
  rgb_cost = torch.zeros_like(depth_cost)
  if col is not None and cfg.w_rgb > 0:
    w = usable.float()[..., None]
    n = w.sum((1, 2), keepdim=True).clamp_min(1.0)

    def normalize(x):
      mu = (x * w).sum((1, 2), keepdim=True) / n
      var = (((x - mu) ** 2) * w).sum((1, 2), keepdim=True) / n
      return (x - mu) / (var + 1e-6).sqrt()     # eps inside the sqrt: sqrt'(0) would make the gradient NaN

    diff = normalize(col) - normalize(obs.rgb[None].expand_as(col))
    per_pixel = ((diff * diff + 1e-4).sqrt() * w).sum((1, 2, 3)) / 3
    # As with depth, mask pixels the render fails to cover pay a fixed cost. Without this, sliding
    # the object out of view leaves nothing to compare, and a cost of zero is the best possible score.
    rgb_cost = (per_pixel + cfg.rgb_cap * (m & ~hard).float().sum((1, 2))) / n_mask

  total = cfg.w_depth * depth_cost + cfg.w_sil * sil_cost + cfg.w_rgb * rgb_cost
  return total, dict(depth=depth_cost, sil=sil_cost, rgb=rgb_cost)


def refine(scene, rgb, depth, mask, K, poses, cfg=None):
  """Refines each of the given poses, (B,4,4) in the object's model frame.

  Returns (refined poses, final cost per hypothesis, initial cost per hypothesis). The rotation
  update is applied about the object's center, so rotating does not also translate it.
  """
  cfg = cfg or RefineConfig()
  poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
  B = len(poses)
  obs = Observation(rgb, depth, mask, np.asarray(K, float), poses[0], scene.radius, cfg)
  dev = obs.depth.device
  Kc = obs.Kc
  size = cfg.crop_size

  R0 = torch.as_tensor(poses[:, :3, :3], device=dev, dtype=torch.float32)
  t0 = torch.as_tensor(poses[:, :3, 3], device=dev, dtype=torch.float32)
  c = scene.center
  center0 = (R0 @ c) + t0        # the object's center in the camera frame

  w = torch.zeros(B, 3, device=dev, requires_grad=True)
  d = torch.zeros(B, 3, device=dev, requires_grad=True)
  opt = torch.optim.Adam([{'params': [w], 'lr': cfg.lr_rot}, {'params': [d], 'lr': cfg.lr_trans}])
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1), eta_min=0.05 * cfg.lr_trans)
  need_rgb = cfg.w_rgb > 0

  def compose(w, d):
    R = so3_exp(w) @ R0
    return R, center0 + d - (R @ c)

  def cost_of(w, d):
    R, t = compose(w, d)
    cov, z, col, hard = scene.render(R, t, Kc, size, need_rgb)
    total, parts = objective(obs, cov, z, col, hard, cfg)
    return total, R, t

  with torch.no_grad():
    best_cost, _, _ = cost_of(w, d)
    initial_cost = best_cost.clone()
  best_w, best_d = w.detach().clone(), d.detach().clone()

  for _ in range(cfg.steps):
    opt.zero_grad(set_to_none=True)
    total, _, _ = cost_of(w, d)
    loss = total.sum()
    if cfg.w_prior > 0:
      loss = loss + cfg.w_prior * ((w * w).sum() + ((d / scene.radius) ** 2).sum())
    loss.backward()
    with torch.no_grad():
      # Keep the best iterate rather than the last one. A NaN pose rasterizes to nothing, which can
      # look cheap, so it has to be ruled out here rather than trusted to the cost.
      finite = torch.isfinite(w).all(-1) & torch.isfinite(d).all(-1)
      better = (total < best_cost) & finite
      best_cost = torch.where(better, total.detach(), best_cost)
      best_w[better], best_d[better] = w.detach()[better], d.detach()[better]
      for p in (w, d):
        if p.grad is not None:
          torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    opt.step()
    sched.step()

  with torch.no_grad():
    final_cost, R, t = cost_of(best_w, best_d)
    better = final_cost < best_cost
    best_cost = torch.where(better, final_cost, best_cost)
    accept = (best_cost < initial_cost * (1 - cfg.min_gain)) & torch.isfinite(R).flatten(1).all(1) & torch.isfinite(t).all(1)

  out = poses.copy()
  R, t = R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64)
  for b in range(B):
    if bool(accept[b]):
      out[b, :3, :3], out[b, :3, 3] = R[b], t[b]
  final = torch.where(accept, best_cost, initial_cost)
  return out, final.cpu().numpy(), initial_cost.cpu().numpy()
