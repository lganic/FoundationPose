"""Controlled image degradations for the sensitivity analysis.

Each condition is a pure function of the inputs and a seed, so that every method sees the exact
same corrupted frame. Masks stay ground truth unless the condition is explicitly about masks,
which isolates each failure mode from detector errors.
"""
import cv2
import numpy as np

CONDITIONS = {'clean': ('clean', 0)}
for _s in (10, 25, 50):
  CONDITIONS[f'rgb_noise_{_s}'] = ('rgb_noise', _s)            # sigma, in 8-bit gray levels
for _k in (5, 11, 21):
  CONDITIONS[f'motion_blur_{_k}'] = ('motion_blur', _k)        # kernel length, in pixels
for _s in (5, 10, 20):
  CONDITIONS[f'depth_noise_{_s}'] = ('depth_noise', _s)        # sigma, in millimeters
for _p in (20, 40, 60):
  CONDITIONS[f'depth_holes_{_p}'] = ('depth_holes', _p)        # percent of pixels lost
for _p in (25, 50, 75):
  CONDITIONS[f'occlusion_{_p}'] = ('occlusion', _p)            # percent of the visible object covered
for _r in (3, 7):
  CONDITIONS[f'mask_erode_{_r}'] = ('mask_erode', _r)          # radius, in pixels
  CONDITIONS[f'mask_dilate_{_r}'] = ('mask_dilate', _r)
CONDITIONS['mask_box'] = ('mask_box', 0)                       # a detector box in place of a mask


def _rgb_noise(rgb, sigma, rng):
  return np.clip(rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape), 0, 255).astype(np.uint8)


def _motion_blur(rgb, length, rng):
  kernel = np.zeros((length, length), np.float32)
  c, a = length // 2, rng.uniform(0, np.pi)
  dx, dy = np.cos(a) * c, np.sin(a) * c
  cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1, cv2.LINE_AA)
  return cv2.filter2D(rgb, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REPLICATE)


def _depth_noise(depth, sigma_mm, rng):
  out = depth + rng.normal(0, sigma_mm / 1e3, depth.shape)
  out[depth < 1e-3] = 0     # a pixel the sensor lost stays lost
  return np.maximum(out, 0)


def _depth_holes(depth, percent, rng):
  """Blobby dropout like a real depth sensor's, not salt-and-pepper noise."""
  h, w = depth.shape
  field = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 6)
  out = depth.copy()
  out[field < np.quantile(field, percent / 100)] = 0
  return out


def _occlude(rgb, depth, mask, percent, rng):
  """Slides an occluder over the object from one side until it hides `percent` of the visible pixels.

  The occluder is a textured surface placed 5-10 cm in front of the object, so it is consistent
  in RGB, depth and mask, like a real object in front of the target. It has a wobbly straight
  edge, and what remains visible stays connected, as it does for real occlusion.
  """
  vs, us = np.nonzero(mask)
  if len(us) == 0:
    return rgb, depth, mask
  h, w = mask.shape
  angle = rng.uniform(0, 2 * np.pi)
  ca, sa = np.cos(angle), np.sin(angle)
  proj = us * ca + vs * sa
  yy, xx = np.mgrid[0:h, 0:w]
  coord = xx * ca + yy * sa
  wobble = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 12)
  coord_w = coord + wobble * (0.1 * (proj.max() - proj.min()) / (wobble.std() + 1e-6))
  cut = np.quantile(coord_w[mask], percent / 100)     # measured on the wobbly edge, so coverage is exact
  occ = (coord_w <= cut) & (coord >= proj.min() - 0.3 * (proj.max() - proj.min()))

  base = rng.uniform(40, 215, 3).astype(np.float32)
  tex = cv2.GaussianBlur(rng.normal(0, 90, (h, w, 3)).astype(np.float32), (0, 0), 3)
  out_rgb = rgb.copy()
  out_rgb[occ] = np.clip(base + tex[occ], 0, 255).astype(np.uint8)
  valid = mask & (depth > 1e-3)
  obj_depth = np.median(depth[valid]) if valid.any() else 0.8
  out_depth = depth.copy()
  out_depth[occ] = max(obj_depth - rng.uniform(0.05, 0.10), 0.05)
  return out_rgb, out_depth, mask & ~occ


def _disk(radius):
  return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def apply(condition, rgb, depth, mask, seed):
  """Returns corrupted copies of (rgb, depth, mask). Never changes the inputs."""
  kind, level = CONDITIONS[condition]
  rng = np.random.default_rng(seed)
  rgb, depth, mask = rgb.copy(), depth.copy(), mask.copy()
  if kind == 'rgb_noise':
    rgb = _rgb_noise(rgb, level, rng)
  elif kind == 'motion_blur':
    rgb = _motion_blur(rgb, level, rng)
  elif kind == 'depth_noise':
    depth = _depth_noise(depth, level, rng)
  elif kind == 'depth_holes':
    depth = _depth_holes(depth, level, rng)
  elif kind == 'occlusion':
    rgb, depth, mask = _occlude(rgb, depth, mask, level, rng)
  elif kind == 'mask_erode':
    eroded = cv2.erode(mask.astype(np.uint8), _disk(level)) > 0
    mask = eroded if eroded.sum() >= 16 else mask     # a tiny object must not vanish entirely
  elif kind == 'mask_dilate':
    mask = cv2.dilate(mask.astype(np.uint8), _disk(level)) > 0
  elif kind == 'mask_box':
    vs, us = np.nonzero(mask)
    if len(us):
      mask = np.zeros_like(mask)
      mask[vs.min():vs.max() + 1, us.min():us.max() + 1] = True
  return rgb, depth, mask
