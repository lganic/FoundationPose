"""Pose error metrics. The scoring functions and thresholds are the official BOP toolkit's.

Evaluator.errors() gives per-target errors. auc() and ar() turn a table of them into the usual
summary numbers. A failed target (no pose) gets infinite errors, so it counts against the method
instead of silently leaving the denominator.
"""
import json

import cv2
import numpy as np

from .data import load_scene_json, models_dir, scene_dir

from bop_toolkit_lib import inout, misc, pose_error   # the path is set up by drpose.data

TAUS = np.arange(0.05, 0.51, 0.05)          # VSD misalignment tolerance, fraction of diameter
VSD_TH = np.arange(0.05, 0.51, 0.05)        # VSD correctness thresholds
MSSD_TH = np.arange(0.05, 0.51, 0.05)       # fraction of diameter
MSPD_TH = np.arange(5, 51, 5)               # pixels, at an image width of 640
VSD_DELTA = 15                              # mm, for both datasets
MAX_SYM_DISC_STEP = 0.01
ERROR_COLUMNS = ['add', 'adds', 'mssd', 'mspd', 'te', 're'] + [f'vsd{i}' for i in range(len(TAUS))]


def auc(errs_m, max_val=0.1, step=0.001):
  """Area under the accuracy-threshold curve up to 10 cm, as in the repo's compute_auc_sklearn. In [0,1]."""
  errs = np.sort(np.asarray(errs_m, float))
  xs = np.arange(0, max_val + step, step)
  ys = np.searchsorted(errs, xs, side='right') / len(errs)
  return float(np.trapezoid(ys, xs) / max_val) if hasattr(np, 'trapezoid') else float(np.trapz(ys, xs) / max_val)


def ar(df, image_width=640):
  """BOP Average Recall, in [0,1]. Takes a table with the columns from Evaluator.errors() plus 'diameter'."""
  mssd = np.mean([(df['mssd'] / df['diameter'] < th).mean() for th in MSSD_TH])
  mspd = np.mean([(df['mspd'] < th * image_width / 640).mean() for th in MSPD_TH])
  if df['vsd0'].isna().any():
    # VSD was skipped. NaN < th is False, so counting anyway would silently report a far lower AR.
    return dict(ar=float('nan'), ar_vsd=float('nan'), ar_mssd=float(mssd), ar_mspd=float(mspd))
  vsd = np.mean([(df[f'vsd{i}'] < th).mean() for i in range(len(TAUS)) for th in VSD_TH])
  return dict(ar=float((vsd + mssd + mspd) / 3), ar_vsd=float(vsd), ar_mssd=float(mssd), ar_mspd=float(mspd))


def rotation_angle_deg(R):
  """The rotation angle of R. Uses atan2, because arccos of the trace is badly conditioned near zero:
  BOP stores rotations to about 8 digits, which alone would read as a false 0.04 degree error."""
  s = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
  return float(np.degrees(np.arctan2(s, (np.trace(R) - 1) / 2)))


class DepthRenderer:
  """The one method pose_error.vsd() needs from a renderer, using nvdiffrast instead of vispy."""

  def __init__(self, glctx, mesh_tensors_for, height, width):
    self.glctx, self.mesh_tensors_for, self.h, self.w = glctx, mesh_tensors_for, height, width

  def render_object(self, obj_id, R, t, fx, fy, cx, cy):
    import torch
    from Utils import nvdiffrast_render
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = np.asarray(t).reshape(3) / 1e3
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    _, depth, _ = nvdiffrast_render(K=K, H=self.h, W=self.w, glctx=self.glctx, mesh_tensors=self.mesh_tensors_for(obj_id),
                                    ob_in_cams=torch.as_tensor(pose[None], device='cuda', dtype=torch.float))
    return {'depth': depth[0].cpu().numpy().astype(np.float64) * 1e3}


class Evaluator:
  def __init__(self, dataset, use_vsd=True):
    self.dataset = dataset
    self.use_vsd = use_vsd
    with open(f"{models_dir(dataset, 'models_eval')}/models_info.json") as f:
      self.info = {int(k): v for k, v in json.load(f).items()}
    self._pts, self._syms, self._mesh, self._depth = {}, {}, {}, {}
    self._renderer = None
    self._any_scene = 48 if dataset == 'ycbv' else 2    # meshes are per dataset; this only picks a reader to load them

  def diameter(self, obj):
    return self.info[obj]['diameter']

  def pts(self, obj):
    if obj not in self._pts:
      self._pts[obj] = inout.load_ply(f"{models_dir(self.dataset, 'models_eval')}/obj_{obj:06d}.ply")['pts']
    return self._pts[obj]

  def syms(self, obj):
    if obj not in self._syms:
      self._syms[obj] = misc.get_symmetry_transformations(self.info[obj], MAX_SYM_DISC_STEP)
    return self._syms[obj]

  def renderer(self):
    if self._renderer is None:
      import nvdiffrast.torch as dr
      from Utils import make_mesh_tensors
      from .data import SceneCache
      cache = SceneCache()
      glctx = dr.RasterizeCudaContext()

      def mesh_tensors_for(obj):
        if obj not in self._mesh:
          self._mesh[obj] = make_mesh_tensors(cache.mesh(self.dataset, self._any_scene, obj))
        return self._mesh[obj]

      self._renderer = DepthRenderer(glctx, mesh_tensors_for, 480, 640)
    return self._renderer

  def depth_mm(self, t):
    """The sensor's depth in mm, read the way the BOP toolkit reads it."""
    key = (t['scene'], t['im'])
    if key not in self._depth:
      self._depth = {}     # one frame's worth is enough, because targets arrive grouped by image
      scale = load_scene_json(self.dataset, t['scene'])['scene_camera'][str(t['im'])]['depth_scale']
      self._depth[key] = cv2.imread(f"{scene_dir(self.dataset, t['scene'])}/depth/{t['im']:06d}.png", -1).astype(np.float64) * scale
    return self._depth[key]

  def K(self, t):
    return np.array(load_scene_json(self.dataset, t['scene'])['scene_camera'][str(t['im'])]['cam_K']).reshape(3, 3)

  def errors(self, t, pose):
    """Per-target errors. add/adds/mssd/te are in mm, mspd in pixels, re in degrees, vsd in [0,1]."""
    if pose is None or not np.isfinite(pose).all():
      return {c: (1.0 if c.startswith('vsd') else np.inf) for c in ERROR_COLUMNS}
    obj = t['obj']
    R_e, t_e = np.asarray(pose)[:3, :3], (np.asarray(pose)[:3, 3] * 1e3).reshape(3, 1)
    R_g, t_g = t['gt_pose'][:3, :3], (t['gt_pose'][:3, 3] * 1e3).reshape(3, 1)
    pts, syms, K = self.pts(obj), self.syms(obj), self.K(t)
    out = dict(add=pose_error.add(R_e, t_e, R_g, t_g, pts),
               adds=pose_error.adi(R_e, t_e, R_g, t_g, pts),
               mssd=pose_error.mssd(R_e, t_e, R_g, t_g, pts, syms),
               mspd=pose_error.mspd(R_e, t_e, R_g, t_g, K, pts, syms))
    # Rotation and translation errors under the object's best-fitting symmetry.
    best = min(syms, key=lambda s: np.linalg.norm(misc.transform_pts_Rt(pts, R_e, t_e) - misc.transform_pts_Rt(pts, R_g @ s['R'], R_g @ s['t'] + t_g), axis=1).max())
    R_s, t_s = R_g @ best['R'], R_g @ best['t'] + t_g
    out['re'] = rotation_angle_deg(R_e @ R_s.T)
    out['te'] = float(np.linalg.norm(t_e - t_s))
    if self.use_vsd:
      vsd = pose_error.vsd(R_e, t_e, R_g, t_g, self.depth_mm(t), K, VSD_DELTA, TAUS, True, self.diameter(obj), self.renderer(), obj, 'step')
      out.update({f'vsd{i}': float(e) for i, e in enumerate(vsd)})
    else:
      out.update({f'vsd{i}': np.nan for i in range(len(TAUS))})
    return out
