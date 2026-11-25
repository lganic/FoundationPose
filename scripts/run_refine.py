"""Post-processes saved FoundationPose results with one refinement method.

Every method starts from the same saved poses and sees the same frame, corrupted the same way,
with the same detector mask, so that differences come only from the method.

  python scripts/run_refine.py --method dr --condition clean --subset dev_subset --name dr_base
  python scripts/run_refine.py --method icp --condition clean --subset dev_subset
  python scripts/run_refine.py --method fp_iters --condition clean --subset dev_subset
"""
import argparse, dataclasses, json, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose import corruptions
from drpose.data import REPO, SceneCache, subset_keys, load_targets, stable_seed, target_key


def parse_overrides(pairs, cfg):
  """Turns ['w_rgb=0', 'steps=30'] into config fields, refusing unknown names."""
  fields = {f.name: f.type for f in dataclasses.fields(cfg)}
  for pair in pairs:
    k, v = pair.split('=')
    if k not in fields:
      raise SystemExit(f'unknown config field {k}')
    setattr(cfg, k, type(getattr(cfg, k))(float(v)))
  return cfg


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--method', required=True, choices=['dr', 'dr_rerank', 'icp', 'fp_iters'])
  p.add_argument('--name', default=None, help='output tag, by default the method name')
  p.add_argument('--source', default='fp_cnos', help='folder under results/raw holding the starting poses')
  p.add_argument('--dataset', default='ycbv')
  p.add_argument('--condition', default='clean', choices=sorted(corruptions.CONDITIONS))
  p.add_argument('--subset', default=None)
  p.add_argument('--mask', default='cnos', choices=['cnos', 'gt'])
  p.add_argument('--top_k', type=int, default=4, help='hypotheses to refine and compare, for dr_rerank')
  p.add_argument('--extra_iters', type=int, default=3, help='added refiner iterations, for fp_iters')
  p.add_argument('--set', nargs='*', default=[], help='config overrides such as w_rgb=0 steps=30')
  a = p.parse_args()

  import nvdiffrast.torch as dr
  import numpy as np
  import torch
  from Utils import bilateral_filter_depth, compute_mesh_diameter, depth2xyzmap, erode_depth, make_mesh_tensors
  from drpose.controls import icp
  from drpose.metrics import Evaluator
  from drpose.refine import RefineConfig, Scene, refine

  name = a.name or a.method
  src = REPO / 'results' / 'raw' / a.source / a.dataset / f'{a.condition}.jsonl'
  with open(src) as f:
    start = {r['key']: r for r in map(json.loads, f) if r}
  targets = [t for t in load_targets(a.dataset) if target_key(t) in start]
  if a.subset:
    keep = subset_keys(a.subset)
    targets = [t for t in targets if target_key(t) in keep]
  targets.sort(key=lambda t: (t['obj'], t['scene'], t['im']))

  out = REPO / 'results' / 'raw' / name / a.dataset / f'{a.condition}.jsonl'
  out.parent.mkdir(parents=True, exist_ok=True)
  done = set()
  if out.exists():
    with open(out) as f:
      done = {json.loads(line)['key'] for line in f if line.strip()}
  todo = [t for t in targets if target_key(t) not in done]
  print(f'{name} {a.dataset}/{a.condition}: {len(targets)} targets, {len(todo)} to run', flush=True)
  if not todo:
    return

  cfg = parse_overrides(a.set, RefineConfig())
  ev = Evaluator(a.dataset, use_vsd=False)
  cache = SceneCache()
  glctx = dr.RasterizeCudaContext()
  refiner = None
  if a.method == 'fp_iters':
    from estimater import PoseRefinePredictor
    refiner = PoseRefinePredictor()
  logging.getLogger().setLevel(logging.ERROR)

  current = None
  with open(out, 'a') as f:
    for n, t in enumerate(todo):
      key = target_key(t)
      row = start[key]
      if row['pose'] is None:      # the detector missed it, so there is nothing to refine
        f.write(json.dumps(dict(key=key, pose=None, seconds=0.0)) + '\n')
        continue
      if 'topk_poses' not in row:  # FoundationPose gave up (too few valid pixels): its pose is a placeholder, not an estimate
        f.write(json.dumps(dict(key=key, pose=row['pose'], seconds=0.0)) + '\n')
        continue
      if t['obj'] != current:
        mesh = cache.mesh(t['dataset'], t['scene'], t['obj'])
        mt = make_mesh_tensors(mesh)
        scene = Scene(mt, glctx, ev.diameter(t['obj']) / 1e3)
        pts, faces = mesh.sample(8000, return_index=True)
        sample_pts, sample_nrm = np.asarray(pts), np.asarray(mesh.face_normals[faces])
        if a.method == 'fp_iters':   # FoundationPose works on a mesh centered at its bounding box
          to_center = np.eye(4)
          to_center[:3, 3] = -0.5 * (mesh.vertices.max(0) + mesh.vertices.min(0))
          centered = mesh.copy()
          centered.vertices = centered.vertices + to_center[:3, 3]
          centered_mt = make_mesh_tensors(centered)
          fp_diameter = compute_mesh_diameter(model_pts=centered.vertices, n_sample=10000)   # as in FoundationPose.reset_object()
        current = t['obj']
      rgb, depth, mask, K = cache.load(t, a.mask)
      rgb, depth, mask = corruptions.apply(a.condition, rgb, depth, mask, stable_seed(a.condition, key))
      # The same depth cleanup FoundationPose.register() applies, so that no method gets better input.
      depth = bilateral_filter_depth(erode_depth(depth, radius=2, device='cuda'), radius=2, device='cuda')
      pose0 = np.array(row['pose'])
      extra = {}
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      if a.method == 'dr':
        poses, final, initial = refine(scene, rgb, depth, mask, K, pose0[None], cfg)
        pose = poses[0]
        extra = dict(cost0=round(float(initial[0]), 5), cost1=round(float(final[0]), 5))
      elif a.method == 'dr_rerank':
        cands = np.array(row.get('topk_poses', [row['pose']]))[:a.top_k]
        poses, final, initial = refine(scene, rgb, depth, mask, K, cands, cfg)
        pick = int(np.argmin(final))
        pose = poses[pick]
        extra = dict(pick=pick)
      elif a.method == 'icp':
        pose = icp(sample_pts, sample_nrm, depth, mask, K, pose0)
      else:
        res, _ = refiner.predict(mesh=centered, mesh_tensors=centered_mt, rgb=rgb, depth=depth, K=K, ob_in_cams=(pose0 @ np.linalg.inv(to_center))[None],
                                 xyz_map=depth2xyzmap(depth, K), glctx=glctx, mesh_diameter=fp_diameter, iteration=a.extra_iters)
        pose = res[0].detach().cpu().numpy() @ to_center
      torch.cuda.synchronize()
      dt = time.perf_counter() - t0
      f.write(json.dumps(dict(key=key, pose=np.asarray(pose, float).round(7).tolist(), seconds=round(dt, 3), **extra)) + '\n')
      f.flush()
      if n % 50 == 0 or n == len(todo) - 1:
        print(f'  {n + 1}/{len(todo)}  {dt:.2f} s', flush=True)


if __name__ == '__main__':
  main()
