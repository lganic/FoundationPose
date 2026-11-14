"""Runs FoundationPose over targets under one condition. Appends to a JSONL file and can resume.

  python scripts/run_fp.py --dataset ycbv --condition clean --tag fp_cnos
  python scripts/run_fp.py --dataset ycbv --condition motion_blur_11 --subset test_subset --tag fp_cnos
  python scripts/run_fp.py --dataset ycbv --condition clean --mask gt --tag fp
"""
import argparse, json, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose import corruptions
from drpose.data import REPO, SceneCache, load_splits, load_targets, stable_seed, target_key

TOP_K = 16   # hypotheses kept per target, for later re-ranking experiments


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', default='ycbv')
  p.add_argument('--condition', default='clean', choices=sorted(corruptions.CONDITIONS))
  p.add_argument('--subset', default=None, help='key in results/splits/ycbv.json, or all targets if omitted')
  p.add_argument('--limit', type=int, default=None)
  p.add_argument('--tag', default='fp', help='output folder under results/raw')
  p.add_argument('--iters', type=int, default=5)
  p.add_argument('--mask', default='cnos', choices=['cnos', 'gt'], help="'gt' is an oracle: BOP renders those masks from the GT pose")
  a = p.parse_args()

  from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, np, torch, trimesh

  targets = load_targets(a.dataset)
  if a.subset:
    keep = set(load_splits()[a.subset])
    targets = [t for t in targets if target_key(t) in keep]
    assert len(targets) == len(keep), 'subset contains unknown targets'
  targets.sort(key=lambda t: (t['obj'], t['scene'], t['im']))   # each object's mesh loads only once
  if a.limit:
    targets = targets[:a.limit]

  out = REPO / 'results' / 'raw' / a.tag / a.dataset / f'{a.condition}.jsonl'
  out.parent.mkdir(parents=True, exist_ok=True)
  done = set()
  if out.exists():
    with open(out) as f:
      done = {json.loads(line)['key'] for line in f if line.strip()}
  todo = [t for t in targets if target_key(t) not in done]
  print(f'{a.dataset}/{a.condition}: {len(targets)} targets, {len(done)} done, {len(todo)} to run', flush=True)
  if not todo:
    return

  cache = SceneCache()
  box = trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4)).to_mesh()
  est = FoundationPose(model_pts=box.vertices.copy(), model_normals=box.vertex_normals.copy(), mesh=box,
                       scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(),
                       debug=0, debug_dir=str(REPO / 'results' / 'raw' / 'scratch'))
  logging.getLogger().setLevel(logging.ERROR)

  current, started = None, time.time()
  with open(out, 'a') as f:
    for n, t in enumerate(todo):
      if t['obj'] != current:
        mesh = cache.mesh(t['dataset'], t['scene'], t['obj'])
        est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), mesh=mesh)
        current = t['obj']
      rgb, depth, mask, K = cache.load(t, a.mask)
      if mask is None or not mask.any():    # the detector missed this target, which counts as a failure
        f.write(json.dumps(dict(key=target_key(t), pose=None, seconds=0.0)) + '\n')
        f.flush()
        continue
      rgb, depth, mask = corruptions.apply(a.condition, rgb, depth, mask, stable_seed(a.condition, target_key(t)))
      est.poses = None
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=t['obj'], iteration=a.iters)
      torch.cuda.synchronize()
      dt = time.perf_counter() - t0
      row = dict(key=target_key(t), pose=np.asarray(pose, float).round(7).tolist(), seconds=round(dt, 3))
      if est.poses is not None:    # register() returns early, with no hypotheses, if too few valid pixels remain
        topk = (est.poses[:TOP_K] @ est.get_tf_to_centered_mesh()).cpu().numpy()
        row['topk_poses'] = topk.astype(float).round(6).tolist()
        row['topk_scores'] = est.scores[:TOP_K].float().cpu().numpy().astype(float).round(5).tolist()
      f.write(json.dumps(row) + '\n')
      f.flush()
      if n % 100 == 0 or n == len(todo) - 1:
        rate = (time.time() - started) / (n + 1)
        print(f'  {n + 1}/{len(todo)} obj {t["obj"]}  {rate:.2f} s/target  ~{rate * (len(todo) - n - 1) / 60:.0f} min left', flush=True)


if __name__ == '__main__':
  main()
