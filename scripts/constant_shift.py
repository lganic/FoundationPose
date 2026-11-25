"""A control that looks at no image: FoundationPose's poses plus one fixed shift.

Exploratory, added after the refiner gained on YCB-V, lost on LM-O, and moved poses up by about the
same amount on both. If one fixed nudge explains that, then adding the refiner's typical motion to
every pose, with no photo or depth involved, should reproduce the gain and the loss. The flipped
shift is the falsifier: if direction is what matters, it should do the opposite.

The constants come from the dev scenes only, from the frozen configuration's dev run, so nothing
here is fitted to test data.

  python scripts/constant_shift.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from drpose.data import REPO, SceneCache, load_scene_json, load_splits, load_targets, target_key

DEV_RUN = 'dr2_depth_s3_30'   # frozen settings, dev_subset
BASE = 'fp_cnos'


def read(tag, dataset):
  rows = {}
  with open(REPO / 'results/raw' / tag / dataset / 'clean.jsonl') as f:
    for line in f:
      if line.endswith('\n'):   # the base file may still be growing
        r = json.loads(line)
        rows[r['key']] = r
  return rows


def center_of(cache, t):
  v = np.asarray(cache.mesh(t['dataset'], t['scene'], t['obj']).vertices)
  return 0.5 * (v.max(0) + v.min(0))


def image_place(P, c, K):
  x = P[:3, :3] @ c + P[:3, 3]
  return K[0, 0] * x[0] / x[2] + K[0, 2], K[1, 1] * x[1] / x[2] + K[1, 2], x[2] * 1e3


def main():
  cache = SceneCache()
  targets = {target_key(t): t for t in load_targets('ycbv')}
  base, moved = read(BASE, 'ycbv'), read(DEV_RUN, 'ycbv')
  rows = []
  for k in load_splits()['dev_subset']:
    if base[k]['pose'] is None or moved[k]['pose'] is None:
      continue
    t = targets[k]
    K = cache.load(t, 'cnos')[3]
    c = center_of(cache, t)
    a, b = image_place(np.array(base[k]['pose']), c, K), image_place(np.array(moved[k]['pose']), c, K)
    rows.append([b[i] - a[i] for i in range(3)])
  d = pd.DataFrame(rows, columns=['du', 'dv', 'dz'])
  shift = d.median().to_dict()
  print(f'dev motion, {len(d)} targets. median du {shift["du"]:+.3f} px, dv {shift["dv"]:+.3f} px, dz {shift["dz"]:+.3f} mm')
  print(f'spread between targets (sd): du {d.du.std():.2f} px, dv {d.dv.std():.2f} px, dz {d.dz.std():.2f} mm')
  pd.DataFrame([shift]).round(4).to_csv(REPO / 'results/tables/constant_shift.csv', index=False)

  for dataset, keep in (('ycbv', set(load_splits()['test_subset'])), ('lmo', None)):
    info = {target_key(t): t for t in load_targets(dataset)}
    rows = read(BASE, dataset)
    cameras = {s: load_scene_json(dataset, s)['scene_camera'] for s in {t['scene'] for t in info.values()}}
    # Fitted to each dataset's own test motion, which favors the idea of a fixed nudge. A diagnostic
    # only: it tests that idea at its strongest, and is not a usable method.
    e = pd.read_csv(REPO / f'results/tables/edge_fit_{dataset}.csv')
    own = dict(du=(e.dr_final_u - e.fp_cnos_u).median(), dv=(e.dr_final_v - e.fp_cnos_v).median(), dz=(e.dr_final_z - e.fp_cnos_z).median())
    print(f'{dataset} own test motion: du {own["du"]:+.3f} px, dv {own["dv"]:+.3f} px, dz {own["dz"]:+.3f} mm')
    for tag, sign, amount in (('const_shift', 1, shift), ('const_shift_neg', -1, shift), ('const_own', 1, own)):
      dst = REPO / 'results/raw' / tag / dataset / 'clean.jsonl'
      dst.parent.mkdir(parents=True, exist_ok=True)
      n = 0
      with open(dst, 'w') as f:
        for k, r in rows.items():
          if keep is not None and k not in keep:
            continue
          pose = r['pose']
          if pose is not None and pose[2][3] > 0.01:   # leave FoundationPose's give-up placeholder alone
            P = np.array(pose)
            t = info[k]
            K = np.array(cameras[t['scene']][str(t['im'])]['cam_K']).reshape(3, 3)
            x = P[:3, :3] @ center_of(cache, t) + P[:3, 3]
            u, v, z = K[0, 0] * x[0] / x[2] + K[0, 2], K[1, 1] * x[1] / x[2] + K[1, 2], x[2]
            u, v, z = u + sign * amount['du'], v + sign * amount['dv'], z + sign * amount['dz'] / 1e3
            P[:3, 3] += np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]) - x
            pose = P.tolist()
          f.write(json.dumps(dict(key=k, pose=pose, seconds=0.0)) + '\n')
          n += 1
      print(f'{tag}/{dataset}: {n} poses written')


if __name__ == '__main__':
  main()
