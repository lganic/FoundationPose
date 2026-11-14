'''Dataset access. Everything is keyed by (dataset, scene, im, obj), which identifies one target.'''
import json
import os
import random
import sys
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BOP_DIR = (REPO / 'data' / 'BOP').as_posix()   # the readers look for 'BOP' in the path and split on '/'

os.environ['BOP_DIR'] = BOP_DIR
os.environ['YCB_VIDEO_DIR'] = f'{BOP_DIR}/ycbv/ycbv_models'

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'third_party' / 'bop_toolkit'))

SPLIT_SEED = 20261105
DATASETS = ('ycbv', 'lmo')

def scene_dir(dataset, scene):
  
  if dataset == 'ycbv':
    return f'{BOP_DIR}/ycbv/test/{scene:06d}'
  
  if dataset == 'lmo':
    return f'{BOP_DIR}/lmo/lmo_test_bop19/test/{scene:06d}'
  
  raise ValueError(dataset)

def models_dir(dataset, kind='models'):
  '''
  kind: 'models' for rendering, 'models_eval' for metrics, as BOP intends
  '''
  return f'{BOP_DIR}/ycbv/ycbv_models/{kind}' if dataset == 'ycbv' else f'{BOP_DIR}/lmo/{kind}'


def stable_seed(*parts):
  '''
  A seed that is the same on every run and machine, unlike Python's hash()
  '''
  return zlib.crc32('|'.join(str(p) for p in parts).encode())

def target_key(t):
  return f"{t['dataset']}/{t['scene']}/{t['im']}/{t['obj']}"


@lru_cache(maxsize=None) # Caching this, cause realistically this is small, and cuts down on file reads
def load_scene_json(dataset, scene):

  # Get the specific dir for the dataset we are working on
  d = scene_dir(dataset, scene)

  # Load up the data for each section.
  out = {}
  for name in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json'):
    with open(f'{d}/{name}') as json_file:
      out[name] = json.load(json_file) # Save loaded file info to output

  return out

def load_targets(dataset):
  '''
  All BOP'19 test targets, with GT pose and visibility attached
  '''
  
  # Grab targets from file
  with open(f'{BOP_DIR}/{dataset}/test_targets_bop19.json') as f:
    rows = json.load(f)

  targets = []
  # Loop over targets
  for target in rows:
    assert target['inst_count'] == 1, 'the mask lookup assumes one instance per object per image'

    scene = target['scene_id']
    im = target['im_id']
    obj = target['obj_id']

    s = load_scene_json(dataset, scene)
    gts = s['scene_gt'][str(im)]

    idx = [i for i, g in enumerate(gts) if g['obj_id'] == obj] # indexes of objects in scene which match the object id

    assert len(idx) == 1

    gt, info = gts[idx[0]], s['scene_gt_info'][str(im)][idx[0]]
    pose = np.eye(4)
    pose[:3, :3] = np.array(gt['cam_R_m2c']).reshape(3, 3)
    pose[:3, 3] = np.array(gt['cam_t_m2c']) / 1e3

    targets.append(dict(dataset=dataset, scene=scene, im=im, obj=obj, gt_index=idx[0], gt_pose=pose,
                        visib_fract=info['visib_fract'], px_visib=info['px_count_visib']))

  return targets


def make_splits(n_dev_scenes=3, n_subset=300, n_dev_subset=150):
  '''
  Fixed before any results exist, so the refiner is never tuned on data it is reported on.

  YCB-V splits by scene rather than by frame, because frames of one video are near-duplicates.
  LM-O has a single scene, so all of it stays test-only and nothing is ever tuned on it.
  '''
  
  targets = load_targets('ycbv')
  scenes = sorted({t['scene'] for t in targets})

  rng = random.Random(SPLIT_SEED)

  dev_scenes = sorted(rng.sample(scenes, n_dev_scenes))
  test_scenes = [s for s in scenes if s not in dev_scenes]

  def balanced_sample(pool, n):
    '''Round-robin over objects, so that no single object dominates the subset.'''
    by_obj = {}

    # 
    for t in pool:
      
      if t['obj'] not in by_obj:
        by_obj[t['obj']] = []

      by_obj[t['obj']].append(t)

    # Shuffle all the values.
    for v in by_obj.values():
      rng.shuffle(v)

    picked, objs = [], sorted(by_obj)

    while len(picked) < min(n, len(pool)):
      for o in objs:
        if by_obj[o] and len(picked) < n:
          picked.append(by_obj[o].pop())

    return sorted(target_key(t) for t in picked)

  return dict(seed=SPLIT_SEED, dev_scenes=dev_scenes, test_scenes=test_scenes,
              dev_subset=balanced_sample([t for t in targets if t['scene'] in dev_scenes], n_dev_subset),
              test_subset=balanced_sample([t for t in targets if t['scene'] in test_scenes], n_subset))


def load_splits():
  with open(REPO / 'results' / 'splits' / 'ycbv.json') as f:
    return json.load(f)


def subset_keys(name):
  '''
  Target keys of a named YCB-V subset. test_all is every target in the test scenes; dev scenes stay out.
  '''
  s = load_splits()
  if name == 'test_all':
    return {target_key(t) for t in load_targets('ycbv') if t['scene'] in s['test_scenes']}
  return set(s[name])


def decode_rle(rle):
  '''
  Uncompressed COCO RLE to a bool mask. Runs alternate background/foreground, in column-major order.
  '''
  counts = np.asarray(rle['counts'], dtype=np.int64)
  h, w = rle['size']
  flat = np.repeat(np.arange(len(counts)) % 2 == 1, counts)
  assert flat.size == h * w, 'RLE does not cover the image'
  return flat.reshape((h, w), order='F')


@lru_cache(maxsize=None)
def load_detections(dataset):
  '''
  BOP's default CNOS detections: the top-scoring one for each (scene, im, obj).

  These come from a real detector run on the image, so unlike the mask_visib files, which BOP
  renders from the GT pose, they carry no information about the GT pose.
  '''
  (path,) = (REPO / 'data' / 'BOP' / 'detections').glob(f'cnos-fastsam_{dataset}-test_*.json')
  with open(path) as f:
    rows = json.load(f)
  best = {}
  for r in rows:
    key = (r['scene_id'], r['image_id'], r['category_id'])
    if key not in best or r['score'] > best[key]['score']:
      best[key] = r
  return best


class SceneCache:
  '''
  Opens each scene's reader once. Imports the FoundationPose stack lazily, because it is slow.
  '''

  def __init__(self):
    self.readers = {}

  def reader(self, dataset, scene):
    from datareader import LinemodOcclusionReader, YcbVideoReader
    key = (dataset, scene)
    if key not in self.readers:
      if dataset == 'ycbv':
        self.readers[key] = YcbVideoReader(scene_dir(dataset, scene), zfar=1.5)   # as in run_ycb_video.py
      else:
        self.readers[key] = LinemodOcclusionReader(scene_dir(dataset, scene), zfar=np.inf)
    return self.readers[key]

  def load(self, t, mask_source='gt'):
    '''Returns rgb (H,W,3 uint8), depth (H,W meters), mask (H,W bool, or None if undetected) and K.

    mask_source 'gt' is BOP's visible mask, which is derived from the GT pose, so it may only be
    used as an oracle. 'cnos' is the real detection.
    '''
    r = self.reader(t['dataset'], t['scene'])
    i = r.id_strs.index(f"{t['im']:06d}")
    if mask_source == 'gt':
      mask = r.get_mask(i, t['obj'], type='mask_visib') > 0
    elif mask_source == 'cnos':
      det = load_detections(t['dataset']).get((t['scene'], t['im'], t['obj']))
      mask = decode_rle(det['segmentation']) if det else None
    else:
      raise ValueError(mask_source)
    return r.get_color(i)[..., :3], r.get_depth(i), mask, r.get_K(i).copy()

  def mesh(self, dataset, scene, obj):
    return self.reader(dataset, scene).get_gt_mesh(obj)
