"""Checks that each image on the blinded page really shows the pose the key says it shows.

The key was written by the same code that drew the images, so reading that code is not enough.
This decodes the images out of the page, redraws both candidate outlines from the saved poses, and
asks which one each picture matches. A swapped key would turn every conclusion upside down.

  python scripts/verify_labeling_key.py [--page path/to/page.html]
"""
import argparse, base64, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import REPO, SceneCache, load_splits, load_targets, target_key

WIDTH = 420


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--page', default=str(REPO / 'results/labeling/outline_fit_check.html'))
  a = p.parse_args()

  import cv2
  import numpy as np
  from drpose.cpu_render import silhouette as cpu_silhouette

  html = Path(a.page).read_text(encoding='utf-8')
  pairs = json.loads(re.search(r'const PAIRS = (\[.*?\]);\s*\n', html, re.S).group(1))
  key = json.load(open(REPO / 'results/labeling/key.json'))
  print(f'{len(pairs)} pairs on the page, {len(key)} in the key, ids equal: {set(q["id"] for q in pairs) == set(key)}')

  keys = set(load_splits()['test_subset'])
  targets = {target_key(t): t for t in load_targets('ycbv') if target_key(t) in keys}

  def read(tag):
    with open(REPO / 'results/raw' / tag / 'ycbv/clean.jsonl') as f:
      return {r['key']: np.array(r['pose']) for r in map(json.loads, f) if r['pose'] is not None}

  poses = dict(base=read('fp_cnos'), refined=read('dr_final'))
  cache = SceneCache()

  def silhouette(t, pose, K, H, W):
    m = cache.mesh('ycbv', t['scene'], t['obj'])
    return cpu_silhouette(np.asarray(m.vertices), np.asarray(m.faces), pose, K, H, W)

  def decode(uri):
    return cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)

  right = wrong = 0
  margins = []
  for q in pairs:
    k = key[q['id']]
    t = targets[k['target']]
    rgb, depth, _, K = cache.load(t, 'cnos')
    H, W = depth.shape
    pose = lambda name: t['gt_pose'] if name == 'gt' else poses[name][k['target']]
    sA, sB = silhouette(t, pose(k['A']), K, H, W), silhouette(t, pose(k['B']), K, H, W)
    vs, us = np.nonzero(sA | sB)
    pad = 30
    y0, y1, x0, x1 = max(vs.min() - pad, 0), min(vs.max() + pad, H), max(us.min() - pad, 0), min(us.max() + pad, W)
    pic_a, pic_b = decode(q['a']), decode(q['b'])
    size = (pic_a.shape[1], pic_a.shape[0])   # this rasterizer can differ from the page's by a pixel at the crop's edge

    def line(sil):
      m = cv2.resize(sil[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
      contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      out = np.zeros(m.shape, np.uint8)
      cv2.drawContours(out, contours, -1, 1, 2, cv2.LINE_8)
      return out.astype(bool)

    def yellow(img):   # the drawn line is the only saturated yellow in the picture
      return (img[..., 0] > 200) & (img[..., 1] > 180) & (img[..., 2] < 110)

    la, lb = line(sA), line(sB)
    ya, yb = yellow(pic_a), yellow(pic_b)
    hit = lambda y, l: (y & l).sum() / max(l.sum(), 1)
    keep = hit(ya, la) + hit(yb, lb)   # as the key says
    swap = hit(ya, lb) + hit(yb, la)   # the other way round
    margins.append(keep - swap)
    right += keep > swap
    wrong += keep < swap
  print(f'key matches the pictures on {right} pairs, contradicts them on {wrong}, undecided {len(pairs) - right - wrong}')
  print(f'margin of match, median {np.median(margins):.3f}, smallest {np.min(margins):.3f}')


if __name__ == '__main__':
  main()
