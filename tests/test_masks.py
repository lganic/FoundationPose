"""The RLE decoder must agree with the official toolkit."""
import sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import decode_rle, load_detections
from bop_toolkit_lib.pycoco_utils import rle_to_binary_mask


@pytest.mark.parametrize('dataset', ['ycbv', 'lmo'])
def test_decode_rle_matches_bop_toolkit(dataset):
  dets = list(load_detections(dataset).values())
  rng = np.random.default_rng(0)
  for i in rng.choice(len(dets), 50, replace=False):
    rle = dets[i]['segmentation']
    want = rle_to_binary_mask(rle).astype(bool)
    assert decode_rle(rle).shape == want.shape
    assert np.array_equal(decode_rle(rle), want)


def test_decoded_mask_matches_its_bbox():
  """Catches a transposed mask. CNOS boxes are loose and about 1% miss by more than 5 px, so this judges the group, not each box."""
  def gap(m, box):
    vs, us = np.nonzero(m)
    x, y, w, h = box
    return max(abs(us.min() - x), abs(vs.min() - y), abs(us.max() - (x + w)), abs(vs.max() - (y + h)))

  dets = list(load_detections('ycbv').values())[:200]
  masks = [decode_rle(d['segmentation']) for d in dets]
  right = np.array([gap(m, d['bbox']) for m, d in zip(masks, dets)])
  flipped = np.array([gap(m.T, d['bbox']) for m, d in zip(masks, dets)])
  assert np.median(right) <= 5 and (right > 5).mean() < 0.05
  assert np.median(flipped) > 20
