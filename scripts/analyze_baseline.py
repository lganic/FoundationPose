"""Week 9-10 analysis: sanity check, then where FoundationPose fails and why.

Reads the per-target CSVs from scripts/evaluate.py and writes tables to results/tables and
charts to results/figures. Any input that has not been produced yet is skipped with a note.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from drpose.data import REPO, SceneCache, load_detections, decode_rle, load_splits, load_targets, target_key
from drpose.metrics import auc
from drpose.stats import METRICS, paired_bootstrap, summarize

TABLES, FIGS, RAW = REPO / 'results' / 'tables', REPO / 'results' / 'figures', REPO / 'results' / 'raw'
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e2', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'figure.dpi': 160})

CORRUPTION_LABELS = {
  'occlusion_25': 'Occlusion 25%', 'occlusion_50': 'Occlusion 50%', 'occlusion_75': 'Occlusion 75%',
  'motion_blur_11': 'Motion blur 11 px', 'motion_blur_21': 'Motion blur 21 px',
  'rgb_noise_25': 'RGB noise σ=25', 'rgb_noise_50': 'RGB noise σ=50',
  'depth_noise_10': 'Depth noise 10 mm', 'depth_noise_20': 'Depth noise 20 mm',
  'depth_holes_40': 'Depth holes 40%', 'depth_holes_60': 'Depth holes 60%',
  'mask_dilate_7': 'Mask dilated 7 px', 'mask_erode_7': 'Mask eroded 7 px', 'mask_box': 'Box instead of mask',
}


def load(tag, dataset, condition):
  p = RAW / 'eval' / tag / dataset / f'{condition}.csv'
  return pd.read_csv(p) if p.exists() else None


def is_failure(df):
  """A pose is a failure if its worst surface point is off by more than 10% of the object's diameter."""
  return df['mssd'] / df['diameter'] > 0.10


def row(df, **labels):
  s = summarize(df)
  return {**labels, 'n': s['n'], 'ADD-S AUC': s['adds_auc'], 'ADD AUC': s['add_auc'], 'AR': s['ar'], 'AR VSD': s['ar_vsd'],
          'AR MSSD': s['ar_mssd'], 'AR MSPD': s['ar_mspd'], 'failure %': 100 * is_failure(df).mean(),
          'median t mm': s['te_median'], 'median R deg': s['re_median']}


def sanity():
  """Compares with the paper's only model-based numbers: BOP AR with Mask R-CNN detections."""
  published = {'ycbv': 88.0, 'lmo': 78.8}
  rows = []
  for ds in ('ycbv', 'lmo'):
    for tag, masks in (('fp_cnos', 'CNOS detector'), ('fp', 'GT (oracle)')):
      df = load(tag, ds, 'clean')
      if df is not None and len(df):
        rows.append(row(df, dataset=ds, masks=masks, published_AR=published[ds]))
  out = pd.DataFrame(rows)
  out.round(2).to_csv(TABLES / 'sanity.csv', index=False)
  print(out.round(1).to_string(index=False))


def detection_quality(dataset):
  """IoU of the top CNOS detection with the GT visible mask. Used only to attribute failures."""
  path = RAW / f'detector_iou_{dataset}.csv'
  if path.exists():
    return pd.read_csv(path)
  cache, det = SceneCache(), load_detections(dataset)
  rows = []
  for t in load_targets(dataset):
    d = det.get((t['scene'], t['im'], t['obj']))
    iou = 0.0
    if d is not None:
      m = decode_rle(d['segmentation'])
      r = cache.reader(dataset, t['scene'])
      g = r.get_mask(r.id_strs.index(f"{t['im']:06d}"), t['obj'], type='mask_visib') > 0
      iou = float((m & g).sum() / max((m | g).sum(), 1))
    rows.append(dict(key=target_key(t), detected=d is not None, det_iou=iou))
  out = pd.DataFrame(rows)
  out.to_csv(path, index=False)
  return out


def binned(df, column, edges, labels):
  df = df.copy()
  df['bin'] = pd.cut(df[column], edges, labels=labels, include_lowest=True)
  return [row(g, bin=str(b)) for b, g in df.groupby('bin', observed=True) if len(g) >= 10]


def breakdown(dataset):
  df = load('fp_cnos', dataset, 'clean')
  if df is None:
    print(f'{dataset}: no CNOS baseline yet')
    return
  q = detection_quality(dataset)
  df = df.merge(q, on='key', how='left')
  print(f'\n=== {dataset}: {len(df)} targets ===')

  # 1. Attribution: was this failure the detector's fault or the pose estimator's?
  fail = is_failure(df)
  bad_det = (~df['detected']) | (df['det_iou'] < 0.5)
  att = pd.DataFrame([dict(dataset=dataset, targets=len(df), failures=int(fail.sum()), failure_pct=100 * fail.mean(),
                           failures_from_bad_detection=int((fail & bad_det).sum()),
                           failures_with_good_detection=int((fail & ~bad_det).sum()),
                           share_from_detection_pct=100 * (fail & bad_det).sum() / max(fail.sum(), 1),
                           failure_rate_given_good_detection_pct=100 * fail[~bad_det].mean())])
  att.round(2).to_csv(TABLES / f'attribution_{dataset}.csv', index=False)
  print(att.round(1).T.to_string(header=False))

  # 2. Occlusion, with and without detector failures, so the two effects can be told apart.
  good = df[~bad_det]
  edges, labels = [0, .3, .5, .7, .9, 1.0], ['10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
  occ = pd.DataFrame(binned(df, 'visib_fract', edges, labels))
  occ_good = pd.DataFrame(binned(good, 'visib_fract', edges, labels))
  occ.round(2).to_csv(TABLES / f'occlusion_{dataset}.csv', index=False)
  occ_good.round(2).to_csv(TABLES / f'occlusion_gooddet_{dataset}.csv', index=False)
  print(occ[['bin', 'n', 'ADD-S AUC', 'AR MSSD', 'failure %']].round(1).to_string(index=False))

  # 3. Viewing angle: the camera's elevation in the object's own frame.
  edges = [-90, -60, -30, 0, 30, 60, 90]
  labels = ['-90..-60', '-60..-30', '-30..0', '0..30', '30..60', '60..90']
  view = pd.DataFrame(binned(good, 'elev', edges, labels))
  view.round(2).to_csv(TABLES / f'viewpoint_{dataset}.csv', index=False)
  print(view[['bin', 'n', 'ADD-S AUC', 'AR MSSD', 'failure %']].round(1).to_string(index=False))

  # 4. Objects.
  per = pd.DataFrame([row(g, obj=int(o)) for o, g in df.groupby('obj')]).sort_values('ADD-S AUC')
  per.round(2).to_csv(TABLES / f'objects_{dataset}.csv', index=False)
  print(per[['obj', 'n', 'ADD-S AUC', 'AR MSSD', 'failure %']].round(1).head(6).to_string(index=False))
  return occ, occ_good, view


def corruption_table():
  """Every condition against clean, on the same held-out targets, with paired confidence intervals."""
  test = set(load_splits()['test_subset'])
  base = load('fp_cnos', 'ycbv', 'clean')
  if base is None:
    return None
  base = base[base['key'].isin(test)]
  rows = [row(base, condition='Clean', key='clean', d_adds=0.0, lo=0.0, hi=0.0)]
  for cond, label in CORRUPTION_LABELS.items():
    df = load('fp_cnos', 'ycbv', cond)
    if df is None:
      continue
    df = df[df['key'].isin(test)]
    if len(df) != len(base):
      print(f'{cond}: only {len(df)} of {len(base)} targets, skipped')
      continue
    b = paired_bootstrap(base, df, METRICS['adds_auc'])
    rows.append(row(df, condition=label, key=cond, d_adds=b['diff'], lo=b['lo'], hi=b['hi']))
  out = pd.DataFrame(rows)
  out.round(2).to_csv(TABLES / 'corruptions.csv', index=False)
  print(out[['condition', 'n', 'ADD-S AUC', 'd_adds', 'lo', 'hi', 'AR MSSD', 'failure %']].round(1).to_string(index=False))
  return out


def chart_occlusion(tables):
  fig, ax = plt.subplots(figsize=(7.2, 4.0))
  for ds, label, color in (('ycbv', 'YCB-Video', BLUE), ('lmo', 'LINEMOD-Occluded', ORANGE)):
    if ds not in tables:
      continue
    occ = tables[ds][1]
    ax.plot(occ['bin'], occ['failure %'], color=color, linewidth=2, marker='o', markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    last = occ.iloc[0]
    ax.annotate(label, (0, last['failure %']), xytext=(8, 8), textcoords='offset points', color=INK, fontsize=10)
  ax.set_xlabel('Visible fraction of the object')
  ax.set_ylabel('Failed poses (%)')
  ax.set_ylim(bottom=0)
  ax.legend(frameon=False, loc='upper right')
  ax.set_title('Pose failures rise as objects become more occluded', loc='left', fontsize=13, color=INK)
  fig.text(0.01, 0.005, 'Targets with a good detection only (IoU ≥ 0.5). Failure: worst surface point off by over 10% of the diameter.', fontsize=8.5, color=MUTED)
  fig.tight_layout(rect=(0, 0.03, 1, 1))
  fig.savefig(FIGS / 'occlusion_failures.png')
  plt.close(fig)


def chart_corruptions(table):
  t = table[table['key'] != 'clean'].sort_values('d_adds', ascending=False)
  fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(t) + 1.5))
  y = np.arange(len(t))
  ax.barh(y, t['d_adds'], color=BLUE, height=0.62)
  ax.errorbar(t['d_adds'], y, xerr=[t['d_adds'] - t['lo'], t['hi'] - t['d_adds']], fmt='none', ecolor=INK, elinewidth=1, capsize=2)
  ax.set_yticks(y, t['condition'])
  ax.invert_yaxis()
  ax.grid(axis='y', visible=False)
  ax.axvline(0, color=MUTED, linewidth=1)
  ax.set_xlabel('Change in ADD-S AUC vs clean (points)')
  ax.set_title('What hurts FoundationPose most', loc='left', fontsize=13, color=INK)
  fig.text(0.01, 0.005, f"Same {int(table.iloc[0]['n'])} held-out YCB-V targets in every row. Bars: 95% cluster-bootstrap interval.", fontsize=8.5, color=MUTED)
  fig.tight_layout(rect=(0, 0.03, 1, 1))
  fig.savefig(FIGS / 'corruption_sensitivity.png')
  plt.close(fig)


def main():
  sys.stdout.reconfigure(encoding='utf-8')   # the labels hold a sigma, which the Windows console's default encoding can't print
  sanity()
  tables = {}
  for ds in ('ycbv', 'lmo'):
    out = breakdown(ds)
    if out:
      tables[ds] = out
  if tables:
    chart_occlusion(tables)
  print()
  corr = corruption_table()
  if corr is not None and len(corr) > 1:
    chart_corruptions(corr)


if __name__ == '__main__':
  main()
