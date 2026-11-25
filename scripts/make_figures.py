"""Slide figures, drawn from the saved tables so they can be regenerated without any GPU work.

Writes to results/figures. Titles live on the slides, so the charts carry none.

  python scripts/make_figures.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from drpose.data import REPO

TAB, FIGS = REPO / 'results/tables', REPO / 'results/figures'
INK, MUTED, GRID, SURFACE = '#14202b', '#52514e', '#e3e4e0', '#fbfbf8'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7, 'axes.axisbelow': True, 'xtick.major.size': 0, 'ytick.major.size': 0,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
DPI = 400   # a 1px slide unit is 1/200 in here, so the file is twice the size it is shown at
PX = 1 / 200

LABELS = {'clean': 'Clean', 'occlusion_25': 'Occlusion 25%', 'occlusion_50': 'Occlusion 50%', 'occlusion_75': 'Occlusion 75%',
          'motion_blur_11': 'Motion blur 11 px', 'motion_blur_21': 'Motion blur 21 px', 'rgb_noise_25': 'RGB noise 25', 'rgb_noise_50': 'RGB noise 50',
          'depth_noise_10': 'Depth noise 10 mm', 'depth_noise_20': 'Depth noise 20 mm', 'depth_holes_40': 'Depth holes 40%', 'depth_holes_60': 'Depth holes 60%',
          'mask_erode_7': 'Mask eroded 7 px', 'mask_dilate_7': 'Mask dilated 7 px', 'mask_box': 'Box instead of mask'}


def occlusion():
  """Failure rate by visibility: every target against those CNOS found, on one shared scale."""
  fig, axes = plt.subplots(1, 2, figsize=(1664 * PX, 560 * PX), sharey=True)
  for ax, name, ds in zip(axes, ('YCB-Video', 'LINEMOD-Occluded'), ('ycbv', 'lmo')):
    every = pd.read_csv(TAB / f'occlusion_{ds}.csv')
    found = pd.read_csv(TAB / f'occlusion_gooddet_{ds}.csv')
    x = {b: i for i, b in enumerate(every['bin'])}
    ax.plot([x[b] for b in every['bin']], every['failure %'], color=BLUE, lw=1.6, marker='o', ms=6.5, mec=SURFACE, mew=1.2, label='All targets')
    ax.plot([x[b] for b in found['bin']], found['failure %'], color=ORANGE, lw=1.6, marker='o', ms=6.5, mec=SURFACE, mew=1.2, label='Good detections only')
    ax.set_xticks(list(x.values()), [b.replace('-', '–') for b in x])
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.tick_params(axis='x', pad=7)
    ax.set_title(name, loc='left', fontsize=12, fontweight='bold', color=INK, pad=8)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Share of the object that is visible')
  axes[0].set_ylabel('Failed poses (%)')
  axes[0].legend(frameon=False, loc='upper right')
  fig.tight_layout(w_pad=2.5)
  fig.savefig(FIGS / 'slide_occlusion.png', dpi=DPI)
  plt.close(fig)


def corruptions():
  """FoundationPose's failure rate under each corruption, worst first."""
  d = pd.read_csv(TAB / 'failures_by_condition.csv').sort_values('fp_failure_pct')
  fig, ax = plt.subplots(figsize=(1040 * PX, 700 * PX))
  y = range(len(d))
  ax.barh(list(y), d['fp_failure_pct'], height=0.56, color=[MUTED if c == 'clean' else BLUE for c in d['condition']])
  for i, v in zip(y, d['fp_failure_pct']):
    ax.text(v + 0.8, i, f'{v:.1f}', va='center', fontsize=10.5, color=INK)
  ax.set_yticks(list(y), [LABELS[c] for c in d['condition']], color=INK)
  ax.set_xlim(0, 60)
  ax.set_xlabel('Failed poses (%), same 300 targets in every row')
  ax.grid(axis='y', visible=False)
  fig.tight_layout()
  fig.savefig(FIGS / 'slide_corruptions.png', dpi=DPI)
  plt.close(fig)


def reversal():
  """The refiner's change under each condition: the planned metric beside the outline check."""
  d = pd.read_csv(TAB / 'corruption_refine.csv')
  d = d[d['method'] == 'dr_final'].set_index('condition').loc[list(LABELS)[::-1]]
  fig, axes = plt.subplots(1, 2, figsize=(1664 * PX, 690 * PX), sharey=True)
  y = list(range(len(d)))
  for ax, col, xlabel in ((axes[0], 'd_adds_auc', 'ADD-S change (right is better)'), (axes[1], 'edge_fit', 'Edge distance change, px (right is worse)')):
    v, lo, hi = d[col], d[f'{col}_lo'], d[f'{col}_hi']
    clear = (lo > 0) | (hi < 0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.hlines(y, lo, hi, color=ORANGE, lw=1.4)
    ax.scatter(v[clear], [i for i, c in zip(y, clear) if c], s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=3, label='Interval excludes zero')
    ax.scatter(v[~clear], [i for i, c in zip(y, clear) if not c], s=42, facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.4, zorder=3, label='Interval includes zero')
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
  axes[0].set_yticks(y, [LABELS[c] for c in d.index], color=INK)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, frameon=False, loc='upper right', ncol=2)
  fig.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.94))
  fig.savefig(FIGS / 'slide_reversal.png', dpi=DPI)
  plt.close(fig)

def fixed_broken():
  """Failures the refiner fixed against good poses it broke, per condition, worst condition on top."""
  d = pd.read_csv(TAB / 'failures_by_condition.csv').sort_values('fp_failures')
  fig, ax = plt.subplots(figsize=(1664 * PX, 720 * PX))
  y = list(range(len(d)))
  ax.barh(y, d['fixed'], height=0.56, color=BLUE, label='Fixed: a failure became a success')
  ax.barh(y, -d['broken'], height=0.56, color=ORANGE, label='Broken: a success became a failure')
  for i, (f, b) in enumerate(zip(d['fixed'], d['broken'])):
    ax.text(f + 0.5, i, str(f), va='center', fontsize=10.5, color=INK)
    if b:
      ax.text(-b - 0.5, i, str(b), va='center', ha='right', fontsize=10.5, color=INK)
  ax.axvline(0, color=MUTED, lw=1)
  ax.set_yticks(y, [f"{LABELS[c]}  ({n} failures)" for c, n in zip(d['condition'], d['fp_failures'])], color=INK)
  ax.set_xlim(-32, 16)
  ax.set_xticks([-30, -20, -10, 0, 10], ['30', '20', '10', '0', '10'])
  ax.set_xlabel('Targets, out of the same 300 in every row')
  ax.grid(axis='y', visible=False)
  ax.legend(frameon=False, loc='lower left', fontsize=10)
  fig.tight_layout()
  fig.savefig(FIGS / 'report_fixed_broken.png', dpi=DPI)
  plt.close(fig)


def before_after():
  """Each target's error before and after refinement: many small nudges along the diagonal, few crossings of the failure line."""
  import numpy as np
  from drpose.data import load_splits
  splits = load_splits()
  test_scenes = set(splits['test_scenes'])
  # Keys are 'ycbv/scene/im/obj'. Reading test_all off the keys avoids loading every BOP scene file.
  subset_keys = lambda name: set(splits[name]) if name != 'test_all' else None
  panels = (('clean', 'test_all', 'Clean, all 3,000 test targets'), ('occlusion_50', 'test_subset', 'Occlusion 50%, 300 targets'),
            ('occlusion_75', 'test_subset', 'Occlusion 75%, 300 targets'), ('depth_noise_20', 'test_subset', 'Depth noise 20 mm, 300 targets'))
  lo, hi, line = 0.004, 3.0, 0.10
  fig, axes = plt.subplots(2, 2, figsize=(1664 * PX, 1500 * PX), sharex=True, sharey=True)
  for ax, (cond, subset, title) in zip(axes.flat, panels):
    keys = subset_keys(subset)
    def ev(tag):
      d = pd.read_csv(REPO / 'results/raw/eval' / tag / 'ycbv' / f'{cond}.csv')
      d = d[d['key'].isin(keys)] if keys is not None else d[d['scene'].isin(test_scenes)]
      return d.sort_values('key').reset_index(drop=True)
    a, b = ev('fp_cnos'), ev('dr_final')
    assert len(a) == (3000 if keys is None else 300), (cond, len(a))
    assert (a['key'] == b['key']).all()
    x, y = (a['mssd'] / a['diameter']).to_numpy(), (b['mssd'] / b['diameter']).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)       # a missed detection has no pose to plot
    x, y = np.clip(x[ok], lo, hi), np.clip(y[ok], lo, hi)
    fixed, broken = (x > line) & (y <= line), (x <= line) & (y > line)
    rest = ~(fixed | broken)
    ax.fill_between([lo, hi], line, hi, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.fill_betweenx([lo, hi], line, hi, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=1)
    ax.scatter(x[rest], y[rest], s=7, color=MUTED, alpha=0.35, lw=0, zorder=2, label='Same side of the failure line')
    ax.scatter(x[fixed], y[fixed], s=34, color=BLUE, edgecolor=SURFACE, lw=1, zorder=3, label='Fixed: failure became a success')
    ax.scatter(x[broken], y[broken], s=34, color=ORANGE, edgecolor=SURFACE, lw=1, zorder=3, label='Broken: success became a failure')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', color=INK, pad=8)
    better = (y < x).mean()
    ax.text(0.03, 0.97, f'{100 * better:.0f}% improved\n{100 * (1 - better):.0f}% got worse or stayed', transform=ax.transAxes, va='top', fontsize=10, color=INK)
    ax.text(0.97, 0.03, f'Fixed {fixed.sum()}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10.5, color=INK, fontweight='bold')
    ax.text(0.03, 0.62, f'Broken {broken.sum()}', transform=ax.transAxes, va='top', fontsize=10.5, color=INK, fontweight='bold')
  ticks = [0.01, 0.03, 0.1, 0.3, 1]
  for ax in axes.flat:
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
    ax.set_yticks(ticks, [f'{t:g}' for t in ticks])
    ax.minorticks_off()
  for ax in axes[1]:
    ax.set_xlabel('FoundationPose error (MSSD / diameter)')
  for ax in axes[:, 0]:
    ax.set_ylabel('After refinement')
  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, frameon=False, loc='upper center', ncol=3, fontsize=10, markerscale=1.3)
  fig.tight_layout(h_pad=2, w_pad=1.5, rect=(0, 0, 1, 0.95))
  fig.savefig(FIGS / 'report_before_after.png', dpi=DPI)
  plt.close(fig)


def error_change():
  """How far each pose moved toward or away from the truth, per condition: a histogram of signed change in millimetres."""
  import numpy as np
  from drpose.data import load_splits
  splits = load_splits()
  test_scenes = set(splits['test_scenes'])
  panels = (('clean', None, 'Clean, all 3,000 test targets'), ('occlusion_50', 'test_subset', 'Occlusion 50%, 300 targets'),
            ('occlusion_75', 'test_subset', 'Occlusion 75%, 300 targets'), ('depth_noise_20', 'test_subset', 'Depth noise 20 mm, 300 targets'))
  lim, width = 12, 0.5
  edges = np.arange(-lim, lim + width, width)
  fig, axes = plt.subplots(len(panels), 1, figsize=(1664 * PX, 1300 * PX), sharex=True)
  for ax, (cond, subset, title) in zip(axes, panels):
    def ev(tag):
      d = pd.read_csv(REPO / 'results/raw/eval' / tag / 'ycbv' / f'{cond}.csv')
      d = d[d['key'].isin(set(splits[subset]))] if subset else d[d['scene'].isin(test_scenes)]
      return d.sort_values('key').reset_index(drop=True)
    a, b = ev('fp_cnos'), ev('dr_final')
    assert (a['key'] == b['key']).all() and len(a) == (300 if subset else 3000)
    gain = (a['mssd'] - b['mssd']).to_numpy()          # positive: the refined pose is closer to the truth
    gain = gain[np.isfinite(gain)]
    same = (gain == 0).mean() * 100                    # poses the refiner skipped because FoundationPose gave up on them
    counts, _ = np.histogram(np.clip(gain[gain != 0], -lim + 1e-6, lim - 1e-6), edges)
    share = counts / len(gain) * 100
    centers = edges[:-1] + width / 2
    ax.bar(centers, share, width=width * 0.86, color=[BLUE if c > 0 else ORANGE for c in centers], lw=0)
    ax.axvline(0, color=INK, lw=1)
    fa, fb = (a['mssd'] / a['diameter'] > 0.1).to_numpy(), (b['mssd'] / b['diameter'] > 0.1).to_numpy()
    better, worse = (gain > 0).mean() * 100, (gain < 0).mean() * 100
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', color=INK, pad=6)
    top = share.max()
    ax.set_ylim(0, top * 1.35)
    ax.text(lim - 0.2, top * 1.2, f'{better:.0f}% closer to the truth', ha='right', va='top', fontsize=10.5, color=INK)
    ax.text(-lim + 0.2, top * 1.2, f'{worse:.0f}% further away', ha='left', va='top', fontsize=10.5, color=INK)
    if same >= 0.5:
      ax.text(-lim + 0.2, top * 0.95, f'{same:.0f}% unchanged (not refined)', ha='left', va='top', fontsize=9.5, color=MUTED)
    ax.text(lim - 0.2, top * 0.95, f'median {np.median(gain):+.1f} mm  ·  failures fixed {int((fa & ~fb).sum())}, broken {int((~fa & fb).sum())}',
            ha='right', va='top', fontsize=9.5, color=MUTED)
    ax.set_ylabel('Targets (%)')
    ax.grid(axis='x', visible=False)
  ticks = list(range(-lim, lim + 1, 3))
  axes[-1].set_xticks(ticks, [f'≤{t}' if t == -lim else f'≥+{t}' if t == lim else f'{t:+d}' if t else '0' for t in ticks])
  axes[-1].set_xlabel('How much closer the refined pose is to the truth (mm, worst surface point)   ←  worse   ·   better  →')
  fig.tight_layout(h_pad=1.4)
  fig.savefig(FIGS / 'report_error_change.png', dpi=DPI)
  plt.close(fig)


def refiner_by_mask():
  """The refiner's AR change on each dataset, with real detector masks and with the GT-mask oracle."""
  from drpose.data import subset_keys
  from drpose.stats import METRICS, paired_bootstrap
  ev = lambda tag, ds: pd.read_csv(REPO / 'results/raw/eval' / tag / ds / 'clean.csv')
  keep = subset_keys('test_subset')
  rows = []
  for ds, name, keys in (('ycbv', 'YCB-Video, 300 test targets', keep), ('lmo', 'LINEMOD-Occluded, 1,445 targets', None)):
    for base, ref, masks in (('fp_cnos', 'dr_final', 'CNOS masks (real detector)'), ('fp', 'dr_final_gtmask', 'GT masks (oracle)')):
      a, b = ev(base, ds), ev(ref, ds)
      if keys is not None:
        a, b = a[a['key'].isin(keys)], b[b['key'].isin(keys)]
      r = paired_bootstrap(a, b, METRICS['ar'], n_boot=1000)
      rows.append(dict(dataset=name, masks=masks, **{k: round(v, 2) for k, v in r.items() if k != 'n_clusters'}))
  d = pd.DataFrame(rows)
  d.to_csv(TAB / 'refiner_by_mask.csv', index=False)
  print(d.to_string(index=False))

  fig, ax = plt.subplots(figsize=(1664 * PX, 470 * PX))
  y = [3, 2, 0.8, -0.2][:len(d)]
  oracle = d['masks'].str.startswith('GT').to_numpy()
  ax.axvline(0, color=MUTED, lw=1)
  ax.hlines(y, d['lo'], d['hi'], color=ORANGE, lw=1.4)
  ax.scatter(d['diff'][~oracle], [v for v, o in zip(y, oracle) if not o], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=3, label='CNOS masks (real detector)')
  ax.scatter(d['diff'][oracle], [v for v, o in zip(y, oracle) if o], s=46, facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.4, zorder=3, label='GT masks (oracle)')
  for v, (_, r) in zip(y, d.iterrows()):
    ax.text(r['hi'] + 0.12, v, f"{r['diff']:+.2f}", va='center', fontsize=10.5, color=INK)
  ax.set_yticks(y, [f"{r['dataset'].split(',')[0]}, {'GT' if r['masks'].startswith('GT') else 'CNOS'}" for _, r in d.iterrows()], color=INK)
  ax.set_xlabel('Change in BOP AR from FoundationPose with the same masks (right is better)')
  ax.grid(axis='y', visible=False)
  ax.legend(frameon=False, loc='lower right', fontsize=10)
  fig.tight_layout()
  fig.savefig(FIGS / 'report_refiner_by_mask.png', dpi=DPI)
  plt.close(fig)


if __name__ == '__main__':
  occlusion()
  corruptions()
  reversal()
  fixed_broken()
  before_after()
  error_change()
  refiner_by_mask()
