"""The last dev-only check before freezing: does the chosen refiner survive corrupted input?

Rule set before seeing results: keep the depth-only refiner unless it does significant harm
(a 95% interval entirely below zero on ADD-S AUC or AR-MSSD) under a condition where an
alternative does not.
"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from drpose.data import REPO, load_splits
from drpose.stats import METRICS, paired_bootstrap

PY = sys.executable
CONDITIONS = ['depth_noise_10', 'depth_holes_40', 'occlusion_50', 'rgb_noise_25', 'motion_blur_11', 'mask_dilate_7']
BASE_ARGS = ['w_sil=0', 'w_rgb=0', 'sigma_depth=0.003', 'steps=30']
CANDIDATES = {
  'cand_depth': (BASE_ARGS, CONDITIONS),
  'cand_depth_rgb': (['w_sil=0', 'w_rgb=0.1', 'sigma_depth=0.003', 'steps=30'], CONDITIONS),
  'cand_depth_wide': (['w_sil=0', 'w_rgb=0', 'sigma_depth=0.01', 'steps=30'], ['depth_noise_10']),
}


def run(*cmd):
  subprocess.run([PY, *cmd], cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def table(tag, condition, keys):
  df = pd.read_csv(REPO / 'results' / 'raw' / 'eval' / tag / 'ycbv' / f'{condition}.csv')
  df = df[df['key'].isin(keys)]
  assert len(df) == len(keys), (tag, condition, len(df))
  return df


def main():
  keys = set(load_splits()['dev_subset'])
  for c in CONDITIONS:
    run('scripts/evaluate.py', '--tag', 'fp_cnos', '--condition', c, '--no_vsd')
  for name, (args, conditions) in CANDIDATES.items():
    for c in conditions:
      run('scripts/run_refine.py', '--method', 'dr', '--name', name, '--subset', 'dev_subset', '--condition', c, '--set', *args)
      run('scripts/evaluate.py', '--tag', name, '--condition', c, '--no_vsd')
      print('done', name, c, flush=True)
  rows = []
  for c in CONDITIONS:
    base = table('fp_cnos', c, keys)
    for name, (_, conditions) in CANDIDATES.items():
      if c not in conditions:
        continue
      other = table(name, c, keys)
      r = dict(condition=c, method=name)
      for m in ('adds_auc', 'ar_mssd'):
        b = paired_bootstrap(base, other, METRICS[m], n_boot=1000)
        r[m] = f"{b['diff']:+.2f} [{b['lo']:+.2f},{b['hi']:+.2f}]"
        r[f'harm_{m}'] = b['hi'] < 0
      rows.append(r)
  out = pd.DataFrame(rows)
  out.to_csv(REPO / 'results' / 'tables' / 'dev_corruption_check.csv', index=False)
  print(out.to_string(index=False))


if __name__ == '__main__':
  main()
