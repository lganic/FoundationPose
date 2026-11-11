"""Runs the refiner variants on the dev subset only and prints them side by side.

  python scripts/tune_dev.py --round terms
"""
import argparse, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

ROUNDS = {
  # Which image evidence helps? Each variant turns terms on or off.
  'terms': {
    'dr_depth': ['w_sil=0', 'w_rgb=0'],
    'dr_sil': ['w_depth=0', 'w_rgb=0'],
    'dr_rgb': ['w_depth=0', 'w_sil=0'],
    'dr_depth_sil': ['w_rgb=0'],
    'dr_depth_rgb': ['w_sil=0'],
    'dr_sil_rgb': ['w_depth=0'],
  },
  # Rerun after fixing the color term, which had a free escape: its first results are not valid.
  'terms2': {
    'dr2_rgb': ['w_depth=0', 'w_sil=0'],
    'dr2_sil_rgb': ['w_depth=0'],
    'dr2_depth_rgb': ['w_sil=0'],
    'dr2_depth_rgb_low': ['w_sil=0', 'w_rgb=0.1'],
    'dr2_all': [],
    'dr2_depth_s3_30': ['w_sil=0', 'w_rgb=0', 'sigma_depth=0.003', 'steps=30'],
  },
  # Depth won the round above, so only its own settings are swept. With 150 targets, small
  # differences are noise: the rule is to keep the defaults unless a variant clearly wins.
  'depth': {
    'dr_d_steps30': ['w_sil=0', 'w_rgb=0', 'steps=30'],
    'dr_d_steps120': ['w_sil=0', 'w_rgb=0', 'steps=120'],
    'dr_d_sigma3': ['w_sil=0', 'w_rgb=0', 'sigma_depth=0.003'],
    'dr_d_sigma10': ['w_sil=0', 'w_rgb=0', 'sigma_depth=0.01'],
    'dr_d_gate10': ['w_sil=0', 'w_rgb=0', 'occ_gate=0.01'],
    'dr_d_gate40': ['w_sil=0', 'w_rgb=0', 'occ_gate=0.04'],
  },
}


def run(cmd):
  subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--round', default='terms', choices=sorted(ROUNDS))
  p.add_argument('--condition', default='clean')
  a = p.parse_args()
  names = []
  for name, overrides in ROUNDS[a.round].items():
    tag = name if a.condition == 'clean' else f'{name}'
    run([PY, 'scripts/run_refine.py', '--method', 'dr', '--name', tag, '--subset', 'dev_subset', '--condition', a.condition, '--set'] + overrides)
    run([PY, 'scripts/evaluate.py', '--tag', tag, '--condition', a.condition, '--no_vsd'])
    names.append(tag)
    print('done', tag, flush=True)
  subprocess.run([PY, 'scripts/compare.py', '--base', 'fp_cnos', '--methods'] + names + ['--subset', 'dev_subset', '--condition', a.condition], cwd=REPO, check=True)


if __name__ == '__main__':
  main()
