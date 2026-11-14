"""Runs chains of run_fp.py jobs in parallel lanes. Jobs inside a chain run in order, because they
share an output file. Everything resumes, so this can be stopped and started again at any time.

  python scripts/run_queue.py --lanes 2 --plan clean
"""
import argparse, queue, subprocess, sys, threading, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]


def job(dataset, condition, subset=None, mask='cnos', tag='fp_cnos'):
  cmd = ['--dataset', dataset, '--condition', condition, '--mask', mask, '--tag', tag]
  return cmd + (['--subset', subset] if subset else [])


SWEEP = ['occlusion_50', 'motion_blur_11', 'rgb_noise_25', 'depth_noise_10', 'depth_holes_40', 'mask_dilate_7',
         'occlusion_25', 'occlusion_75', 'motion_blur_21', 'rgb_noise_50', 'depth_noise_20', 'depth_holes_60', 'mask_erode_7', 'mask_box']
DEV_CONDITIONS = SWEEP[:6]

PLANS = {
  'clean': [
    [job('ycbv', 'clean', 'dev_subset'), job('ycbv', 'clean', 'test_subset')],
    [job('ycbv', 'clean', 'test_subset', mask='gt', tag='fp')],
    [job('lmo', 'clean')],
  ],
  # One chain per condition: mid-severity conditions also run on dev, so the refiner can be tuned
  # on corrupted frames without touching the test scenes.
  'corrupt': [
    [job('ycbv', c, 'dev_subset'), job('ycbv', c, 'test_subset')] if c in DEV_CONDITIONS else [job('ycbv', c, 'test_subset')]
    for c in SWEEP
  ],
  'full': [[job('ycbv', 'clean')]],
  'lmo': [[job('lmo', 'clean')]],
  # Corrupted dev frames first: the refiner cannot be frozen, and test frames cannot be touched, until these exist.
  'dev': [[job('ycbv', c, 'dev_subset')] for c in ['depth_noise_10', 'depth_holes_40', 'occlusion_50', 'rgb_noise_25', 'motion_blur_11', 'mask_dilate_7']],
}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--lanes', type=int, default=2)
  p.add_argument('--plan', default='clean', choices=sorted(PLANS))
  a = p.parse_args()
  log_dir = REPO / 'results' / 'raw' / 'logs'
  log_dir.mkdir(parents=True, exist_ok=True)
  q = queue.Queue()
  for chain in PLANS[a.plan]:
    q.put(chain)
  failed = []

  def lane(i):
    while True:
      try:
        chain = q.get_nowait()
      except queue.Empty:
        return
      for cmd in chain:
        name = '_'.join(x for x in cmd if not x.startswith('--'))
        t0 = time.time()
        with open(log_dir / f'{name}.txt', 'a') as f:
          code = subprocess.call([sys.executable, str(REPO / 'scripts' / 'run_fp.py')] + cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
        print(f'[lane {i}] {name}: exit {code}, {(time.time() - t0) / 60:.1f} min', flush=True)
        if code:
          failed.append(name)
          break

  threads = [threading.Thread(target=lane, args=(i,)) for i in range(a.lanes)]
  [t.start() for t in threads]
  [t.join() for t in threads]
  print('FAILED: ' + ', '.join(failed) if failed else 'all jobs finished', flush=True)
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
