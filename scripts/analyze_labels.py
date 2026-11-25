"""Unblinds the human A/B judgments and tests them.

Primary question, set before the labels were read: among pairs where the two outlines visibly
differ, do people prefer the refined pose to the FoundationPose pose? Ties are reported but left
out of the test. The gt_vs_* pairs ask a second question: is ground truth itself a better fit to
the picture than the estimates it is used to score?

  python scripts/analyze_labels.py
"""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from drpose.data import REPO

LAB = REPO / 'results' / 'labeling'
TAB = REPO / 'results' / 'tables'


def load():
  key = json.load(open(LAB / 'key.json'))
  rows = []
  for f in glob.glob(str(LAB / 'raw_ratings' / 'ratings' / '*.json')):
    r = json.load(open(f))
    k = key[r['pair']]
    winner = 'tie' if r['choice'] == 'tie' else k[r['choice']]
    rows.append(dict(pair=r['pair'], rater=r['rater'], at=r['at'], kind=k['kind'], target=k['target'], side_refined=next((s for s in 'AB' if k[s] == 'refined'), None), choice=r['choice'], winner=winner))
  d = pd.DataFrame(rows).sort_values('at').reset_index(drop=True)
  d['rater'] = d.rater.map({r: f'rater{i + 1}' for i, r in enumerate(sorted(d.rater.unique()))})   # account ids stay out of the tables
  parts = d.target.str.split('/', expand=True)
  d['scene'], d['obj'] = parts[1].astype(int), parts[3].astype(int)
  return d


def cluster_share(sub, first, n_boot=10000, seed=0):
  """95% interval for the preference among decided pairs, resampling scene-object groups."""
  g = sub.assign(c=sub.scene.astype(str) + '_' + sub.obj.astype(str)).groupby('c').winner
  wins = g.apply(lambda w: (w == first).sum()).to_numpy()
  dec = g.apply(lambda w: (w != 'tie').sum()).to_numpy()
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, len(wins), (n_boot, len(wins)))
  w, n = wins[idx].sum(1), dec[idx].sum(1)
  share = w[n > 0] / n[n > 0]
  return np.percentile(share, [2.5, 97.5]), len(wins)


def test(d, kind, first, second):
  sub = d[d.kind == kind]
  a, b, t = (sub.winner == first).sum(), (sub.winner == second).sum(), (sub.winner == 'tie').sum()
  n = a + b
  p = binomtest(int(a), int(n), 0.5).pvalue if n else float('nan')
  ci = binomtest(int(a), int(n), 0.5).proportion_ci(0.95, 'exact') if n else None
  lo, hi = (ci.low, ci.high) if n else (float('nan'),) * 2
  (clo, chi), groups = cluster_share(sub, first)
  return dict(comparison=f'{first} vs {second}', pairs=len(sub), first_wins=int(a), second_wins=int(b), ties=int(t), first_share=round(a / n, 3) if n else None,
              exact_lo=round(lo, 3), exact_hi=round(hi, 3), cluster_lo=round(clo, 3), cluster_hi=round(chi, 3), groups=groups, sign_test_p=round(p, 4))


def against_metrics(d):
  """Does the person side with the pose that ground truth calls better?"""
  base = pd.read_csv(REPO / 'results/raw/eval/fp_cnos/ycbv/clean.csv').set_index('key')
  ref = pd.read_csv(REPO / 'results/raw/eval/dr_final/ycbv/clean.csv').set_index('key')
  sub = d[d.kind == 'refined_vs_base'].copy()
  out = []
  for col in ['mspd', 'te', 're', 'adds', 'mssd']:
    gain = base.loc[sub.target, col].to_numpy() - ref.loc[sub.target, col].to_numpy()   # positive means ground truth prefers the refined pose
    sub[f'gain_{col}'] = gain
    dec = sub[sub.winner != 'tie']
    g = dec[f'gain_{col}'].to_numpy()
    human = np.where(dec.winner == 'refined', 1, -1)
    keep = g != 0
    agree = int((np.sign(g[keep]) == human[keep]).sum())
    out.append(dict(metric=col, metric_prefers_refined=int((sub[f'gain_{col}'] > 0).sum()), of=len(sub), decided_pairs=int(keep.sum()), agree=agree, agree_share=round(agree / keep.sum(), 3), sign_test_p=round(binomtest(agree, int(keep.sum()), 0.5).pvalue, 4)))
  return pd.DataFrame(out), sub


def main():
  d = load()
  print(f'{len(d)} judgments, {d.rater.nunique()} rater(s), {(d["at"].max() - d["at"].min()) / 1e3 / len(d):.1f} s per pair')
  print('duplicates:', d.duplicated(['pair', 'rater']).sum())

  res = pd.DataFrame([test(d, 'refined_vs_base', 'refined', 'base'), test(d, 'gt_vs_base', 'gt', 'base'), test(d, 'gt_vs_refined', 'gt', 'refined')])
  print(res.to_string(index=False))

  main_pairs = d[d.kind == 'refined_vs_base']
  dec = main_pairs[main_pairs.winner != 'tie']
  pa = (d.choice == 'A').sum()
  print(f'\nposition check, all pairs: A {pa}, B {(d.choice == "B").sum()}, tie {(d.choice == "tie").sum()}; p={binomtest(int(pa), int((d.choice != "tie").sum()), 0.5).pvalue:.3f}')
  print('refined side won, by side:', dec.groupby('side_refined').winner.apply(lambda w: f'{(w == "refined").sum()}/{len(w)}').to_dict())

  half = len(d) // 2   # does preference drift as the rater tires or adapts?
  early = main_pairs[main_pairs.index < half]
  late = main_pairs[main_pairs.index >= half]
  for name, s in [('first half', early), ('second half', late)]:
    print(f'{name}: refined {(s.winner == "refined").sum()}, base {(s.winner == "base").sum()}, tie {(s.winner == "tie").sum()}')

  agree, sub = against_metrics(d)
  print('\nperson vs ground-truth metrics, refined_vs_base pairs:')
  print(agree.to_string(index=False))

  print('\nby object:')
  print(main_pairs.groupby('obj').winner.value_counts().unstack(fill_value=0).to_string())

  TAB.mkdir(parents=True, exist_ok=True)
  res.to_csv(TAB / 'human_labels.csv', index=False)
  agree.to_csv(TAB / 'human_vs_metrics.csv', index=False)
  d.drop(columns=['at']).to_csv(LAB / 'ratings_unblinded.csv', index=False)


if __name__ == '__main__':
  main()
