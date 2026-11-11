"""Writes the fixed dev/test split. Run once, before any results exist."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import REPO, load_targets, make_splits

out = REPO / 'results' / 'splits' / 'ycbv.json'
if out.exists():
  sys.exit(f'{out} already exists. Regenerating it would invalidate the split, so delete it by hand if you mean to.')
out.parent.mkdir(parents=True, exist_ok=True)
s = make_splits()
out.write_text(json.dumps(s, indent=1))
targets = load_targets('ycbv')
print('dev scenes', s['dev_scenes'], ' test scenes', s['test_scenes'])
for name, scenes in (('dev', s['dev_scenes']), ('test', s['test_scenes'])):
  pool = [t for t in targets if t['scene'] in scenes]
  print(f'{name}: {len(pool)} targets, objects {sorted({t["obj"] for t in pool})}')
for name in ('dev_subset', 'test_subset'):
  c = collections.Counter(k.split('/')[3] for k in s[name])
  print(f'{name}: {len(s[name])} targets over {len(c)} objects, per-object min {min(c.values())} max {max(c.values())}')
