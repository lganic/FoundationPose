"""Builds the blinded A/B page for human judgments of pose alignment.

Ground truth is itself off by millimeters, so it cannot settle which of two close poses fits the
image better. A person looking at the picture can. The page shows two outlines of one object and
asks which hugs it better.

Blinding: the page holds only random pair ids and images. Which method is on which side lives in
key.json, which is never published. Pairs are chosen by how much the two poses differ on screen,
never by their error against ground truth, so selection cannot favor either method.

  python scripts/make_labeling.py --method dr_final --subset test_subset
"""
import argparse, base64, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpose.data import REPO, SceneCache, load_splits, load_targets, target_key

MIN_PX, MAX_PX = 2.0, 60.0    # below this nobody can see a difference; above it one pose is simply wrong
WIDTH = 420                   # displayed image width

PAGE = r"""<title>Outline Fit Check</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* Layout: one judgment per screen, two equal panels, three answers. Neutral gray, so that the
   surroundings do not tint how the photo's colors and edges are perceived. */
:root { --bg:#26272a; --panel:#1c1d1f; --line:#3d3f44; --fg:#f1f1ee; --muted:#b4b5b0; --accent:#ffd84d; --accent-ink:#1c1d1f; color-scheme:dark;
        --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace; }
body { background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.5; }
main { max-width:1000px; margin:0 auto; padding-block:20px 32px; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); display:grid; gap:16px; }
h1 { font-size:22px; margin:0; font-weight:600; text-wrap:balance; }
p { margin:0; color:var(--muted); max-width:68ch; }
.top { display:flex; justify-content:space-between; align-items:baseline; gap:16px; flex-wrap:wrap; }
.count { font-family:var(--mono); font-variant-numeric:tabular-nums; color:var(--muted); }
.bar { height:4px; background:var(--line); border-radius:2px; overflow:hidden; }
.bar i { display:block; height:100%; width:0; background:var(--accent); }
.panels { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
figure { margin:0; background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; min-width:0; }
figure img { display:block; width:100%; height:auto; }
figcaption { padding:6px 10px; font-family:var(--mono); color:var(--muted); }
.answers { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
button { font:inherit; color:var(--fg); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px 10px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
button.picked { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
kbd { font-family:var(--mono); font-size:12px; color:inherit; opacity:.75; margin-left:6px; }
.tools { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.tools button { padding:8px 12px; }
.status { color:var(--muted); font-size:13px; }
.done { display:grid; gap:12px; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:20px; }
textarea { width:100%; min-height:120px; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; font-family:var(--mono); font-size:12px; }
@media (max-width:640px) { .panels { grid-template-columns:1fr; } .answers { grid-template-columns:1fr; } }
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
  <div class="top"><h1>Which outline fits the object better?</h1><span class="count" id="count"></span></div>
  <p>Look at where each yellow outline sits on the real object's edges. Ignore any part hidden behind another object. If you honestly can't see a difference, choose Can't tell. There is no right answer to hunt for.</p>
  <div class="bar"><i id="bar"></i></div>
  <section id="task">
    <div class="panels">
      <figure><img id="imgA" alt="Option A: the object with a yellow outline"><figcaption>A</figcaption></figure>
      <figure><img id="imgB" alt="Option B: the object with a yellow outline"><figcaption>B</figcaption></figure>
    </div>
  </section>
  <div class="answers" id="answers">
    <button id="pickA" data-choice="A">A fits better<kbd>1</kbd></button>
    <button id="pickT" data-choice="tie">Can't tell<kbd>2</kbd></button>
    <button id="pickB" data-choice="B">B fits better<kbd>3</kbd></button>
  </div>
  <div class="tools">
    <button id="back">Previous<kbd>←</kbd></button>
    <button id="peek">Hold to hide outlines<kbd>H</kbd></button>
    <span class="status" id="status"></span>
  </div>
  <section class="done" id="done" hidden>
    <h1>All done. Thank you.</h1>
    <p id="doneText"></p>
    <div id="manual" hidden>
      <p>Your answers could not be saved automatically. Copy this text and paste it into the chat.</p>
      <textarea id="dump" readonly></textarea>
      <button id="copy">Copy answers</button>
    </div>
  </section>
</main>
<script>
const PAIRS = __PAIRS__;
const answers = {};
let i = 0, db = null, rater = 'local', hiding = false;
const $ = id => document.getElementById(id);

function show() {
  const finished = i >= PAIRS.length;
  $('done').hidden = !finished;
  ['task', 'answers'].forEach(id => $(id).hidden = finished);
  $('peek').hidden = finished;
  const n = Object.keys(answers).length;
  $('count').textContent = `${Math.min(i + 1, PAIRS.length)} of ${PAIRS.length}`;
  $('bar').style.width = `${100 * n / PAIRS.length}%`;
  $('back').disabled = i === 0;
  if (finished) {
    $('doneText').textContent = `You judged ${n} pairs.` + (db ? ' Your answers are saved.' : '');
    if (!db) { $('manual').hidden = false; $('dump').value = JSON.stringify(answers); }
    return;
  }
  const p = PAIRS[i];
  $('imgA').src = hiding ? p.clean : p.a;
  $('imgB').src = hiding ? p.clean : p.b;
  document.querySelectorAll('#answers button').forEach(b => b.classList.toggle('picked', answers[p.id] === b.dataset.choice));
}

async function pick(choice) {
  if (i >= PAIRS.length) return;
  const p = PAIRS[i];
  answers[p.id] = choice;
  try { localStorage.setItem('answers', JSON.stringify(answers)); } catch (e) {}
  i += 1;
  show();
  if (db) {
    try {
      await db.doc(`ratings/${rater}_${p.id}`).set({ pair: p.id, choice, rater, at: Date.now() });
      $('status').textContent = 'Saved';
    } catch (e) {
      db = null;
      $('status').textContent = "Saving stopped working. Your answers are kept here and you can copy them at the end.";
    }
  }
}

document.querySelectorAll('#answers button').forEach(b => b.addEventListener('click', () => pick(b.dataset.choice)));
$('back').addEventListener('click', () => { if (i > 0) { i -= 1; show(); } });
const hide = on => { hiding = on; show(); };
$('peek').addEventListener('pointerdown', () => hide(true));
['pointerup', 'pointerleave', 'pointercancel'].forEach(e => $('peek').addEventListener(e, () => hide(false)));
addEventListener('keydown', e => {
  if (e.repeat) return;
  if (e.key === '1') pick('A');
  if (e.key === '2') pick('tie');
  if (e.key === '3') pick('B');
  if (e.key === 'ArrowLeft' && i > 0) { i -= 1; show(); }
  if (e.key === 'h' || e.key === 'H') hide(true);
});
addEventListener('keyup', e => { if (e.key === 'h' || e.key === 'H') hide(false); });
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('dump').value); $('copy').textContent = 'Copied'; }
  catch (e) { $('dump').select(); }
});

try { Object.assign(answers, JSON.parse(localStorage.getItem('answers') || '{}')); } catch (e) {}
i = PAIRS.findIndex(p => !(p.id in answers));
if (i < 0) i = PAIRS.length;
show();

(async () => {
  try {
    const [d, u] = await Promise.all([claude.use('db'), claude.use('user')]);
    if (!d) return;
    rater = u ? await u.id() : 'anon';
    const snap = await d.collection('ratings').where('rater', '==', rater).get();
    snap.docs.forEach(doc => { const r = doc.data(); answers[r.pair] = r.choice; });
    db = d;
    const next = PAIRS.findIndex(p => !(p.id in answers));
    i = next < 0 ? PAIRS.length : next;
    show();
  } catch (e) {
    $('status').textContent = "Answers can't be saved automatically here. You can copy them at the end.";
  }
})();
</script>
"""


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--method', required=True, help='refined results folder under results/raw')
  p.add_argument('--base', default='fp_cnos')
  p.add_argument('--subset', default='test_subset')
  p.add_argument('--n_main', type=int, default=40)
  p.add_argument('--n_gt', type=int, default=10)
  p.add_argument('--out', default='results/labeling')
  a = p.parse_args()

  import cv2
  import numpy as np
  import nvdiffrast.torch as dr
  import torch
  from Utils import make_mesh_tensors, nvdiffrast_render
  from bop_toolkit_lib import misc
  from drpose.metrics import Evaluator

  ev = Evaluator('ycbv', use_vsd=False)
  keys = set(load_splits()[a.subset])
  targets = {target_key(t): t for t in load_targets('ycbv') if target_key(t) in keys}

  def read(tag):
    with open(REPO / 'results' / 'raw' / tag / 'ycbv' / 'clean.jsonl') as f:
      rows = [json.loads(line) for line in f if line.strip()]
    return {r['key']: np.array(r['pose']) for r in rows if r['pose'] is not None and r['key'] in keys}

  base, refined = read(a.base), read(a.method)
  gt = {k: t['gt_pose'] for k, t in targets.items()}

  def screen_gap(k, P, Q):
    t = targets[k]
    pts, K = ev.pts(t['obj']), ev.K(t)
    a_ = misc.project_pts(pts, K, P[:3, :3], (P[:3, 3] * 1e3).reshape(3, 1))
    b_ = misc.project_pts(pts, K, Q[:3, :3], (Q[:3, 3] * 1e3).reshape(3, 1))
    return float(np.linalg.norm(a_ - b_, axis=1).max())

  rng = random.Random(20261106)

  def pick(kind, first, second, n, used):
    pool = [k for k in sorted(set(first) & set(second)) if k not in used and MIN_PX <= screen_gap(k, first[k], second[k]) <= MAX_PX]
    by_obj = {}
    for k in pool:
      by_obj.setdefault(targets[k]['obj'], []).append(k)
    for v in by_obj.values():
      rng.shuffle(v)
    out = []
    while len(out) < min(n, len(pool)):
      for o in sorted(by_obj):
        if by_obj[o] and len(out) < n:
          out.append(by_obj[o].pop())
    used.update(out)
    print(f'{kind}: {len(pool)} eligible, {len(out)} chosen', flush=True)
    return [(kind, k, first[k], second[k]) for k in out]

  used = set()
  jobs = pick('refined_vs_base', refined, base, a.n_main, used)
  jobs += pick('gt_vs_base', gt, base, a.n_gt, used)
  jobs += pick('gt_vs_refined', gt, refined, a.n_gt, used)
  rng.shuffle(jobs)

  cache = SceneCache()
  glctx = dr.RasterizeCudaContext()
  meshes = {}

  def silhouette(obj, scene, pose, K, H, W):
    if obj not in meshes:
      meshes[obj] = make_mesh_tensors(cache.mesh('ycbv', scene, obj))
    _, depth, _ = nvdiffrast_render(K=K, H=H, W=W, glctx=glctx, mesh_tensors=meshes[obj],
                                    ob_in_cams=torch.as_tensor(pose[None], device='cuda', dtype=torch.float))
    return depth[0].cpu().numpy() > 1e-3

  def encode(img):
    ok, buf = cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 78])
    assert ok
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

  pairs, key = [], {}
  for kind, k, P, Q in jobs:
    t = targets[k]
    rgb, depth, _, K = cache.load(t, 'cnos')
    H, W = depth.shape
    s1, s2 = silhouette(t['obj'], t['scene'], P, K, H, W), silhouette(t['obj'], t['scene'], Q, K, H, W)
    vs, us = np.nonzero(s1 | s2)
    pad = 30
    y0, y1, x0, x1 = max(vs.min() - pad, 0), min(vs.max() + pad, H), max(us.min() - pad, 0), min(us.max() + pad, W)
    scale = WIDTH / (x1 - x0)
    size = (WIDTH, int(round((y1 - y0) * scale)))
    photo = cv2.resize(rgb[y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC)

    def outlined(sil):
      m = cv2.resize(sil[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
      contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      out = photo.copy()
      cv2.drawContours(out, contours, -1, (0, 0, 0), 3, cv2.LINE_AA)
      cv2.drawContours(out, contours, -1, (255, 235, 0), 2, cv2.LINE_8)
      return out

    pid = f'{rng.getrandbits(40):010x}'
    first_on_a = rng.random() < 0.5
    a_img, b_img = (outlined(s1), outlined(s2)) if first_on_a else (outlined(s2), outlined(s1))
    pairs.append(dict(id=pid, a=encode(a_img), b=encode(b_img), clean=encode(photo)))
    names = kind.split('_vs_')
    key[pid] = dict(kind=kind, target=k, A=names[0] if first_on_a else names[1], B=names[1] if first_on_a else names[0])

  out = REPO / a.out
  out.mkdir(parents=True, exist_ok=True)
  (out / 'key.json').write_text(json.dumps(key, indent=1))
  html = PAGE.replace('__PAIRS__', json.dumps(pairs))
  (out / 'outline_fit_check.html').write_text(html, encoding='utf-8')
  print(f'{len(pairs)} pairs, page {len(html) / 1e6:.1f} MB, written to {out}')


if __name__ == '__main__':
  main()
