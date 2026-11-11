# Can differentiable rendering improve FoundationPose?

Track: Differentiable Rendering. Group 4: Jonathan Nazario, David Tran. Deeper analysis: Sensitivity.

## Summary

A depth-only differentiable render-and-compare step added after FoundationPose gives a small gain on YCB-Video, a small loss on LINEMOD-Occluded, and fits the photo worse according to a blinded rater.

| Check | Result |
|---|---|
| YCB-V held-out, 3,000 targets | AR +1.05 [+0.27, +1.81], 84.88 to 85.93 |
| LM-O, 1,445 targets, never tuned on | AR -0.83 [-1.12, -0.56], 67.41 to 66.58 |
| Failure points | Rarely rescued: 5 fixed and 2 broken of 326 clean failures |
| Heavy depth noise (20 mm) | Harmful: ADD-S -0.57 [-0.72, -0.40], 28 poses broken |
| Blinded human votes on YCB-V | Refined 7, FoundationPose 26, ties 7 (p = 0.0013, one rater) |

Brackets are 95% paired cluster-bootstrap intervals throughout. The cause of the LM-O loss and of the human reversal is not established. Of six explanations tested, mask quality is the best lead for LM-O: with GT masks the refiner gains AR +1.04 [+0.62, +1.41] there, though those masks partly carry the answer (see [Exploratory follow-ups](#exploratory-follow-ups)).

The planned metric gain must not be quoted alone. The proposal's stated risk, that FoundationPose already performs well enough for rendering not to help, largely came true.

## Setup

| Part | Choice |
|---|---|
| Model | FoundationPose, released weights, 252 hypotheses in the code (42 views x 6 in-plane; the paper describes 504) |
| Datasets | `ycbv_test_bop19` (900 images, 4,123 targets), `lmo_test_bop19` (200 images, 1,445 targets) |
| Detector | CNOS default detections. BOP's GT masks are used only as a labeled oracle |
| Metrics | BOP AR with the official toolkit (VSD, MSSD, MSPD), ADD-S and ADD AUC |
| Failure | Worst surface point off by more than 10% of the object's diameter (`mssd / diameter > 0.10`) |
| Split | YCB-V dev scenes 50, 58, 59; test scenes 48, 49, 51-57; `dev_subset` 150 targets, `test_subset` 300 targets, `test_all` 3,000 targets |
| Statistics | Paired cluster bootstrap. Clusters are scene x object on YCB-V, object x 100-frame block on LM-O |
| Hardware | One RTX 3070 (8 GB), native Windows, Python 3.10, torch 2.9.1 + CUDA 13.0, nvdiffrast 0.4.0 |

The refiner's configuration was frozen in `results/frozen_config.json` before test data was opened. Work added after seeing the labels is marked exploratory.

Deviations from the proposal: the Dockerfile was rewritten but never built, because this machine has no Docker. Everything ran on a native Windows build (`scripts/setup_windows.md`).

## Baseline against published numbers

The paper's only model-based figures are BOP AR with Mask R-CNN detections.

| Dataset | Masks | Targets | AR | VSD | MSSD | MSPD | Published AR |
|---|---|---|---|---|---|---|---|
| YCB-V | CNOS | 4,123 | 86.2 | 83.5 | 90.0 | 85.2 | 88.0 |
| YCB-V | GT (oracle) | 450 | 91.8 | 87.4 | 96.3 | 91.6 | 88.0 |
| LM-O | CNOS | 1,445 | 67.4 | 56.2 | 71.5 | 74.6 | 78.8 |
| LM-O | GT (oracle) | 1,445 | 81.7 | 67.7 | 86.6 | 90.7 | 78.8 |

YCB-V lands 1.8 below the paper, which is plausible given CNOS in place of Mask R-CNN, half the hypotheses and the released weights, which exclude the paper's diffusion-augmented data.

LM-O's CNOS run landed 11.4 below, large enough to suspect a scoring bug, so all of LM-O was rerun with GT masks. AR rose by +14.28 [+11.28, +17.12] to 81.7, above the paper. On both datasets the published figure sits between the CNOS run and the GT-mask run, so the evaluation looks sound and the LM-O gap comes from masks. Mask R-CNN was not run here, so the split between detector, hypothesis count and weights is not measured. Even with GT masks, LM-O fails more than YCB-V: 23.0% of poses against 3.3%, with a median translation error of 6.7 mm.

## Where FoundationPose fails

Detection causes most failures. A bad detection means no detection or mask IoU under 0.5.

| Dataset | Failures | From bad detection | AR, good detection | AR, bad detection |
|---|---|---|---|---|
| YCB-V | 392 of 4,123 (9.5%) | 272 (69%) | 92.3 (3,714 targets) | 31.2 (409 targets) |
| LM-O | 511 of 1,445 (35.4%) | 307 (60%) | 85.1 (1,102 targets) | 10.6 (343 targets) |

Good-detection targets are also easier targets, so these splits are suggestive and are not a causal estimate.

**Occlusion.** Failures by visible fraction (`results/tables/occlusion_*.csv`, `results/figures/slide_occlusion.png`):

| Visible | YCB-V | LM-O |
|---|---|---|
| 10-30% | 100.0% | 95.5% |
| 30-50% | 42.1% | 71.2% |
| 50-70% | 17.1% | 42.2% |
| 70-90% | 15.6% | 23.7% |
| 90-100% | 5.0% | 26.6% |

**Viewing angle** matters much less. With good detections, failures by camera elevation stay between 0 and 5.2% on YCB-V and between 13.2 and 21.1% on LM-O.

**Objects.** The YCB-V clamps (objects 19 and 20) and object 7 fail 33-54% of the time with CNOS and 0% with GT masks, on the 13 to 15 targets of each that have both. These are detection problems.

**Added corruptions**, on the same 300 held-out YCB-V targets (`results/tables/failures_by_condition.csv`):

| Condition | Failures | | Condition | Failures |
|---|---|---|---|---|
| Occlusion 75% | 53.0% | | Clean | 10.7% |
| Occlusion 50% | 25.0% | | RGB noise 25, 50 | 10.3% |
| Motion blur 21 px | 19.7% | | Mask eroded 7 px | 10.0% |
| Depth noise 20 mm | 16.7% | | Mask dilated 7 px | 9.7% |
| Occlusion 25% | 14.0% | | Depth noise 10 mm | 9.3% |
| Motion blur 11 px | 11.3% | | Box instead of mask | 9.3% |
| Depth holes 60% | 11.0% | | Depth holes 40% | 9.3% |

The worst failure points are therefore 50-75% occlusion, 21 px motion blur and 20 mm depth noise.

## The refiner

`drpose/refine.py` reuses nvdiffrast. Starting from FoundationPose's final pose it renders depth, compares it with sensor depth inside the CNOS mask using a robust cost, backpropagates through the renderer and takes Adam steps on an SE(3) update about the object's center. It keeps the best iterate and rejects the result if the objective did not improve. No step sees ground truth. It adds about 0.3-0.5 s per target to FoundationPose's 2.4-3.2 s.

Frozen settings: depth term only, `sigma_depth=0.003`, 30 steps.

Dev ablation, change in ADD-S on 150 targets:

| Term | Change |
|---|---|
| Depth, frozen settings | +0.18 [+0.07, +0.32] |
| Depth, first settings | +0.16 |
| RGB only | -2.70 |
| Silhouette only | -6.57 (16.5 mm along the viewing ray) |

The first RGB runs (`dr_v0`, `dr_rgb`, `dr_depth_rgb`, `dr_sil_rgb`) are invalid: the color term scored zero once the object left the mask. They were rerun as `dr2_*` after the fix, and `test_every_term_charges_for_leaving_the_mask` guards it.

The occlusion gate does nothing in the frozen configuration. A reading past the 20 mm gate is already past the cost cap, so gated and ungated runs match. The gate only matters when the silhouette term is on, where it cuts error to 0.26-0.51 of the ungated value.

### The silhouette leak

BOP's `mask` and `mask_visib` are rendered at the GT pose, so fitting them copies the answer. Adding the silhouette term to the depth term, on the 300 held-out targets:

| Masks | AR | VSD |
|---|---|---|
| GT (oracle) | +1.52 [+1.03, +2.07] | +3.06 [+2.00, +4.21] |
| CNOS (real) | -0.49 [-1.34, +0.13] | -0.93 [-2.19, +0.08] |

Any silhouette refinement scored against those masks is measuring a leak.

## Held-out results on YCB-V

Change from FoundationPose with CNOS masks. `Refiner` is `dr_final`.

| Method | Targets | AR | VSD | MSSD | MSPD | ADD-S | ADD |
|---|---|---|---|---|---|---|---|
| Refiner | 3,000 | +1.05 [+0.27, +1.81] | +1.43 [+0.18, +2.78] | +0.34 [-0.07, +0.79] | +1.38 [+0.51, +2.15] | +0.25 [+0.14, +0.37] | +0.38 [+0.17, +0.57] |
| Refiner | 300 | +1.20 [+0.21, +2.26] | +1.93 [+0.07, +3.97] | +0.40 [-0.19, +1.00] | +1.27 [+0.20, +2.48] | +0.32 [+0.18, +0.47] | +0.39 [+0.15, +0.65] |
| ICP | 300 | -0.82 [-3.29, +1.32] | +0.32 [-3.03, +3.35] | -1.40 [-2.95, -0.04] | -1.37 [-4.58, +1.25] | +0.02 [-0.42, +0.35] | -0.28 [-0.96, +0.37] |
| More FP iterations | 300 | +0.06 [-0.27, +0.41] | -0.11 [-0.75, +0.59] | +0.10 [-0.32, +0.56] | +0.20 [-0.48, +0.95] | +0.00 [-0.04, +0.05] | -0.04 [-0.37, +0.29] |

Compared directly on the 300 targets, the refiner's AR leads ICP by +2.01 [+0.30, +4.21] and leads extra iterations by +1.14 [+0.18, +2.13]. On the 3,000 targets, 61% improve and 29% get worse.

With GT masks (oracle), the refiner's AR change is +3.14 [+1.94, +4.36] from FoundationPose's 90.47.

### Across the 15 conditions

300 targets per row, from `results/tables/corruption_refine.csv`. Edge is the exploratory distance from image edges, where positive is worse.

| Condition | ADD-S change | MSPD change | Edge change, px | Fixed / broken |
|---|---|---|---|---|
| Clean | +0.32 [+0.18, +0.47] | +1.27 [+0.20, +2.48] | +0.168 [+0.098, +0.244] | 0 / 0 |
| RGB noise 25 | +0.29 [+0.14, +0.46] | +1.50 [+0.52, +2.56] | +0.194 [+0.133, +0.264] | 1 / 0 |
| RGB noise 50 | +0.32 [+0.17, +0.49] | +1.50 [+0.45, +2.59] | +0.151 [+0.082, +0.226] | 1 / 0 |
| Motion blur 11 px | +0.28 [+0.14, +0.42] | +1.73 [+0.67, +2.85] | +0.134 [+0.071, +0.202] | 1 / 0 |
| Motion blur 21 px | +0.37 [+0.23, +0.50] | +2.00 [+1.08, +2.98] | +0.050 [-0.029, +0.124] | 2 / 0 |
| Depth noise 10 mm | +0.16 [-0.01, +0.36] | +0.40 [-0.63, +1.42] | +0.151 [+0.086, +0.222] | 0 / 0 |
| Depth noise 20 mm | -0.57 [-0.72, -0.40] | -5.03 [-6.19, -3.94] | +0.328 [+0.242, +0.417] | 2 / 28 |
| Depth holes 40% | +0.34 [+0.19, +0.50] | +1.20 [+0.03, +2.30] | +0.155 [+0.087, +0.230] | 1 / 0 |
| Depth holes 60% | +0.25 [+0.09, +0.42] | +0.77 [-0.37, +1.92] | +0.192 [+0.125, +0.269] | 1 / 0 |
| Occlusion 25% | +0.22 [+0.10, +0.35] | +1.10 [+0.21, +2.03] | +0.133 [+0.083, +0.191] | 0 / 1 |
| Occlusion 50% | +0.20 [+0.07, +0.31] | +0.70 [-0.07, +1.41] | +0.112 [+0.058, +0.169] | 6 / 2 |
| Occlusion 75% | -0.08 [-0.36, +0.14] | -0.10 [-0.94, +0.62] | +0.130 [+0.073, +0.189] | 11 / 9 |
| Mask eroded 7 px | +0.18 [-0.04, +0.40] | +0.50 [-0.68, +1.66] | +0.219 [+0.154, +0.298] | 0 / 1 |
| Mask dilated 7 px | +0.30 [+0.15, +0.47] | +1.10 [-0.09, +2.22] | +0.230 [+0.166, +0.296] | 1 / 0 |
| Box instead of mask | +0.25 [+0.11, +0.41] | +1.23 [+0.26, +2.20] | +0.220 [+0.160, +0.287] | 1 / 0 |

The ADD-S gain is clear in 11 of 15 conditions, unclear in 3 and negative in 1. The edge distance grows clearly in 14 of 15. ICP's MSSD interval is below zero in 14 of 15. Extra FP iterations stay near zero: 3 of 60 intervals exclude it, about chance.

### Failure points

On all 3,000 clean targets, the refiner fixes 5 of 326 failures and breaks 2. By visibility (`results/tables/failures_by_visibility_test_all.csv`):

| Visible | Targets | Clusters | FP failure | Fixed / broken | ADD-S change | MSPD change |
|---|---|---|---|---|---|---|
| Under 50% | 88 | 5 | 38.6% | 0 / 0 | -0.48 [-0.76, +0.19] | -0.91 [-2.03, +1.88] |
| 50-70% | 219 | 13 | 18.7% | 1 / 0 | +0.32 [+0.03, +0.60] | +2.01 [+0.16, +4.09] |
| 70-90% | 660 | 19 | 17.4% | 0 / 2 | +0.30 [+0.13, +0.49] | +1.73 [+0.24, +2.97] |
| 90-100% | 2,033 | 34 | 6.7% | 4 / 0 | +0.25 [+0.13, +0.38] | +1.29 [+0.36, +2.25] |

The under-50% row is inconclusive at 5 clusters. Overall the refiner polishes poses that are nearly right and does not rescue bad ones, which is what a local search should do. That answers the proposal's open question: it helps overall performance slightly and does not help at the failure points.

## Transfer to LM-O

All 1,445 targets, frozen settings, never tuned on.

| Method | AR | VSD | MSSD | MSPD | ADD-S | ADD |
|---|---|---|---|---|---|---|
| Refiner | -0.83 [-1.12, -0.56] | -2.06 [-2.54, -1.58] | -0.08 [-0.34, +0.17] | -0.35 [-0.57, -0.13] | -0.12 [-0.17, -0.08] | -0.20 [-0.29, -0.11] |
| ICP | -1.51 [-1.99, -1.09] | -2.58 [-3.26, -1.96] | -0.87 [-1.32, -0.48] | -1.09 [-1.54, -0.67] | -0.17 [-0.35, +0.00] | -0.51 [-0.78, -0.28] |
| More FP iterations | +0.14 [+0.05, +0.23] | +0.13 [-0.01, +0.27] | +0.19 [+0.07, +0.31] | +0.11 [+0.01, +0.22] | +0.00 [-0.02, +0.03] | +0.01 [-0.02, +0.04] |

7 of 8 objects dip, 42% of targets improve and 44% get worse, and the refiner fixes 21 failures and breaks 45.

## Human check

Ground truth is itself off by millimeters, so it cannot settle which of two close poses fits the image. A blinded page (`results/labeling/outline_fit_check.html`) showed two outlines on the same photo and asked which fit better, with the side randomized. Pairs were picked by on-screen difference and never by error against ground truth.

| Pair | First wins | Second wins | Ties | Sign test p |
|---|---|---|---|---|
| Refined vs FoundationPose | 7 | 26 | 7 | 0.0013 |
| GT vs FoundationPose | 4 | 5 | 1 | 1.00 |
| GT vs refined | 2 | 5 | 3 | 0.45 |

The refined share is 0.212, exact interval [0.090, 0.389], cluster interval [0.086, 0.367]. There was no position bias: A 26, B 23, tie 11 (p = 0.775). The rater spent 6.3 s per pair, and the preference was 4-10-5 in the first half and 3-16-2 in the second. The answer key was verified 60 of 60 with an independent CPU rasterizer. GT metrics agreed with the rater on 13-16 of the 33 decided refined pairs, which is chance.

One rater and 60 pairs on one dataset is thin evidence. It is enough to stop the metric gain from being reported as a plain improvement. It isn't enough to say the refiner makes poses worse.

## Exploratory follow-ups

Everything here came after the labels and is hypothesis-generating only.

**Edge fit.** How far each outline sits from Canny edges in the photo, with no ground truth involved (`scripts/check_edge_fit.py`).

| Dataset | Refined - FP, px | GT - FP, px | Means: FP, refined, GT |
|---|---|---|---|
| YCB-V, 298 targets | +0.168 [+0.098, +0.244] | +0.212 [+0.114, +0.326] | 2.081, 2.249, 2.293 |
| LM-O, 1,367 targets | +0.063 [+0.047, +0.080] | +0.236 [+0.146, +0.326] | 1.965, 2.028, 2.200 |

On YCB-V the outlines moved farther from edges on 214 targets and closer on 76. The edge check predicted the rater on 27 of 33 decided pairs (p = 0.0003), 28 of 33 on the visible contour, and 21 of 33 (p = 0.16) with looser Canny thresholds. GT itself fits edges worse than FoundationPose on both datasets.

**Six explanations tested:**

| Explanation | Finding | Verdict |
|---|---|---|
| Pixel-center slip in our renderer | Against analytic ray-sphere depth at frozen settings, pull is du -0.001 +/- 0.013 px and dv +0.003 +/- 0.017 px over 40 positions, and a planted 0.5 px shift reads 0.45-0.46 px | Ruled out |
| Sensor shadow holes | Holes are lopsided, but unrelated to motion (rank correlation -0.015) | Ruled out |
| LM-O mask quality | With GT masks the refiner's LM-O change flips to AR +1.04 [+0.62, +1.41], VSD +1.55 [+0.79, +2.29], 57 fixed and 38 broken. CNOS bins alone had suggested otherwise: every visibility and mask-quality bin dips. GT masks are rendered from the true pose, so part of this gain may be leak | Best lead |
| LM-O sensor depth offset | The sensor reads a median 2.47 mm behind GT on LM-O and 0.23 mm on YCB-V, but LM-O's VSD falls about 2 at every gap, including -1.68 [-2.74, -0.78] under 1 mm | Not supported |
| One fixed shift in every pose | Fitted to each dataset's own test motion, it gives YCB-V ADD-S +0.09 [+0.02, +0.15] and LM-O -0.05 [-0.07, -0.03], with no YCB-V MSPD gain. Fitted on test, so these are upper estimates | Partial |
| YCB-V depth and color misregistered | Depth edges sit about 2 px below color in all 12 scenes, but refined poses move up about 0.4 px, and LM-O's offset is 0 | Wrong direction |

On YCB-V the VSD gain does shrink as the sensor-to-GT gap grows (+2.51 under 1 mm, +0.35 at 1-3 mm, -0.57 at 3-6 mm), though the last bin has only 14 targets. Most of the refiner's motion varies from target to target (sd about 1.9 px against a median near 0.4 px).

## Limitations

- One rater, 60 pairs, YCB-V only. The edge check that supports the rater was added after the labels.
- One frozen refiner configuration. Other designs could behave differently.
- 252 hypotheses and released weights, not the paper's setup.
- GT-mask oracle results on YCB-V cover 450 targets, not all 4,123.
- Corruptions are synthetic and cover the 300-target subset.
- The bootstrap has few clusters in small bins, such as 5 under 50% visible.
- The Dockerfile was never built.

Any edge-aware variant is post-hoc. It would need tuning on dev scenes, testing on targets outside `test_subset`, and a new round of labels from raters who haven't seen this result.

## Reproducing

Setup is in `scripts/setup_windows.md`. Run from the repository root with the `.venv` Python.

```bash
python scripts/make_splits.py
python scripts/run_queue.py --lanes 1 --plan full
python scripts/run_refine.py --method dr --name dr_final --source fp_cnos --dataset ycbv --condition clean --subset test_all --set w_sil=0 w_rgb=0 sigma_depth=0.003 steps=30
python scripts/evaluate.py --tag dr_final --dataset ycbv
python scripts/compare.py --base fp_cnos --methods dr_final --subset test_all --metrics adds_auc add_auc ar_vsd ar_mssd ar_mspd ar
python scripts/analyze_baseline.py
python scripts/failure_points.py --subset test_all
python scripts/make_figures.py
python -m pytest tests -q
```

The test suite last ran at 114 passed and 23 skipped, on three runs in a row. The skips are reference comparisons that need `P3D_REF` or `MYCPP_REF`.

Tables are in `results/tables`, figures in `results/figures`. `results/raw` is not tracked.
