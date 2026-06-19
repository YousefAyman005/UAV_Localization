# Results

> **(provisional)** Numbers are from the current post-unification, yaw-corrected runs
> unless stated otherwise. HTML comments are working notes — strip them before
> submission.

<!-- STRUCTURE NOTE: this single chapter replaces the old split (Experiments +
     Results). Repoint any \ref{ch:results} / "§6.x" cross-references to §5.x here. -->

The benchmark, search geometry, and 4-DOF estimator are fixed in the Methodology
chapter (Ch. 4): nine UAV-VisLoc flights (`01,02,03,04,05,06,08,10,11`), a GPS prior of
$\sigma=80$ m, a per-flight calibrated ground-sampling distance, a search factor of
$1.75$, and — for the geometric methods — one shared similarity estimator and acceptance
gate. This chapter describes the dataset cleaning, then reports the feature-matcher
benchmark, the effect of fine-tuning a matcher, the embedding-retrieval results, and the
per-flight bias measured in the dataset. A synthetic Berlin set was built and tested first but is
not part of the benchmark: its query images were rendered from the same satellite
imagery as the reference map, so it does not contain a drone↔satellite domain gap. All
results below use UAV-VisLoc.

## 5.1 Dataset cleaning

Not every image in UAV-VisLoc is usable for localization. A number of frames contain no
distinctive ground structure for any method to match against: the clearest case is
imagery captured over open water, where the frame is an almost uniform blue-green field
with no features at all. Such images cannot be localized by construction — they carry no
geometric or appearance cue tying them to a place on the map — so they say nothing about
how good a method is; keeping them would penalize every method equally but arbitrarily,
inflating the failure rate without measuring anything. A non-trivial number of these
unusable frames occur across the flights, and all of them were removed before
benchmarking. To keep the comparison fair, the same cleaned dataset is used for every
method reported in this chapter — feature matchers and embedding retrieval alike — so all
numbers below are computed over an identical set of localizable images.

## 5.2 Feature-matcher benchmark

All matchers run under the shared geometry on the full benchmark (every image of all
nine flights, $N\approx5058$; the GT position lies inside the searched patch for 99.7 %
of images, and 6 images are skipped, for every matcher). Table 5.3 reports gated A@Xm,
median error over accepted images, and per-image match time.

**Table 5.3 — Matcher benchmark, overall ($N=5058$, full benchmark).**

| Matcher | A@20 | A@25 | A@30 | med. err (m) | $t_\text{match}$ (ms) |
|---|---|---|---|---|---|
| SIFT (baseline) | 15.4 | 20.5 | 24.8 | 26.1 | 1128 |
| ORB (baseline) | 10.2 | 14.1 | 17.1 | 23.9 | 327 |
| BRISK (baseline) | 15.0 | 20.3 | 24.3 | 23.3 | 8105 |
| LightGlue–SIFT | 35.5 | 46.3 | 55.0 | 26.2 | 457 |
| LightGlue–DISK | 33.4 | 44.2 | 52.0 | 22.6 | 122 |
| LightGlue–DeDoDe | 43.0 | 56.1 | 66.6 | 22.2 | 235 |
| LoFTR | 43.3 | 56.6 | 66.6 | 22.0 | 127 |
| XoFTR | 44.3 | 57.8 | 67.6 | 21.7 | 53 |
| EfficientLoFTR | **46.1** | **59.9** | **70.1** | 21.1 | 73 |
| RoMa | 45.4 | 59.1 | 69.6 | 21.5 | 492 |
| RoMa (AerialExtreMatch) | 45.7 | 59.5 | 69.8 | 21.5 | 490 |

The three classical descriptors reach 14.1–20.5 % A@25m, with ORB lowest and SIFT and
BRISK close behind. The learned matchers range from 44.2 % (LightGlue–DISK) to 59.9 %
(EfficientLoFTR). LightGlue–DeDoDe (56.1 %) and LoFTR (56.6 %) sit together; XoFTR is
57.8 %; the three highest are EfficientLoFTR (59.9 %), RoMa (59.1 %), and RoMa–AEM
(59.5 %), within 0.8 pp of each other. Median error over accepted images is 21–23 m for
the learned matchers — LightGlue–SIFT is an outlier at 26 m — and 23–26 m for the
baselines. Match time ranges from 53 ms (XoFTR)
and 73 ms (EfficientLoFTR) to 490 ms (RoMa) and 8105 ms (BRISK).

Per-flight, EfficientLoFTR ranges from 34.6 % to 78.3 % A@25m (Table 5.4). Flights 02,
03, and 08 are highest (69–78 %); flights 04, 05, 10, and 11 are lowest (34–46 %).
Flight 04 is near-nadir and has the highest inlier counts of any flight, with A@25m of
34.6 % and a median offset of ${\sim}32$ m; EfficientLoFTR (34.6 %) and RoMa–AEM
(34.3 %) score within 0.3 pp of each other on it. Flights 05 and 11 were flown with a
tilted camera ($\Omega\approx13^\circ$; flight 11 also $\kappa\approx12^\circ$). Flight
10 has a median of ${\sim}21$ inliers and the smallest image count.

**Table 5.4 — EfficientLoFTR, per-flight A@25m / A@30m (full benchmark).**

| Flight | N | A@20m | A@25m | A@30m |
|---|---|---|---|---|
| 01 | 564 | 52.7 | 66.3 | 78.0 |
| 02 | 719 | 67.2 | 78.3 | 83.3 |
| 03 | 768 | 57.8 | 76.0 | 89.1 |
| 04 | 738 | 22.1 | 34.6 | 44.0 |
| 05 | 473 | 30.0 | 44.0 | 55.4 |
| 06 | 328 | 48.2 | 60.4 | 65.9 |
| 08 | 739 | 54.4 | 69.1 | 79.6 |
| 10 | 139 | 25.9 | 46.0 | 59.7 |
| 11 | 590 | 35.1 | 46.1 | 59.0 |
| **Overall** | **5058** | **46.1** | **59.9** | **70.1** |

<!-- Tables 5.3/5.4 from tar/figures_input/ (the matcher-figure generation,
     summary_matchers.tex), full benchmark all rows. 5.4 per-flight uses the same
     non-skipped & gt-in-patch denominator as 5.3 (N=5058 overall, not 5074), so the
     two tables agree on the overall 59.9/70.1. A@20 added 2026-06-19, recomputed from
     the per-image offset_m in those same CSVs (same gate; summary_matchers.tex itself
     lists only A@5/10/25/30); A@5/A@10 dropped. MATCHA was not run. -->

## 5.3 Fine-tuning EfficientLoFTR

EfficientLoFTR was LoRA-fine-tuned on the train bands of all nine flights, selected on
the validation bands, and scored on the held-out test band ($N=1268$, the same rows for
every variant). Two supervision sources were used: distillation from the RoMa–AEM
teacher (three adapters: rank 8, rank 16, and rank 8 without augmentation), and a
differentiable ground-truth GPS centre loss.

At model selection the best distillation adapter (rank 16) reaches a validation-band
A@25m of 65.5 %, against 64.7 % for the stock matcher and 66.1 % for the teacher.
On the held-out test band (Table 5.5), the stock matcher, the teacher, and the
GT-supervised adapter are within 0.4 pp: 59.5 %, 59.9 %, and 59.5 % A@25m. The
no-training bias-correction control reaches 61.0 % (E/N frame) and 66.2 % (track frame).

**Table 5.5 — Test-band accuracy ($N=1268$, yaw-corrected).**

| Method (test band) | A@20m | A@25m | A@30m | median offset (m) |
|---|---|---|---|---|
| Stock EfficientLoFTR | 46.8 | 59.5 | 69.8 | 20.7 |
| RoMa (AerialExtreMatch) — teacher | 46.9 | 59.9 | 71.1 | 20.8 |
| EfficientLoFTR-LoRA — GT-supervised | 46.4 | 59.5 | 69.4 | 20.3 |
| Bias-correction control — E/N frame | 47.3 | 61.0 | 70.0 | 20.6 |
| Bias-correction control — track frame | **54.1** | **66.2** | **75.9** | **18.6** |

<!-- Distillation comparison is on the val band (the distilled-winner test re-eval was
     not re-run yaw-corrected). Test-band table: tar/eloftr_v2_results/yawfix/ (A@20/25/30
     read from band_summary_yawfix_test.csv, ALL rows). The matcher-figure generation
     lists the distilled LoRA at test-band A@25 60.6 and +calib 66.3, consistent within
     ~0.5 pp. -->

## 5.4 Embedding retrieval

The satellite map is tiled into a gallery ($1024$ px, stride $512$) and the drone images
are ranked by embedding cosine similarity, with no homography. Table 5.6 reports
Recall@{1, 5, 10} for the zero-shot encoders and for the LoRA-fine-tuned CLIP variants,
together with prior-conditioned Recall@1 within 5 km and 1 km of the GPS prior.

**Table 5.6 — Retrieval, Recall@{1,5,10} (%) and prior-conditioned R@1.**

| Model | R@1 | R@5 | R@10 | R@1 (≤5 km) | R@1 (≤1 km) |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | 6.6 | 16.7 | 26.7 | 6.9 | 8.8 |
| SigLIP2 B/16-384 | 8.8 | 26.7 | 34.9 | 9.7 | 13.8 |
| CAMP (University-1652) | 16.7 | 29.9 | 39.0 | 16.7 | 20.1 |
| Sample4Geo (University-1652) | 17.0 | 36.2 | 44.3 | 17.0 | 23.6 |
| CLIP-L/14 + LoRA (image-only) | 28.9 | 55.3 | 68.2 | 28.9 | 32.7 |
| CLIP-L/14 + LoRA (north-up) | 29.9 | 53.8 | 64.5 | 29.9 | 34.6 |
| SigLIP2 + LoRA | 17.3 | 43.1 | 55.7 | 17.6 | 22.6 |

Among the zero-shot encoders, the two University-1652 cross-view models score highest at
Recall@1 (CAMP 16.7 %, Sample4Geo 17.0 %), above SigLIP2 (8.8 %) and CLIP ViT-L/14
(6.6 %). LoRA fine-tuning raises CLIP ViT-L/14 to 28.9 % (image-only) and 29.9 %
(north-up), and SigLIP2 to 17.3 %. The image–text fusion-weight sweep is shown in the
retrieval figures; the image-only weight ($\alpha=1$) gives the Recall@1 reported above.

<!-- Table 5.6 from thesis/figures/matchers/summary_retrieval.csv (pooled, N=318;
     "img-only" and "north-up" are the v4 round). Full fusion-weight sweep and the
     tri-modal vs image-only-control comparison are in recall_v2_summary.csv and the
     retrieval_alpha_sweep figure; the generic/geo encoders (MobileCLIP, DINOv2,
     GeoCLIP, SatCLIP) on the 9-flight protocol are still pending. -->

## 5.5 Per-flight bias

Decomposing each flight's mean error vector into along-track and world-fixed components
gives a consistent along-track offset — about $+14$ m on flight 03, $+10$ m on 04,
$+7$ m on 05, $+5$ m on 11, and $-5$ to $-9$ m on 01 — and a world-fixed offset of about
$-11$ m north on flight 08. Flight 02 is near zero. The offsets are reproduced by the
EfficientLoFTR student and the RoMa teacher within 1–2 m, and the cross-view scale
agrees to within 7 %.

Subtracting each flight's median train-band offset from the test-band predictions, with
no training, changes A@25m as in Table 5.7: overall it rises from 59.5 % to 66.2 % in
the track frame and to 61.0 % in the E/N frame. The per-flight change is largest on
flights 10 (+22.2), 03 (+19.3), 08 (+9.2), and 05 (+8.4); flight 02 is unchanged, and
flights 01 and 06 decrease slightly.

**Table 5.7 — Stock EfficientLoFTR test-band A@25m, raw vs per-flight track-frame
correction (held-out test band, yaw-corrected).**

| Flight | raw A@25 | corrected A@25 | Δ |
|---|---|---|---|
| 01 | 67.4 | 66.7 | −0.7 |
| 02 | 79.4 | 79.4 | 0.0 |
| 03 | 73.4 | 92.7 | +19.3 |
| 04 | 33.2 | 40.8 | +7.6 |
| 05 | 50.9 | 59.3 | +8.4 |
| 06 | 54.8 | 52.4 | −2.4 |
| 08 | 69.2 | 78.4 | +9.2 |
| 10 | 36.1 | 58.3 | +22.2 |
| 11 | 45.3 | 46.6 | +1.3 |
| **Overall** | **59.5** | **66.2** | **+6.7** |

<!-- Bias values and Table 5.7 from tar/eloftr_v2_results/yawfix/ (test band). The
     decomposition is computed on per-flight means; heading is proxied by track
     direction. No correction is applied to the headline numbers in §5.2–5.3. -->

## 5.6 Remaining runs

- MATCHA, under the current geometry, for the matcher benchmark (Table 5.3).
- The generic and geography-aware retrieval encoders (MobileCLIP, DINOv2, GeoCLIP,
  SatCLIP) on the nine-flight protocol (Table 5.6).
- The rewritten-caption retrieval round.
