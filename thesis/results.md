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
benchmark, the embedding-retrieval results, and the per-flight bias measured in the
dataset. A synthetic Berlin set was built and tested first but is
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
| SIFT (baseline) | 12.9 | 17.7 | 21.7 | 28.2 | 730 |
| ORB (baseline) | 10.2 | 14.1 | 17.1 | 23.9 | 327 |
| BRISK (baseline) | 9.5 | 12.9 | 15.6 | 23.7 | 1143 |
| LightGlue–SIFT | 35.5 | 46.3 | 55.0 | 26.2 | 457 |
| LightGlue–DISK | 33.4 | 44.2 | 52.0 | 22.6 | 122 |
| LightGlue–DeDoDe | 43.0 | 56.1 | 66.6 | 22.2 | 235 |
| LoFTR | 43.3 | 56.6 | 66.6 | 22.0 | 127 |
| XoFTR | 44.3 | 57.8 | 67.6 | 21.7 | 53 |
| EfficientLoFTR | **46.1** | **59.9** | **70.1** | 21.1 | 73 |
| RoMa | 45.4 | 59.1 | 69.6 | 21.5 | 492 |
| RoMa (AerialExtreMatch) | 45.7 | 59.5 | 69.8 | 21.5 | 490 |

The three classical descriptors reach 12.9–17.7 % A@25m, with BRISK lowest and SIFT
highest. The learned matchers range from 44.2 % (LightGlue–DISK) to 59.9 %
(EfficientLoFTR). LightGlue–DeDoDe (56.1 %) and LoFTR (56.6 %) sit together; XoFTR is
57.8 %; the three highest are EfficientLoFTR (59.9 %), RoMa (59.1 %), and RoMa–AEM
(59.5 %), within 0.8 pp of each other. Median error over accepted images is 21–23 m for
the learned matchers — LightGlue–SIFT is an outlier at 26 m — and 24–28 m for the
baselines. Match time ranges from 53 ms (XoFTR)
and 73 ms (EfficientLoFTR) to 730 ms (SIFT) and 1143 ms (BRISK).

Per-flight, EfficientLoFTR ranges from 34.6 % to 78.3 % A@25m (Table 5.4). Flights 02,
03, and 08 are highest (69–78 %); flights 04, 05, 10, and 11 are lowest (34–46 %).
Flight 04 is near-nadir and has the highest inlier counts of any flight, with A@25m of
34.6 % and a median offset of ${\sim}32$ m; EfficientLoFTR (34.6 %) and RoMa–AEM
(34.3 %) score within 0.3 pp of each other on it. Flights 05 and 11 were flown with a
tilted camera ($\Omega\approx13^\circ$; flight 11 also $\kappa\approx12^\circ$). Flight
10 has a median of ${\sim}21$ inliers and the smallest image count.

**Table 5.4 — EfficientLoFTR, per-flight A@20m / A@25m / A@30m (full benchmark).**

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

These results separate into two regimes. The classical descriptors (13–18 % A@25m) and
the learned matchers (44–60 %) differ by roughly a factor of three, which reflects the
drone↔satellite domain gap rather than raw keypoint quality: hand-crafted descriptors are
not invariant to the change in viewpoint, resolution, and season between the aerial query
and the satellite map, whereas the learned dense and semi-dense matchers recover
correspondences across it. Within the learned group, however, the three strongest methods
fall within 0.8 pp of each other (${\approx}59$–60 % A@25m), and the aerial-specialised
RoMa–AEM does not improve on the generic EfficientLoFTR or RoMa: accuracy has plateaued,
which suggests the ceiling is no longer set by match quality. Flight 04 makes this
concrete — it is near-nadir and yields the highest inlier counts of any flight, yet it has
the lowest accuracy (34.6 % A@25m): high inlier counts with low accuracy show that the
limit on flight 04 is not match quality. The per-flight ranking is otherwise stable and tracks
acquisition geometry — the tilted flights 05 and 11 and the low-relief flight 04 are
hardest for every matcher — not the choice of matcher. For deployment EfficientLoFTR is
the clear pick: it gives the best accuracy at one of the lowest match times (73 ms),
whereas RoMa matches its accuracy at roughly seven times the cost.

## 5.3 Embedding retrieval

The satellite map is tiled into a gallery ($1024$ px, stride $512$) and the drone images
are ranked by embedding cosine similarity, with no homography. Table 5.5 reports
Recall@{1, 5, 10} for the zero-shot encoders and for the LoRA-fine-tuned CLIP variants,
together with prior-conditioned Recall@1 within 5 km and 1 km of the GPS prior.

**Table 5.5 — Retrieval, Recall@{1,5,10} (%) and prior-conditioned R@1.**

| Model | R@1 | R@5 | R@10 | R@1 (≤5 km) | R@1 (≤1 km) |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | 6.6 | 16.7 | 26.7 | 6.9 | 8.8 |
| SigLIP2 B/16-384 | 8.8 | 26.7 | 34.9 | 9.7 | 13.8 |
| CAMP (University-1652) | 16.7 | 29.9 | 39.0 | 16.7 | 20.1 |
| Sample4Geo (University-1652) | 17.0 | 36.2 | 44.3 | 17.0 | 23.6 |
| CLIP-L/14 + LoRA (image-only) | 28.9 | 55.3 | 68.2 | 28.9 | 32.7 |
| CLIP-L/14 + LoRA (v5) | 32.1 | 62.3 | 70.4 | 32.7 | 36.2 |
| CLIP-L/14 + LoRA (v5, fused) | 34.3 | 60.4 | 68.2 | 34.6 | 39.3 |
| SigLIP2 + LoRA | 17.3 | 43.1 | 55.7 | 17.6 | 22.6 |

Among the zero-shot encoders, the two University-1652 cross-view models score highest at
Recall@1 (CAMP 16.7 %, Sample4Geo 17.0 %), above SigLIP2 (8.8 %) and CLIP ViT-L/14
(6.6 %). LoRA fine-tuning of CLIP ViT-L/14 raises Recall@1 from 28.9 % (image-only
control) to 32.1 % for the latest caption-round adapter (v5, image-only query), and
SigLIP2 to 17.3 %. For v5, adding caption text at the fusion-weight peak ($\alpha=0.8$)
lifts Recall@1 further to 34.3 % — the first round in which text fusion improves on
image-only retrieval ($\alpha=1$) rather than degrading it — while image-only keeps the
best Recall@{5,10}. The full image–text fusion-weight sweep is shown in the retrieval
figures.

<!-- Table 5.5 from thesis/figures/matchers/summary_retrieval.csv (pooled, N=318).
     LoRA rows are the v5 caption round (compare/ adapter, 2026-06-17); the two v5 rows
     are alpha=1.0 (image-only query) and alpha=0.8 (fusion peak), kept as separate rows
     so no alpha column is needed; the image-only control was only run at alpha=1.0. Full
     fusion-weight sweep in the retrieval_alpha_sweep figure; the generic/geo encoders
     (MobileCLIP, DINOv2, GeoCLIP, SatCLIP) on the 9-flight protocol are still pending. -->

Two patterns stand out. First, retrieval is a much coarser localiser than matching: the
best Recall@1 (${\approx}34$ %) is well below the matchers' ${\approx}60$ % A@25m, and a
retrieval hit only places the query in a $1024$-px gallery tile, with no metric pose and
no geometric verification. Embedding retrieval is therefore better read as a coarse prior
or a recovery stage than as a metric solution. Second, adapting to the task matters far
more than the choice of backbone: the two University-1652 cross-view models roughly double
the generic zero-shot CLIP/SigLIP encoders (${\approx}17$ % vs 6.6–8.8 % R@1) because they
were trained on the drone↔satellite cross-view problem, but LoRA fine-tuning on the target
flights is the larger lever still, lifting CLIP from 6.6 % to 32.1 % R@1 — past even the
cross-view-pretrained models. The caption text is only a secondary signal: not until the
v5 round does fusing it help at all (34.3 % vs 32.1 % image-only), and then by only
${\approx}2$ pp at a tuned fusion weight, with image-only retrieval still best at
Recall@{5,10}; text is a marginal, conditional gain rather than the decisive view-invariant
bridge it was intended to be. Finally, conditioning on the GPS prior consistently raises
Recall@1 (e.g. 34.3 → 39.3 % within 1 km), confirming that the noisy prior is informative
enough to prune distant confusers from the gallery.

## 5.4 Limitations

<!-- Limitations of the benchmark; drafted 2026-06-19. Flowing prose to match 5.2/5.3. -->

The numbers in this chapter are produced under a controlled benchmark, and several
deliberate simplifications bound what they measure. They are stated here so the results
are not over-read as a claim about end-to-end field deployment.

The task is posed with a *simulated* GPS prior: each query's search patch is centred on
its true position perturbed by an isotropic Gaussian offset of $\sigma = 80$ m, drawn
reproducibly per image. This is a stand-in for onboard GPS uncertainty, not a measurement
of it. Real GPS error is often biased, heavy-tailed, or temporally correlated rather than
zero-mean Gaussian, and its magnitude varies with receiver, multipath, and flight
conditions. Because the search factor ($1.75$) is sized so that the true position falls
inside the patch for 99.7 % of queries *under this prior*, a larger or more skewed
real-world error would lower that containment rate and, with it, the achievable accuracy —
independently of any matcher.

All flights are localized against a single satellite image, captured at one time from one
source. The drone↔satellite domain gap measured here is therefore the gap to that specific
basemap. A deployed system would face reference imagery of a different season, age,
resolution, or provider, which would widen the appearance gap; the benchmark does not probe
robustness to changing the reference map.

The evaluation also covers a narrow slice of conditions: nine flights from a single dataset
(UAV-VisLoc), collected with one class of platform over a limited set of sites, with flights
07 and 09 excluded. Although $N \approx 5058$ images is large, consecutive frames overlap
heavily, so the number of independent scenes is much smaller, and the per-flight results
(Table 5.4) show how strongly accuracy depends on a handful of acquisition geometries.
Generalization to other regions, altitudes, land-cover types, or camera classes is untested
and should not be assumed from these numbers.

All geometric methods share a 4-DOF similarity estimator (translation, rotation, uniform
scale) fitting the drone image to the satellite patch, which assumes a planar scene and a
similarity relation between the two views. This recovers a 2D map position only — not a full
camera pose or height — and treats terrain relief and view tilt as nuisance to be absorbed
rather than modelled. The assumption holds well for the near-nadir, low-relief flights that
dominate the dataset, but on strongly oblique or high-relief scenes (e.g. the tilted flights
05 and 11) it is an approximation, and a full homographic or 3D treatment is left to future
work.

The per-flight ground-sampling distance and the per-flight-leg yaw offsets are calibrated
from matcher residuals on these same flights and then applied identically to every method.
Sharing one calibration across methods keeps the comparison fair, but it is tuned to this
dataset, so the benchmark does not test calibration-free operation and a new dataset would
require re-calibration. The yaw calibration was validated on held-out frames (median
residual $\leq 2.6^\circ$ per flight), which limits but does not fully remove the
circularity of calibrating and evaluating on the same flights.

Three narrower caveats round this out. First, frames with no localizable content (chiefly
open water) were removed before benchmarking (§5.1), so every figure is computed over
localizable queries; the reported accuracies describe localization quality given a usable
query, not end-to-end coverage of a raw flight. Second, the embedding-retrieval results
(§5.3) are a coarse, tile-level localiser with no metric pose or geometric verification,
are scored on a smaller pooled test band ($N = 318$), and the text-conditioned variant
delivered only a marginal, conditional gain rather than the view-invariant bridge it was
intended to provide — that line is best read as a largely negative result. Third, match
times are measured on datacentre-class GPUs; real-time feasibility on embedded UAV hardware
(power, memory, and latency on onboard accelerators) is not evaluated, so the deployment
recommendation in §5.2 is a relative-cost statement, not a demonstration of onboard
real-time operation.
