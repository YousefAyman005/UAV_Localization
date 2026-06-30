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

**Table 5.5 — Retrieval, Recall@{1,5,10} (%) and prior-conditioned R@1 (spatial test
band, $N=1270$).**

| Model | R@1 | R@5 | R@10 | R@1 (≤5 km) | R@1 (≤1 km) |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | 6.4 | 20.3 | 29.5 | 6.6 | 9.6 |
| SigLIP2 B/16-384 | 9.4 | 27.3 | 37.9 | 9.8 | 13.4 |
| CAMP (University-1652) | 17.6 | 39.5 | 50.6 | 18.0 | 21.7 |
| Sample4Geo (University-1652) | 19.8 | 42.0 | 53.1 | 20.6 | 24.9 |
| CLIP-L/14 + LoRA (image-only) | 33.3 | 63.9 | 73.7 | 33.9 | 38.0 |
| CLIP-L/14 + LoRA (v5) | **37.1** | **66.9** | **76.5** | **37.5** | **40.1** |
| SigLIP2 + LoRA | 21.4 | 50.7 | 64.2 | 22.0 | 26.3 |

Among the zero-shot encoders, the two University-1652 cross-view models score highest at
Recall@1 (CAMP 17.6 %, Sample4Geo 19.8 %), above SigLIP2 (9.4 %) and CLIP ViT-L/14
(6.4 %). LoRA fine-tuning is the decisive lever: it lifts CLIP ViT-L/14 to 33.3 %
(image-only training control) and 37.1 % for the v5 caption-round adapter — past both
cross-view models — and SigLIP2 to 21.4 %. Adding caption text at query time does not
help: across the fusion-weight sweep (Table 5.6) image-only retrieval ($\alpha=1$) gives
the best Recall@1 and Recall@10, so the v5 figures above are for the image-only query.

<!-- Table 5.5 from analyze/plot_matcher_figures.py over cache/retrieval_v5_input
     (N=1270 = the full 25% spatial test band; the earlier N=318 was a DOUBLE test-band
     split, fixed in _load_retrieval). v5 row = compare/ adapter (2026-06-17), image-only
     query (alpha=1.0), evaluated WITH --north-up (the adapter is trained north_up=True).
     Fusion (alpha<1) does not beat image-only — see Table 5.6 — so the v5-fused row was
     dropped. The generic/geo encoders (MobileCLIP, DINOv2, GeoCLIP, SatCLIP) on the
     9-flight protocol are still pending. -->

Two patterns stand out. First, retrieval is a much coarser localiser than matching: the
best Recall@1 (${\approx}37$ %) is well below the matchers' ${\approx}60$ % A@25m, and a
retrieval hit only places the query in a $1024$-px gallery tile, with no metric pose and
no geometric verification. Embedding retrieval is therefore better read as a coarse prior
or a recovery stage than as a metric solution. Second, adapting to the task matters far
more than the choice of backbone: the two University-1652 cross-view models roughly double
the generic zero-shot CLIP/SigLIP encoders (18–20 % vs 6.4–9.4 % R@1) because they were
trained on the drone↔satellite cross-view problem, but LoRA fine-tuning on the target
flights is the larger lever still, lifting CLIP from 6.4 % to 37.1 % R@1 — past even the
cross-view-pretrained models. The caption text, by contrast, is not a useful query-time
signal: across the fusion-weight sweep (Table 5.6) image-only retrieval is best on
Recall@1 and Recall@10, so fusing the VLM caption does not deliver the view-invariant
bridge it was intended to be. Finally, conditioning on the GPS prior consistently raises
Recall@1 (e.g. 37.1 → 40.1 % within 1 km), confirming that the noisy prior is informative
enough to prune distant confusers from the gallery.

Fusion does not improve over image-only retrieval. Sweeping the image weight $\alpha$
finely from $0.7$ to $1.0$ (Table 5.6, and the v5 fusion-weight figure) shows recall
rising as the image share increases and then flattening across $\alpha\in[0.8,1.0]$.
Image-only retrieval ($\alpha=1.0$) sits at the top of this plateau — it gives the best
Recall@1 ($37.1$ %) and Recall@10 ($76.5$ %) — while the intermediate weights gain at most
${\approx}0.5$–$1$ pp on Recall@{3,5} (e.g. R@3 $58.3$ % at $\alpha=0.85$ vs $56.6$ % at
$\alpha=1.0$). Adding the caption therefore yields no net benefit: the curves are flat over
the plateau and image-only is best on the headline metrics, confirming the caption is at
most a negligible re-ranking signal rather than a cross-view cue. (An earlier reading that
fusion helped at $\alpha{\approx}0.8$ was an artefact of evaluating on too small a band; see
the note below.)

**Table 5.6 — v5 image+text fusion: Recall@{1,3,5,10} (%) vs fusion weight $\alpha$
(spatial test band, $N=1270$).**

| $\alpha$ | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| 0.70 | 31.7 | 52.8 | 63.1 | 73.2 |
| 0.75 | 34.5 | 56.8 | 64.9 | 75.6 |
| 0.80 | 36.6 | 57.8 | 66.4 | 76.4 |
| 0.85 | 36.8 | **58.3** | **67.2** | 76.3 |
| 0.90 | 37.0 | 57.4 | 67.1 | 76.3 |
| 0.95 | 36.9 | 57.2 | **67.2** | 76.1 |
| 1.00 | **37.1** | 56.6 | 66.9 | **76.5** |

<!-- Table 5.6 + retrieval_v5_alpha_sweep.{pdf,png} from analyze/plot_v5_alpha_sweep.py over
     results/v5_sweep_387204_northup/ (job 387204). N=1270 is the full 25% spatial test
     band: the fusion pipeline already restricts to it (clip_fusion_pipeline.py:268), so the
     CSVs ARE the test band. The earlier N=318 was a DOUBLE test-band split (the plot code
     re-applied test_band_mask) ~ 6.25% of data; fixed in plot_v5_alpha_sweep.py by not
     re-splitting. On the correct band fusion no longer beats image-only (R@1 max at
     alpha=1.0). REPRODUCIBILITY: v5 is trained north_up=True, so the fusion run MUST pass
     --north-up. -->

## 5.4 Limitations

<!-- Limitations: rewritten from scratch 2026-06-20. Blunt, structured, most-serious only. -->

These numbers bound localization quality under a controlled benchmark, not field
deployment. The most serious limitations:

1. **Simulated GPS prior.** The search patch is centred on the true position plus a
   zero-mean isotropic Gaussian offset ($\sigma = 80$ m); real GPS error is biased,
   heavy-tailed, and temporally correlated. The search factor ($1.75$) is sized so the true
   position falls inside the patch for 99.7 % of queries *under this prior* — a larger or
   skewed real-world error lowers that containment and caps achievable accuracy for every
   matcher.

2. **Ground-truth labels set an error floor.** The "true" position is the dataset's onboard
   GPS, itself uncertain, and several flights carry a matcher-independent label bias: an
   along-track offset (gimbal pitch / GPS lag) on flights 01/03/04/05/11 and a world-fixed
   ${\sim}11$ m northward bias on flight 08, with flight 02 as the clean control (two
   independent matchers agree to within $1$–$2$ m). No method can score below this floor,
   so part of the per-flight spread and the ${\sim}21$ m median error is label error, not
   matching.

3. **Calibration is circular.** The per-flight ground-sampling distance and per-flight-leg
   yaw offsets are fit from matcher residuals on these same flights and then evaluated on
   them. Held-out validation bounds the yaw error (median ${\leq}2.6^\circ$ per flight),
   but the benchmark does not test calibration-free operation and a new dataset would need
   re-calibration.

4. **One reference basemap.** Every flight is matched against a single satellite image from
   one source and date. The measured drone↔satellite gap is the gap to *that* basemap;
   robustness to a different season, age, resolution, or provider is untested.

5. **Narrow, correlated sample; no significance testing.** Nine flights from one dataset
   (UAV-VisLoc; 07/09 excluded), one platform class, few sites. Consecutive frames overlap
   heavily, so $N \approx 5058$ images cover far fewer independent scenes, and no confidence
   intervals are reported — the ${<}1$ pp spread among the top matchers is within noise.
   Generalization to other regions, altitudes, land-cover, or cameras is untested.

6. **2D position only.** The shared 4-DOF similarity estimator assumes a planar scene and
   recovers a map position, not camera pose or height. On the tilted flights (05, 11) and
   in high relief this is an approximation, not a model.

7. **Partial benchmark; text fusion failed.** MATCHA was not run and four retrieval
   encoders (MobileCLIP, DINOv2, GeoCLIP, SatCLIP) are still pending, so neither table is
   the full field. Retrieval is scored on the held-out spatial test band ($N = 1270$), not
   the matchers' full set, so the ${\sim}37$ % R@1 vs ${\sim}60$ % A@25m comparison is not
   over identical queries. Fusing VLM captions at query time gave no net benefit over
   image-only retrieval — the intended view-invariant bridge is a negative result.

8. **Measured, not deployed.** Match times are on datacentre-class GPUs; embedded UAV
   feasibility (power, memory, latency) is not evaluated, so the §5.2 recommendation is a
   relative-cost statement only. Every figure is computed over hand-cleaned localizable
   queries (§5.1) — it describes accuracy *given* a usable query, not end-to-end coverage of
   a raw flight.
