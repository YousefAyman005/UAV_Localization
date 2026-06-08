# Methodology

> **Drafting conventions.** Equations and constants in this chapter are taken
> verbatim from the implementation; the source location is cited inline as
> `file:line` (e.g. `helpers/utils.py:72`). Cross-references to method
> background use `\ref{}`-style tags (e.g. `Sec.~\ref{bg:roma}`) so the chapter
> ports mechanically to LaTeX. Literature citations are left as `[CITE: …]`
> placeholders. Sections marked **(provisional)** describe work that may still
> change and should be revisited before submission.

This chapter describes the experimental apparatus built to benchmark UAV→satellite
visual localization. The individual matchers and embedding models are introduced in
the Background chapter; here the focus is on *what was done with them*: the geometric
core that turns image correspondences into a metric ground error, the shared
evaluation harness, the diagnostic visualization, the CLIP-based retrieval line, and
the evaluation protocol and infrastructure that make the results reproducible.

## 3.1 Problem Formulation

Each flight provides a sequence of drone images together with a single
georeferenced satellite map of the overflown area. For a drone image $I_d$ acquired
at true ground position $p^\*=(\phi^\*,\lambda^\*)$ (latitude, longitude), the system
receives a **noisy GPS prior**

$$
\hat p = p^\* + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\sigma^2 I_2),
$$

and must estimate the position $\hat{p}_{\text{est}}$ on the satellite map. Performance
is reported as the great-circle (haversine) distance between estimate and ground
truth, **in meters** — not in pixels — so that results are comparable across flights
with different satellite resolutions.

Two solution paradigms are studied. The **geometric** paradigm establishes
pixel correspondences between the drone image and a satellite patch, fits a
homography, and projects the drone-image centre onto the map to obtain a metric
position. The **retrieval** paradigm embeds the drone image and a grid of satellite
tiles into a common feature space and ranks tiles by similarity, without estimating
any geometry. The two are evaluated on different but complementary protocols
(Sec. 3.7).

## 3.2 Dataset

Experiments use the **UAV-VisLoc** dataset [CITE: UAV-VisLoc]. Each flight folder
contains a georeferenced satellite GeoTIFF, a directory of drone images, and a CSV of
per-image metadata. The relevant drone-CSV columns are `filename, lat, lon, height`
and `Phi1` (the yaw angle, in compass convention, clockwise from north); the satellite
metadata CSV gives each map's top-left (LT) and bottom-right (RB) corner coordinates
(`helpers/utils.py:105`, `helpers/utils.py:129`).

Georeferencing is treated as a linear map between pixel coordinates and geographic
coordinates. From the corner coordinates and the image size $w\times h$, the pixels
per degree are

$$
p_{\text{lat}} = \frac{h}{\phi_{\text{LT}}-\phi_{\text{RB}}}, \qquad
p_{\text{lon}} = \frac{w}{\lambda_{\text{RB}}-\lambda_{\text{LT}}},
$$

(`helpers/utils.py:107`), which underpin all GPS↔pixel conversions in Sec. 3.3.5.

**Flight selection (provisional).** The dataset nominally contains eleven flights.
Two are unusable by the single-tile pipeline: flight **07**, whose satellite image is
a narrow $3000\times170$-pixel strip too small for metric cropping, and flight **09**,
whose satellite coverage is split across four separate tiles. This leaves nine usable
flights, `01, 02, 03, 04, 05, 06, 08, 10, 11` (`helpers/utils.py:57`). A four-flight
subset (`01, 02, 03, 08`) — for which the ground-sampling calibration of Sec. 3.3.4 is
hand-anchored — serves as the primary benchmark set, and the set can be widened to the
full nine when broader coverage is required.
<!-- TODO: confirm the final flight list used for the reported results. -->

## 3.3 The Geo Core: From Correspondences to Meters

The geometric paradigm is realised by a single per-image procedure
(`collect_pipeline_rows_multitile`, `helpers/utils.py:287`). It is matcher-agnostic:
any matcher is plugged in through a *match factory* (Sec. 3.4.1), while the geometric
bookkeeping below is shared.

### 3.3.1 Per-image pipeline

For each drone image the pipeline (`helpers/utils.py:305`):

1. loads the drone image, resizes it to the working resolution
   $W\times H = 1024\times680$ (`SZ_W, SZ_H`, `helpers/utils.py:21`), and applies
   contrast normalization (Sec. 3.4.4);
2. selects the satellite tile containing the GPS prior and locates the prior in
   satellite pixels (`tile_for_gps`, `helpers/utils.py:142`); images whose prior falls
   outside the map are skipped;
3. perturbs that centre with a simulated GPS-prior offset (Sec. 3.3.2);
4. samples a metric-isotropic, heading-aligned satellite patch about the perturbed
   centre (Sec. 3.3.3);
5. runs the matcher to obtain a homography $H$ (Sec. 3.4);
6. projects the drone-image centre $(W/2, H/2)=(512,340)$ through $H$, converts the
   result to geographic coordinates, and measures the haversine error against the
   ground truth (Sec. 3.3.5).

### 3.3.2 Simulating a noisy GPS prior

The dataset's recorded GPS is effectively clean, so a controlled prior noise is
injected to model a realistic localization scenario and to ensure the drone is **not**
at the trivial dead-centre of the satellite patch. A two-dimensional Gaussian offset

$$
\varepsilon=(\Delta x,\Delta y)\sim\mathcal{N}(0,\sigma^2 I_2),\qquad \sigma=80\text{ m}
$$

(`PRIOR_OFFSET_STD_M = 80.0`, `helpers/utils.py:51`) is added to the prior, converted
from meters to satellite pixels by the local pixel-per-meter scales of Sec. 3.3.3
(`helpers/utils.py:327`). The offset is drawn from a generator **seeded per image** by
the CRC-32 of the string `"<flight>/<filename>"` (`helpers/utils.py:326`). CRC-32 is
stable across processes — unlike Python's built-in `hash`, which is randomized per
process — so every image receives the same offset on every run and across parallel
workers, making the whole benchmark reproducible.

### 3.3.3 Metric-isotropic, heading-aligned satellite crop

Matching a drone image against a raw slice of the satellite map is unreliable: the
slice has an arbitrary scale and orientation relative to the drone view. Instead the
pipeline resamples a **metric-isotropic** patch — one whose ground-sampling distance
(GSD) is identical in both axes — rotated to the drone's heading (`metric_crop` /
`_metric_affine`, `helpers/utils.py:157`).

Let $g$ be the target GSD in meters per pixel (Sec. 3.3.4), $\theta$ the drone yaw
(`Phi1`), and let the local satellite pixel-per-meter scales at the patch's mid-latitude
be

$$
s_x = \frac{p_{\text{lon}}}{\cos(\phi_{\text{mid}})\cdot D},\qquad
s_y = \frac{p_{\text{lat}}}{D},\qquad D = 111{,}320\ \text{m/}^{\circ},
$$

with $D$ (`DEG_TO_M`, `helpers/utils.py:30`) the meters-per-degree-latitude constant
and the $\cos\phi_{\text{mid}}$ term correcting longitude for the local meridian
convergence. The $2\times3$ affine mapping a patch pixel $(u,v)$ to a satellite pixel
$(X,Y)$ is

$$
\begin{bmatrix} X \\ Y \end{bmatrix}
= \underbrace{\begin{bmatrix}
g\,s_x\cos\theta & -\,g\,s_x\sin\theta & t_x \\
g\,s_y\sin\theta & \phantom{-}g\,s_y\cos\theta & t_y
\end{bmatrix}}_{M}
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix},
$$

where the translation $(t_x,t_y)$ centres the patch on the (perturbed) prior
$(c_x,c_y)$: $t_x = c_x - \tfrac{W}{2}M_{00} - \tfrac{H}{2}M_{01}$ and analogously for
$t_y$ (`helpers/utils.py:165`). The patch is produced by an inverse-warp resample
(`cv2.warpAffine`, `WARP_INVERSE_MAP`, replicate border; `helpers/utils.py:206`). $M$
is kept in double precision so that the later pixel→GPS conversion retains sub-centimetre
precision near large satellite coordinates.

A crop is **rejected** (and the image skipped) when less than
$\text{MIN\_PATCH\_COVERAGE}=0.2$ of the patch's source rectangle overlaps the tile
(`helpers/utils.py:194`), discarding samples whose footprint lies mostly off the map.

### 3.3.4 Per-flight ground-sampling calibration

The target GSD sets the metric scale of the whole patch, so it directly determines
whether the reported meters are trustworthy: too small and the satellite patch is too
zoomed-in to contain the drone footprint, too large and the inliers spread thin. The
GSD is modelled as proportional to flight altitude $a$ (`height`),

$$
g \;=\; \frac{\text{SEARCH\_FACTOR}\cdot K_f \cdot a}{W},
$$

(`metric_m_per_px`, `helpers/utils.py:89`). Here $\text{SEARCH\_FACTOR}=1.75$
(`helpers/utils.py:50`) enlarges the satellite patch beyond the drone's own footprint
so there is search margin around the prior, and $K_f$ is a **per-flight drone-footprint
factor** stored in `K_PER_FLIGHT` (`helpers/utils.py:40`). The factors for flights
`01/02/03/08` are hand-anchored; the remaining flights are calibrated automatically from
the intrinsic scale of estimated homographies, $\sqrt{\lvert\det H_{[:2,:2]}\rvert}$,
via `pipelines/calibrate_k.py`. Flights not present in the table fall back to a
geometric default $K_{\text{def}} = 1.75\cdot 2.0\cdot\tan(35^\circ)\approx 2.45$
derived from a nominal $35^\circ$ camera half-FOV (`helpers/utils.py:44`).

### 3.3.5 Geo-referencing conversions and the error metric

GPS↔pixel conversion within a tile is linear (`helpers/utils.py:81`):

$$
\text{gps\_to\_px}:\;
s_x=(\lambda-\lambda_{\text{LT}})\,p_{\text{lon}},\;
s_y=(\phi_{\text{LT}}-\phi)\,p_{\text{lat}},
$$

$$
\text{sat\_px\_to\_gps}:\;
\phi=\phi_{\text{LT}}-s_y/p_{\text{lat}},\;
\lambda=\lambda_{\text{LT}}+s_x/p_{\text{lon}}.
$$

A patch pixel is mapped to geographic coordinates by first applying the crop affine
$M$ and then `sat_px_to_gps` (`patch_px_to_gps`, `helpers/utils.py:213`). The localization
error is the great-circle distance (`haversine_m`, `helpers/utils.py:72`)

$$
d(p_1,p_2)=2R\,\arcsin\sqrt{\sin^2\tfrac{\Delta\phi}{2}
+\cos\phi_1\cos\phi_2\sin^2\tfrac{\Delta\lambda}{2}},\qquad R=6{,}371{,}000\text{ m}.
$$

Concretely, the drone-image centre is projected through $H$ to a patch pixel, mapped to
$(\hat\phi,\hat\lambda)$ via $M$, and compared to the ground truth: this haversine
distance is the per-image `offset_m` reported throughout (`helpers/utils.py:361`).

## 3.4 Feature-Matching Localization

### 3.4.1 Shared pipeline harness

All feature-matching methods share one driver, `run_pipeline`
(`helpers/workers.py`). Each method contributes only a thin script defining how to load
its model and how to build a **match factory** — a closure that, given a drone image,
returns a function mapping a satellite patch to a dictionary

$$
\{\texttt{sat\_kp},\ \texttt{drone\_kp},\ \texttt{raw},\ \texttt{good},\
\texttt{inliers},\ H\},
$$

i.e. keypoint counts, raw and filtered match counts, the RANSAC inlier count, and the
estimated homography. The harness owns everything else: flight iteration, parallelism
(one worker per GPU, or per-flight CPU chunking), the per-image geometric bookkeeping of
Sec. 3.3, visualization, and writing the fixed-schema results CSV. New matchers are
added by reproducing this contract rather than re-implementing the loop, which keeps the
geometric core identical across methods and makes their comparison fair. The specific
matchers (SIFT/ORB baseline, LightGlue, LoFTR, EfficientLoFTR, XoFTR, RoMa, MATCHA) are
described in the Background chapter (Sec.~\ref{bg:matchers}).

### 3.4.2 Classical baseline

The classical baseline (`pipelines/Baseline_pipeline.py`) provides a detector-and-
descriptor lower bound against which the learned matchers are measured. It supports
SIFT, ORB (5000 features), and BRISK detectors (`Baseline_pipeline.py:23`). Descriptors
are matched with a $k=2$ nearest-neighbour search — FLANN (KD-tree, `trees=5`,
`checks=50`) for SIFT, brute-force Hamming for the binary descriptors — and filtered by
**Lowe's ratio test** at threshold $0.75$ (`LOWE = 0.75`, `Baseline_pipeline.py:19`).

### 3.4.3 Robust homography estimation and acceptance

Surviving correspondences are passed to a robust homography fit at a common reprojection
threshold of $\text{RANSAC\_THRESH}=5.0$ px (`helpers/utils.py:22`). The robust
estimator itself varies by matcher: the classical baseline uses plain `cv2.RANSAC`,
most learned matchers use the MAGSAC estimator (`cv2.USAC_MAGSAC`, confidence
$0.9999$), and RoMa fits a partial-affine model (`cv2.estimateAffinePartial2D`,
RANSAC) that is then promoted to a homography. A homography is **accepted** as a valid
localization only when it has at least $\text{MIN\_INL}=7$ inliers
(`helpers/utils.py:23`, `helpers/utils.py:360`); images below this gate count toward the
acceptance rate but contribute no metric error. The fraction of accepted images is
reported alongside the accuracy figures, since a method that localizes a few images very
precisely is not equivalent to one that localizes most images adequately.

### 3.4.4 Preprocessing and determinism

Unless disabled (`--no-clahe`), both the drone image and the satellite patch are
contrast-normalized with **CLAHE** (clip limit $2.0$, $8\times8$ tile grid) applied to
the lightness channel in LAB colour space, which improves matching across the drone↔
satellite appearance gap. All random number generators (`random`, NumPy, OpenCV, and —
in the GPU workers — PyTorch) are seeded to $0$ at import (`helpers/utils.py:17`).
Together with the fixed RANSAC settings and the CRC-32-seeded priors of Sec. 3.3.2, this
makes every run bit-for-bit reproducible.

## 3.5 Match and Localization Visualization

To inspect behaviour qualitatively and to sanity-check that the geometry and GSD
calibration are correct, the pipeline can emit per-image diagnostic figures
(`--visualize`, `helpers/visualization.py`). Each figure places the drone image and the
satellite patch side by side, drawing the surviving correspondences between them
(`cv2.drawMatches`; dense matchers route through `save_dense_viz`,
`helpers/visualization.py:76`). On the satellite patch four overlays are rendered
(`_draw_overlays`, `helpers/visualization.py:18`):

- a **green cross and circle** at the true ground-truth location;
- a faint **grey ring** at the GPS prior, i.e. the crop centre, shown only when it
  differs from the ground truth, making the injected prior offset of Sec. 3.3.2 visible;
- a **yellow/red dot** at the predicted location (the drone centre projected through
  $H$), joined to the ground truth by a line annotated with the error in meters;
- **accuracy rings** at 20 m and 25 m around the ground truth, giving an immediate
  visual sense of whether a prediction falls within tolerance.

Rather than dumping a figure per image, the harness keeps only the best- and
worst-scoring cases per flight (ranked by inlier count, three each;
`helpers/utils.py:392`), which surfaces both typical successes and instructive failure
modes for the discussion. Figures are written as JPEGs at quality 85.

## 3.6 CLIP-Based Retrieval **(provisional)**

<!-- TODO: this line of work may still change; the prose below is provisional. -->

The retrieval paradigm dispenses with geometry: it embeds the drone image and a grid of
satellite tiles into a shared feature space and ranks tiles by cosine similarity. On top
of plain embedding retrieval, a text-conditioned variant uses natural-language captions
as a view-invariant bridge across the drone↔satellite appearance gap. The motivation and
the underlying encoders are detailed in the Background chapter (Sec.~\ref{bg:clip});
this section records the procedure.

### 3.6.1 Embedding retrieval

The satellite map is tiled into a gallery (tile size $1024$, stride $512$, i.e. 50 %
overlap; `pipelines/clip_pipeline.py`). Each tile and the query drone image are encoded
and L2-normalized, and tiles are ranked by cosine similarity to the query; no homography
is estimated. Gallery embeddings are cached per (model, tile size, stride, file mtime)
so repeated runs are cheap. The results CSV records, per image, the rank of the
ground-truth tile (`gt_tile_rank`), whether any top-$k$ tile is within the distance
threshold (`top{k}_hit`), and — to model retrieval given a coarse prior — the
ground-truth rank computed only among tiles within radius $R$ of the noisy prior
(`gt_rank_r{R}`).

### 3.6.2 View-invariant captions

Captions are generated for satellite crops, drone queries, and gallery tiles by a
vision-language model (`pipelines/caption_crops.py`; backends Ollama (default), Qwen2-VL,
Moondream, or Anthropic). The model is prompted for 10–15-word descriptions of *physical*
content — roads, buildings, water, land cover, cardinal layout — while explicitly
avoiding appearance cues such as colour and brightness, which differ sharply between the
two views. Captions are written to resumable JSONL caches so the (slow) captioning step
can be interrupted and continued.

### 3.6.3 Tri-modal LoRA fine-tuning

To pull the three modalities into a common space, the CLIP backbone
(`openai/clip-vit-base-patch32`) is fine-tuned with low-rank adapters (LoRA; rank 8,
$\alpha=16$, dropout 0.05) inserted into the attention and MLP projections
(`q/k/v/out_proj`, `fc1`, `fc2`) of both the vision and text towers
(`pipelines/clip_lora_train.py`). Training minimizes a **symmetric InfoNCE** loss. For a
batch of $N$ paired, L2-normalized embeddings $\{a_i\},\{b_i\}$ and learned temperature
(logit) scale $s$,

$$
\ell(A,B)=\tfrac{1}{2}\Big[\mathrm{CE}\big(s\,AB^\top, y\big)
+\mathrm{CE}\big(s\,BA^\top, y\big)\Big],\qquad y_i=i,
$$

(`info_nce`, `clip_lora_train.py:207`), i.e. each row should match its diagonal
counterpart. The total loss couples all three modality pairs — drone↔text, satellite↔
text, and drone↔satellite,

$$
\mathcal{L}= \ell(D,T)+\ell(S,T)+\ell(D,S),
$$

(`clip_lora_train.py:253`) so that text acts as a shared anchor while the two image views
are also aligned directly. Optimization uses AdamW at learning rate $10^{-4}$ with a
cosine schedule.

### 3.6.4 Spatial within-flight split

Because consecutive drone frames along a flight overlap heavily, a random train/test
split leaks: nearly identical frames land on both sides. Instead the rows of each flight
are split **spatially** (`split_flight_rows`, `helpers/utils.py:220`): the frames are
sorted along the wider-spread geographic axis and a contiguous band at one end (a
fraction $\text{test\_frac}=0.25$) is held out as the test set, with the remainder used
for training. Every flight therefore contributes both a training band and a disjoint
test band, and an optional guard band can be dropped between them to remove seam overlap.

### 3.6.5 Image–text fusion retrieval

At retrieval time, query and gallery representations blend the image and text embeddings
(`pipelines/clip_fusion_pipeline.py`):

$$
e = \alpha\, e_{\text{img}} + (1-\alpha)\, e_{\text{txt}}, \qquad
e \leftarrow e/\lVert e\rVert,
$$

where $\alpha$ is swept (`--fuse-alpha`) from $0$ (text-only) through balanced and
image-favoured blends to $1$ (image-only); a `--no-sat-text` ablation removes captions
from the gallery side. The fusion pipeline reuses the gallery, retrieval, and CSV
machinery of Sec. 3.6.1, so the recall analysis of Sec. 3.7.2 applies unchanged.

## 3.7 Evaluation Protocol

### 3.7.1 Geometric localization: accuracy at X meters

Geometric methods are scored by **accuracy at X meters** (A@Xm): the fraction of
accepted images whose localization error is within $X$,

$$
A@X\text{m}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\,\text{offset\_m}_i\le X\,\right],
\qquad X\in\{5,10,15,20,25\}\text{ m},
$$

(`ACC_THRESHOLDS`, `helpers/utils.py:26`; summary in `helpers/results.py`). Alongside the
accuracy curve, the summary reports the acceptance rate (Sec. 3.4.3) and error statistics
(mean, median, RMSE, and 90th percentile) over the accepted images, so that precision and
coverage can be read separately.

### 3.7.2 Retrieval: Recall@k **(provisional)**

Retrieval methods are scored by **Recall@k**: the fraction of queries whose ground-truth
tile is ranked in the top $k$,

$$
R@k=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\,\text{rank}_i < k\,\right],
\qquad k\in\{1,5,10\},
$$

computed from the `gt_tile_rank` column (`analyze/retrieval_recall.py`). A
prior-conditioned variant ranks the ground-truth tile only among tiles within a radius
$R$ of the noisy GPS prior (`gt_rank_r{R}`), reporting recall *given* a coarse location
estimate. Because geometric and retrieval methods produce different quantities — meters
versus tile rank — they are reported on their respective protocols and are not collapsed
into a single number.

## 3.8 Experimental Setup and Reproducibility

All reported runs are executed on a SLURM cluster inside an Apptainer container that
bakes in the matcher and embedding dependencies; the repository is bind-mounted
read-only at run time, so code changes need no image rebuild. Compute nodes are
**offline**: every model weight and cache is pre-staged, and the text-CLIP jobs set
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. Each method has a `slurm/run_*.sh` script
that binds the dataset, weights, and caches, runs one pipeline, and archives the results
back. Jobs are placed on A100 partitions (80 GB for the largest backbones, 40 GB
otherwise) to avoid out-of-memory failures on smaller GPUs. Reproducibility rests on the
fixed seeds, fixed RANSAC parameters, and CRC-32-seeded priors described above, together
with a stable results-CSV schema that the analysis scripts and the result-archiving step
both depend on.

## 3.9 Method Comparison

<!-- TODO (deferred until results are in): decide and write the framing —
     whether the text-conditioned CLIP experiment is presented as the central
     contribution or the three families are weighted as a balanced benchmark —
     and any "main method" narrative. No prose until results land. -->
