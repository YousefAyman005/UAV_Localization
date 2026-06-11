# Methodology

> **Drafting conventions.** Equations and constants in this chapter are taken
> verbatim from the implementation. Cross-references to method
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
at true ground position $p^{\ast}=(\phi^{\ast},\lambda^{\ast})$ (latitude, longitude), the system
receives a **noisy GPS prior**

$$
\hat p = p^{\ast} + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\sigma^2 I_2),
$$

and must estimate the position $\hat{p}_{\text{est}}$ on the satellite map. Performance
is reported as the great-circle (haversine) distance between estimate and ground
truth, **in meters** — not in pixels — so that results are comparable across flights
with different satellite resolutions.

The apparatus implements two families of localization method, described in turn
below. The **geometric** family (Sec. 3.3–3.5) establishes pixel correspondences
between the drone image and a satellite patch, fits a homography, and projects the
drone-image centre onto the map to obtain a metric position. The **retrieval**
family (Sec. 3.6) embeds the drone image and a grid of satellite tiles into a common
feature space and ranks tiles by similarity, without estimating any geometry. Because
the two yield different quantities — a position in metres versus a tile ranking —
their evaluation protocols differ accordingly (Sec. 3.7).

## 3.2 Dataset

Experiments use the **UAV-VisLoc** dataset [CITE: UAV-VisLoc]. Each flight folder
contains a georeferenced satellite GeoTIFF, a directory of drone images, and a CSV of
per-image metadata. The relevant drone-CSV columns are `filename, lat, lon, height`
and `Phi1` (the yaw angle, in compass convention, clockwise from north); the satellite
metadata CSV gives each map's top-left (LT) and bottom-right (RB) corner coordinates.

**Flight selection.** The dataset contains eleven flights, of which two cannot be
processed by the single-tile pipeline: flight **07**, whose satellite image is a narrow
$3000\times170$-pixel strip too small for metric cropping, and flight **09**, whose
satellite coverage is split across four separate tiles. The remaining nine —
`01, 02, 03, 04, 05, 06, 08, 10, 11` — constitute the usable set
on which the geometric experiments operate.

## 3.3 The Geo Core: From Correspondences to Meters

The geometric paradigm is realised by a single per-image procedure. It is
matcher-agnostic: any matcher is plugged in through a uniform matching interface
(Sec. 3.4.1), while the geometric bookkeeping below is shared.

Throughout the geo core, georeferencing within a satellite map is treated as a linear
map between pixel and geographic coordinates. From the map's top-left (LT) and
bottom-right (RB) corner coordinates (Sec. 3.2) and its size $w\times h$, the pixels
per degree are

$$
p_{\text{lat}} = \frac{h}{\phi_{\text{LT}}-\phi_{\text{RB}}}, \qquad
p_{\text{lon}} = \frac{w}{\lambda_{\text{RB}}-\lambda_{\text{LT}}}
$$

These two scales underpin both the metric crop (Sec. 3.3.3)
and the pixel↔GPS conversions and error metric (Sec. 3.3.5).

### 3.3.1 Per-image pipeline

For each drone image the pipeline:

1. loads the drone image, resizes it to the working resolution
   $W\times H = 1024\times680$ with area-averaging interpolation (the alias-free
   filter for the $\approx 4\times$ downscale), and applies contrast normalization
   (Sec. 3.4.4);
2. selects the satellite tile containing the GPS prior and locates the prior in
   satellite pixels; images whose prior falls outside the map are skipped;
3. perturbs that centre with a simulated GPS-prior offset (Sec. 3.3.2);
4. samples a metric-isotropic, heading-aligned satellite patch about the perturbed
   centre (Sec. 3.3.3);
5. runs the matcher to obtain a transform $H$ mapping drone-image pixels to
   patch pixels (Sec. 3.4);
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

is added to the true position to form the noisy prior on which the patch is centred,
converted from meters to satellite pixels by the local pixel-per-meter scales of
Sec. 3.3.3. The offset is drawn from a generator **seeded per image** by the CRC-32
checksum of the flight and image name. CRC-32 is stable across processes — unlike
hash functions that are randomized per process — so every image receives the same
offset on every run and across parallel workers, making the whole benchmark
reproducible.

### 3.3.3 Metric-isotropic, heading-aligned satellite crop

Matching the drone image against a raw slice of the satellite map is unreliable, because
that slice sits at an arbitrary scale and orientation relative to the drone view. The
pipeline therefore resamples a **standardised** satellite patch: it has a fixed
ground-sampling distance (GSD) — the same
metres-per-pixel on both axes — and is rotated so the drone's heading points up. Drone
image and patch then differ only by a small residual offset, which is exactly what the
matcher recovers.

The patch is produced by a single affine warp that composes three steps:

1. **Scale** the target GSD $g$ (metres/pixel, Sec. 3.3.4) into satellite-pixel units.
   Since the map is georeferenced in degrees, this uses the local pixel-per-meter scales
   at the patch's mid-latitude,
   $$
   s_x = \frac{p_{\text{lon}}}{\cos(\phi_{\text{mid}})\cdot D},\qquad
   s_y = \frac{p_{\text{lat}}}{D},\qquad D = 111{,}320\ \text{m/}^{\circ},
   $$
   where the $\cos\phi_{\text{mid}}$ term corrects east–west distance for meridian
   convergence and $D$ is the meters-per-degree-latitude constant.
2. **Rotate** by the drone yaw $\theta$ (`Phi1`).
3. **Translate** so the patch is centred on the (perturbed) GPS prior $(c_x,c_y)$.

These collapse into one $2\times3$ matrix $M$, applied as an inverse-warp resample
with bilinear interpolation and replicated borders. Projecting a patch pixel through
$M$ and converting to GPS
(Sec. 3.3.5) yields that pixel's geographic coordinate. $M$ is kept in double precision
so the pixel→GPS conversion stays sub-centimetre even at large satellite coordinates.

A crop is **rejected** (and the image skipped) when less than $0.2$ of the patch's
source footprint overlaps the tile, discarding samples whose footprint lies mostly
off the map.

In full, the warp from patch pixel $(u,v)$ to satellite pixel $(X,Y)$ is

$$
\begin{bmatrix} X \\ Y \end{bmatrix}
= \underbrace{\begin{bmatrix}
g\,s_x\cos\theta & -\,g\,s_x\sin\theta & t_x \\
g\,s_y\sin\theta & \phantom{-}g\,s_y\cos\theta & t_y
\end{bmatrix}}_{M}
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix},
$$

with the translation $t_x = c_x - \tfrac{W}{2}M_{00} - \tfrac{H}{2}M_{01}$ (and
analogously $t_y$) centring the patch on $(c_x,c_y)$.

### 3.3.4 Per-flight ground-sampling calibration

The target GSD sets the metric scale of the whole patch, so it directly determines
whether the reported meters are trustworthy: too small and the satellite patch is too
zoomed-in to contain the drone footprint, too large and the inliers spread thin. The
GSD is modelled as proportional to flight altitude $a$ (`height`),

$$
g \;=\; \frac{F\cdot K_f \cdot a}{W}.
$$

Here the search factor $F=1.75$ enlarges the satellite patch beyond the drone's own
footprint so there is search margin around the prior, and $K_f$ is a **per-flight
drone-footprint factor**. The factors for flights
`01/02/03/08` are hand-anchored; the remaining flights are calibrated automatically from
the intrinsic scale of estimated homographies. Because the homography RoMa fits is a
similarity, its scale $s=\sqrt{\lvert\det H_{[:2,:2]}\rvert}$ equals the drone-to-patch
scale ratio, so the footprint factor is recovered as
$K_f = s\cdot F\cdot K_{\text{used}}$ — a relation **independent of the
$K_{\text{used}}$ at which the calibration patches were built**, hence robust to a poor
initial guess. Run on the four hand-anchored
flights, the procedure recovers their factors to within a few percent, which validates it
before it is applied to the rest. (An earlier sweep that instead minimized median
localization error failed to pin $K_f$: RoMa matches densely at almost any scale, so that
objective is too flat — motivating the scale-from-geometry approach above.) Flights not
present in the table fall back to a geometric default
$K_{\text{def}} = 1.75\cdot 2.0\cdot\tan(35^\circ)\approx 2.45$ derived from a nominal
$35^\circ$ camera half-FOV.

### 4.3.5 Geo-referencing conversions and the error metric

Conversion between a satellite map's **pixel coordinates** $(X,Y)$ — the same
$(X,Y)$ produced by the crop warp of Sec. 3.3.3 — and geographic coordinates is
linear within a tile, using the pixels-per-degree scales
$p_{\text{lon}},p_{\text{lat}}$ of Sec. 3.3 and the map's top-left corner
$(\phi_{\text{LT}},\lambda_{\text{LT}})$ (Sec. 3.2). The forward (GPS→pixel) and
inverse (pixel→GPS) mappings are

$$
X=(\lambda-\lambda_{\text{LT}})\,p_{\text{lon}},\qquad
Y=(\phi_{\text{LT}}-\phi)\,p_{\text{lat}},
$$

$$
\phi=\phi_{\text{LT}}-Y/p_{\text{lat}},\qquad
\lambda=\lambda_{\text{LT}}+X/p_{\text{lon}}.
$$

Here $X,Y$ are absolute pixel coordinates in the satellite tile and are distinct
from the pixels-per-meter scale factors $s_x,s_y$ of Sec. 3.3.3.

A patch pixel is mapped to geographic coordinates by first applying the crop affine
$M$ and then the pixel→GPS mapping above. The localization error is the great-circle
distance $d$ between the estimated position and the ground truth. For any two
geographic points
$p_1=(\phi_1,\lambda_1)$ and $p_2=(\phi_2,\lambda_2)$ — each a (latitude $\phi$,
longitude $\lambda$) pair as in Sec. 3.1 —

$$
d(p_1,p_2)=2R\,\arcsin\sqrt{\sin^2\tfrac{\Delta\phi}{2}
+\cos\phi_1\cos\phi_2\sin^2\tfrac{\Delta\lambda}{2}},\qquad R=6{,}371{,}000\text{ m},
$$

where $\Delta\phi=\phi_2-\phi_1$ and $\Delta\lambda=\lambda_2-\lambda_1$ are the
coordinate differences and $R$ is the Earth radius.

The per-image error is this distance evaluated at the prediction and the ground truth:
the drone-image centre is projected through $H$ to a patch pixel and mapped via $M$ to
the estimate $\hat p_{\text{est}}=(\hat\phi,\hat\lambda)$, giving the per-image error
$e=d(\hat p_{\text{est}},\,p^{\ast})$ against the true position
$p^{\ast}=(\phi^{\ast},\lambda^{\ast})$ of Sec. 3.1.

## 3.4 Feature-Matching Localization

### 3.4.1 Shared pipeline harness

All feature-matching methods share one driver. Each method contributes only a thin
adapter defining how to load its model and how to match a given drone image against a
satellite patch; the adapter reports the per-image keypoint counts, the raw and
filtered match counts, the RANSAC inlier count, and the estimated homography $H$. The
harness owns everything else: flight iteration, parallelism (one worker per GPU, or
per-flight CPU chunking), the per-image geometric bookkeeping of Sec. 3.3,
visualization, and writing the fixed-schema results table. New matchers are added by
reproducing this contract rather than re-implementing the loop, which keeps the
geometric core identical across methods and makes their comparison fair. The specific
matchers (SIFT/ORB baseline, LightGlue, LoFTR, EfficientLoFTR, XoFTR, RoMa, MATCHA) are
described in the Background chapter (Sec.~\ref{bg:matchers}).

### 3.4.2 Classical baseline

The classical baseline provides a detector-and-descriptor lower bound against which
the learned matchers are measured. It supports SIFT, ORB (5000 features), and BRISK
detectors. Descriptors are matched with a $k=2$ nearest-neighbour search — an
approximate KD-tree search (five randomized trees, 50 leaf checks) for SIFT,
brute-force Hamming for the binary descriptors — and filtered by **Lowe's ratio test**
at threshold $0.75$.

### 3.4.3 Robust transform estimation and acceptance

Surviving correspondences are passed to a robust-fit stage that is **identical for every
matcher**: a 4-DOF similarity transform estimated under RANSAC at a reprojection
threshold of $5$ px ($5000$ iterations, confidence $0.9999$, followed by a
$10$-iteration least-squares refinement on the inliers), promoted to a $3\times3$
homography. This is the transform $H$ of Sec. 3.3.1: it maps drone-image pixels to
satellite-patch pixels, and projecting the drone-image centre through it yields the
predicted location (Sec. 3.3.5). A similarity is the appropriate model class
here: the satellite patch is metric-isotropic and heading-aligned (Sec. 3.3.3), so the
true drone$\to$patch mapping is approximately a translation plus a fixed scale, and —
unlike an 8-DOF projective fit — a 4-DOF model cannot hallucinate perspective from a
handful of inliers. Sharing one estimator across all methods ensures that accuracy
differences are attributable to the quality of the correspondences alone, never to the
robust-fit stage. A transform is **accepted** as a valid localization only when it has
at least $7$ inliers;
images below this gate count toward the acceptance rate but contribute no metric error.
The fraction of accepted images is reported alongside the accuracy figures, since a
method that localizes a few images very precisely is not equivalent to one that
localizes most images adequately.

### 3.4.4 Preprocessing and determinism

Unless disabled, both the drone image and the satellite patch are
contrast-normalized with **CLAHE** (clip limit $2.0$, $8\times8$ tile grid) applied to
the lightness channel in LAB colour space, which improves matching across the drone↔
satellite appearance gap. All random number generators (Python, NumPy, OpenCV, and —
in the GPU workers — PyTorch) are seeded to $0$.
Together with the fixed RANSAC settings and the CRC-32-seeded priors of Sec. 3.3.2, this
makes every run bit-for-bit reproducible.

## 3.5 Match and Localization Visualization

To inspect behaviour qualitatively and to sanity-check that the geometry and GSD
calibration are correct, the pipeline can emit per-image diagnostic figures. Each
figure places the drone image and the satellite patch side by side, drawing the
surviving correspondences between them. On the satellite patch four overlays are
rendered:

- a **green cross and circle** at the true ground-truth location;
- a faint **grey ring** at the GPS prior, i.e. the crop centre, shown only when it
  differs from the ground truth, making the injected prior offset of Sec. 3.3.2 visible;
- a **yellow/red dot** at the predicted location (the drone centre projected through
  $H$), joined to the ground truth by a line annotated with the error in meters;
- **accuracy rings** at 20 m and 25 m around the ground truth, giving an immediate
  visual sense of whether a prediction falls within tolerance.

Rather than dumping a figure per image, the harness keeps only the best- and
worst-scoring cases per flight (ranked by inlier count, three each),
which surfaces both typical successes and instructive failure
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
overlap). Each tile and the query drone image are encoded
and L2-normalized, and tiles are ranked by cosine similarity to the query; no homography
is estimated. Gallery embeddings are cached per model and tiling configuration
so repeated runs are cheap. The results table records, per image, the rank of the
ground-truth tile, whether any top-$k$ tile is within the distance
threshold, and — to model retrieval given a coarse prior — the
ground-truth rank computed only among tiles within radius $R$ of the noisy prior.

### 3.6.2 View-invariant captions

Captions are generated for satellite crops, drone queries, and gallery tiles by a
vision-language model served from a local Ollama instance. The model is prompted for
10–15-word descriptions of *physical*
content — roads, buildings, water, land cover, cardinal layout — while explicitly
avoiding appearance cues such as colour and brightness, which differ sharply between the
two views. Captions are written to resumable caches so the (slow) captioning step
can be interrupted and continued.

### 3.6.3 Tri-modal LoRA fine-tuning

To pull the three modalities into a common space, the CLIP backbone
(`openai/clip-vit-base-patch32` by default) is fine-tuned with low-rank
adapters (LoRA; rank 8, $\alpha=16$, dropout 0.05) inserted into the attention
projections (query/key/value/output) and the two MLP layers of every transformer
block, in both the vision and text towers. Training minimizes a **symmetric InfoNCE** loss. For a
batch of $N$ paired, L2-normalized embeddings $\{a_i\},\{b_i\}$ and logit scale $s$ —
the backbone's pretrained temperature, kept **frozen** (clamped at 100) during
fine-tuning —

$$
\ell(A,B)=\tfrac{1}{2}\Big[\mathrm{CE}\big(s\,AB^\top, y\big)
+\mathrm{CE}\big(s\,BA^\top, y\big)\Big],\qquad y_i=i,
$$

i.e. each row should match its diagonal
counterpart. Because consecutive drone frames overlap spatially (the same concern that
motivates the spatial split of Sec. 3.6.4), in-batch negatives are not all true
negatives: any pair whose ground-truth positions lie within
$100$ m of each other is **masked out of the negatives** before the
cross-entropy. The total loss couples
four modality pairings — drone↔satellite, and the image views against two caption sets:
$T_S$ (captions of the ground-truth satellite crops) and $T_D$ (captions of the drone
images themselves) —

$$
\mathcal{L}= w_{ds}\,\ell(D,S)+w_{dt}\,\ell(D,T_S)+w_{st}\,\ell(S,T_S)
+w_{ddt}\,\ell(D,T_D),
$$

with per-pairing weights (all $1$ by default) so that text
acts as a shared anchor while the two image views are also aligned directly; setting
$w_{dt}=w_{st}=w_{ddt}=0$ yields the image-only control used for attribution.
Optimization uses AdamW at learning rate $10^{-4}$ (weight decay $0.01$) with a cosine
schedule.

### 3.6.4 Spatial within-flight split

Because consecutive drone frames along a flight overlap heavily, a random train/test
split leaks: nearly identical frames land on both sides. Instead the rows of each flight
are split **spatially**: the frames are
sorted along the wider-spread geographic axis and a contiguous band at one end (a
fraction of $0.25$) is held out as the test set, with the remainder used
for training. Every flight therefore contributes both a training band and a disjoint
test band, and an optional guard band can be dropped between them to remove seam overlap.

### 3.6.5 Image–text fusion retrieval

At retrieval time, query and gallery representations blend the image and text
embeddings:

$$
e = \alpha\, e_{\text{img}} + (1-\alpha)\, e_{\text{txt}}, \qquad
e \leftarrow e/\lVert e\rVert,
$$

where $\alpha$ is swept from $0$ (text-only) through balanced and
image-favoured blends to $1$ (image-only); gallery tiles without a caption keep their
pure image embedding. The gallery weight can also be decoupled from the query weight —
e.g. an image-only query ($\alpha=1$) retrieved against a text-fused gallery
($\alpha_g=0.7$) removes the VLM from query time entirely. The fusion
pipeline reuses the gallery, retrieval, and reporting machinery of Sec. 3.6.1, so the
recall analysis of Sec. 3.7.2 applies unchanged.

## 3.7 Evaluation Protocol

### 3.7.1 Geometric localization: accuracy at X meters

Geometric methods are scored by **accuracy at X meters** (A@Xm): the fraction of
*scored* images (all non-skipped images, $N$) whose localization error is within $X$ —
images that fail the acceptance gate of Sec. 3.4.3 count as failures,

$$
A@X\text{m}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\,e_i\le X\,\right],
\qquad X\in\{5,10,15,20,25,30\}\text{ m},
$$

where $e_i$ is the per-image localization error of Sec. 3.3.5. Alongside the
accuracy curve, the summary reports the acceptance rate (Sec. 3.4.3) and error statistics
(mean, median, RMSE, and 90th percentile) over the accepted images, so that precision and
coverage can be read separately. Two diagnostics complement the gated metric: an
**ungated** accuracy computed from the centre-projection error against the true GT for
any estimated transform, which shows how much the acceptance gate costs or
protects; and the **oracle solvability rate** — the fraction of images
whose true location lies inside the searched patch at all, since a prior offset large
enough to push the ground truth out of the crop makes the image unsolvable for every
method by construction (Sec. 3.3.2).

### 3.7.2 Retrieval: Recall@k **(provisional)**

Retrieval methods are scored by **Recall@k**: the fraction of queries whose ground-truth
tile is ranked in the top $k$,

$$
R@k=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\,\text{rank}_i < k\,\right],
\qquad k\in\{1,5,10\},
$$

where $\text{rank}_i$ is the zero-based rank of the ground-truth tile (so
$\text{rank}_i<k$ means "in the top $k$"). A prior-conditioned variant ranks
the ground-truth tile only among tiles within a radius $R$ of the noisy GPS prior,
reporting recall *given* a coarse location estimate; queries whose
ground-truth tile lies outside that radius are unsolvable in this setting and are
excluded from the variant's denominator. Because geometric and retrieval methods
produce different quantities — meters
versus tile rank — they are reported on their respective protocols and are not collapsed
into a single number.

## 3.8 Experimental Setup and Reproducibility

All reported runs are executed on a SLURM cluster inside an Apptainer container that
bakes in the matcher and embedding dependencies; the repository is bind-mounted
read-only at run time, so code changes need no image rebuild. Compute nodes are
**offline**: every model weight and cache is pre-staged, and all jobs run in fully
offline mode. Each method has a dedicated job script
that binds the dataset, weights, and caches, runs one pipeline, and archives the results
back. Jobs are placed on A100 partitions (80 GB for the largest backbones, 40 GB
otherwise) to avoid out-of-memory failures on smaller GPUs. Reproducibility rests on the
fixed seeds, fixed RANSAC parameters, and CRC-32-seeded priors described above, together
with a stable results schema that the analysis scripts and the result-archiving step
both depend on.

## 3.9 Method Comparison

<!-- TODO (deferred until results are in): decide and write the framing —
     whether the text-conditioned CLIP experiment is presented as the central
     contribution or the three families are weighted as a balanced benchmark —
     and any "main method" narrative. No prose until results land. -->
