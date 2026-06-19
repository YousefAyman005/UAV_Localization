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
evaluation harness and the per-matcher configurations plugged into it, the
teacher-distilled fine-tuning that adapts one matcher to the drone↔satellite domain
gap, the diagnostic visualization, the retrieval-based localization line, and
the evaluation protocol and infrastructure that make the results reproducible.

## 4.1 Problem Formulation

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
below. The **geometric** family (Sec. 4.3–4.5) establishes pixel correspondences
between the drone image and a satellite patch, fits a homography, and projects the
drone-image centre onto the map to obtain a metric position. The **retrieval**
family (Sec. 4.6) embeds the drone image and a grid of satellite tiles into a common
feature space and ranks tiles by similarity, without estimating any geometry. Because
the two yield different quantities — a position in metres versus a tile ranking —
their evaluation protocols differ accordingly (Sec. 4.7).

## 4.2 Dataset

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

## 4.3 The Geo Core: From Correspondences to Meters

The geometric paradigm is realised by a single per-image procedure. It is
matcher-agnostic: any matcher is plugged in through a uniform matching interface
(Sec. 4.4.1), while the geometric bookkeeping below is shared.

Throughout the geo core, georeferencing within a satellite map is treated as a linear
map between pixel and geographic coordinates. From the map's top-left (LT) and
bottom-right (RB) corner coordinates (Sec. 4.2) and its size $w\times h$, the pixels
per degree are

$$
p_{\text{lat}} = \frac{h}{\phi_{\text{LT}}-\phi_{\text{RB}}}, \qquad
p_{\text{lon}} = \frac{w}{\lambda_{\text{RB}}-\lambda_{\text{LT}}}
$$

These two scales underpin both the metric crop (Sec. 4.3.3)
and the pixel↔GPS conversions and error metric (Sec. 4.3.5).

### 4.3.1 Per-image pipeline

For each drone image the pipeline:

1. loads the drone image, resizes it to the working resolution
   $W\times H = 1024\times680$ with area-averaging interpolation (the alias-free
   filter for the $\approx 4\times$ downscale), and applies contrast normalization
   (Sec. 4.4.1);
2. selects the satellite tile containing the GPS prior and locates the prior in
   satellite pixels; images whose prior falls outside the map are skipped;
3. perturbs that centre with a simulated GPS-prior offset (Sec. 4.3.2);
4. samples a metric-isotropic, heading-aligned satellite patch about the perturbed
   centre (Sec. 4.3.3);
5. runs the matcher to obtain a transform $H$ mapping drone-image pixels to
   patch pixels (Sec. 4.4);
6. projects the drone-image centre $(W/2, H/2)=(512,340)$ through $H$, converts the
   result to geographic coordinates, and measures the haversine error against the
   ground truth (Sec. 4.3.5).

### 4.3.2 Simulating a noisy GPS prior

The dataset's recorded GPS is effectively clean, so a controlled prior noise is
injected to model a realistic localization scenario and to ensure the drone is **not**
at the trivial dead-centre of the satellite patch. A two-dimensional Gaussian offset

$$
\varepsilon=(\Delta x,\Delta y)\sim\mathcal{N}(0,\sigma^2 I_2),\qquad \sigma=80\text{ m}
$$

is added to the true position to form the noisy prior on which the patch is centred,
converted from meters to satellite pixels by the local pixel-per-meter scales of
Sec. 4.3.3. The offset is drawn from a generator **seeded per image** by the CRC-32
checksum of the flight and image name. CRC-32 is stable across processes — unlike
hash functions that are randomized per process — so every image receives the same
offset on every run and across parallel workers, making the whole benchmark
reproducible.

### 4.3.3 Metric-isotropic, heading-aligned satellite crop

Matching the drone image against a raw slice of the satellite map is unreliable, because
that slice sits at an arbitrary scale and orientation relative to the drone view. The
pipeline therefore resamples a **standardised** satellite patch: it has a fixed
ground-sampling distance (GSD) — the same
metres-per-pixel on both axes — and is rotated so the drone's heading points up.

The recorded yaw, however, only approximates the camera's true image
orientation: measured as the residual rotation of the 4-DOF fit (Sec. 4.4.1)
on RoMa matches, the deviation is constant within a flight leg but differs
between legs — the sign flips with flight direction, consistent with
wind-induced crab plus a mount offset — and reaches $23°$ on flight 08. The
crop therefore applies a per-flight, per-leg **yaw correction**, the rotation
analogue of the GSD calibration of Sec. 4.3.4: frames are clustered into legs
by heading, each leg's offset is the median measured residual ($40$
frames/flight), and the corrected offsets are applied identically for every
evaluated method. A held-out re-measurement with corrected crops validates the
table. Drone image and patch then differ only by the residual offset of the
noisy prior, which is exactly what the matcher recovers.
Fig.~\ref{fig:metric-crop} shows the effect.

![Metric-isotropic, heading-aligned crop](figures/metric_crop_08.png)

> **Fig.~\label{fig:metric-crop}** The standardized satellite crop, for one
> flight-08 image (altitude $552$ m; corrected yaw $89°$ — the recorded yaw,
> $112°$, is off by $-23°$ on this leg). (a) A raw north-up satellite slice
> around the noisy prior, at native map scale: orientation and scale are
> arbitrary relative to the drone view. (b) The metric crop of the same area:
> fixed GSD, heading rotated up — directly comparable to (c), the drone image
> at working resolution. What remains is the offset between the patch centre
> (the noisy prior, marked $+$, here $39$ m off) and the true position
> (circle), which is what the matcher recovers. (d) The full satellite map
> with the rotated crop footprint.

The patch is produced by a single affine warp that composes three steps:

1. **Scale** the target GSD $g$ (metres/pixel, Sec. 4.3.4) into satellite-pixel units.
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
(Sec. 4.3.5) yields that pixel's geographic coordinate. $M$ is kept in double precision
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

### 4.3.4 Per-flight ground-sampling calibration

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
$(X,Y)$ produced by the crop warp of Sec. 4.3.3 — and geographic coordinates is
linear within a tile, using the pixels-per-degree scales
$p_{\text{lon}},p_{\text{lat}}$ of Sec. 4.3 and the map's top-left corner
$(\phi_{\text{LT}},\lambda_{\text{LT}})$ (Sec. 4.2). The forward (GPS→pixel) and
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
from the pixels-per-meter scale factors $s_x,s_y$ of Sec. 4.3.3.

A patch pixel is mapped to geographic coordinates by first applying the crop affine
$M$ and then the pixel→GPS mapping above. The localization error is the great-circle
distance $d$ between the estimated position and the ground truth. For any two
geographic points
$p_1=(\phi_1,\lambda_1)$ and $p_2=(\phi_2,\lambda_2)$ — each a (latitude $\phi$,
longitude $\lambda$) pair as in Sec. 4.1 —

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
$p^{\ast}=(\phi^{\ast},\lambda^{\ast})$ of Sec. 4.1.

## 4.4 Feature-Matching Localization

### 4.4.1 Shared pipeline harness

All feature-matching methods share one driver. Each method contributes only a thin
adapter defining how to load its model and how to match a given drone image against a
satellite patch; the adapter reports the per-image keypoint counts, the raw and
filtered match counts, the RANSAC inlier count, and the estimated homography $H$. The
harness owns everything else: flight iteration, parallelism (one worker per GPU, or
per-flight CPU chunking), the per-image geometric bookkeeping of Sec. 4.3,
visualization, and writing the fixed-schema results table. Preprocessing is also
shared: unless disabled, both images are contrast-normalized with CLAHE (clip limit
$2.0$, $8\times8$ tile grid) on the lightness channel in LAB colour space before
matching, which narrows the drone↔satellite appearance gap. So is the robust fit:
$H$ is a 4-DOF similarity estimated with RANSAC ($5$ px
threshold, $5000$ iterations, confidence $0.9999$, least-squares refinement on the
inliers) — sufficient because the patch is already scale- and heading-normalised
(Sec. 4.3.3) — and an image counts as localized only if the fit has at least $7$
inliers (the acceptance rate is reported alongside accuracy). New matchers are added by
reproducing this contract rather than re-implementing the loop, which keeps the
geometric core identical across methods and makes their comparison fair. The specific
matchers (SIFT/ORB baseline, LightGlue, LoFTR, EfficientLoFTR, XoFTR, RoMa, MATCHA) are
described in the Background chapter (Sec.~\ref{bg:matchers}); the configuration each
one is run with here is given in Sec. 4.4.3.

### 4.4.2 Classical baseline

The classical baseline provides a detector-and-descriptor lower bound against which
the learned matchers are measured. It supports SIFT, ORB, and BRISK detectors, each
capped to a common budget of $5000$ keypoints (the highest-response features) so the
methods are compared at equal keypoint count rather than at each detector's default
density. Descriptors are matched with a $k=2$ nearest-neighbour search — an
approximate KD-tree search (five randomized trees, 50 leaf checks) for SIFT,
brute-force Hamming for the binary descriptors — and filtered by **Lowe's ratio test**
at threshold $0.75$.

### 4.4.3 Learned-matcher adapters

Each learned matcher is wrapped in a thin adapter that converts the drone image
and the satellite patch into the model's expected input and returns its
correspondences in the shared $1024\times680$ working frame; everything
downstream — robust fit, acceptance, scoring — is the shared machinery of
Sec. 4.4.1 and Sec. 4.3, so the adapters below are the *only* place the methods
differ. The configurations used are:

- **LightGlue** is evaluated with three interchangeable feature backbones, each
  treated as a separate method variant: **DISK**, **DeDoDe**, and **SIFT**
  (8192 keypoints, RootSIFT-normalised descriptors). Inputs are RGB tensors at
  the working resolution.
- **LoFTR** is detector-free and semi-dense; it consumes the grayscale image
  pair directly, using the *outdoor*-pretrained weights.
- **EfficientLoFTR** likewise takes grayscale pairs, padded at the bottom/right
  with replicated borders to dimensions divisible by 32; padding preserves the
  top-left origin, so returned matches live in the original pixel frame without
  rescaling. The full-precision full model is used. An optional LoRA adapter
  produced by the fine-tuning of Sec. 4.4.4 can be merged into the model at
  load time, evaluated as the additional variant *EfficientLoFTR-LoRA*.
- **XoFTR**, a LoFTR-family matcher trained for **cross-modal**
  (thermal↔visible) matching, is included because cross-modal matching is the
  closest published proxy for the drone↔satellite appearance gap. Its native
  I/O wrapper grayscales and aspect-preservingly resizes the pair internally
  (coarse confidence threshold $0.3$) and returns matches already rescaled to
  the working frame.
- **RoMa** regresses a dense warp with per-pixel certainty, from which $5000$
  correspondences are sampled. Two checkpoints are evaluated as separate
  variants: the stock *outdoor* model, and an **AerialExtreMatch** fine-tune
  (`roma_extre`) — the same architecture with aerial-specialised weights —
  which also serves as the distillation teacher of Sec. 4.4.4.
- **MATCHA** (DISK keypoints with learned descriptors) operates at its native
  $512\times352$ input resolution (area-averaged downscale); its
  mutual-nearest-neighbour matches are rescaled back to the working frame, with
  the descriptor score matrix providing the match confidence.

### 4.4.4 Adapting a matcher to the domain gap: teacher-distilled LoRA fine-tuning

None of the matchers above was trained on real drone↔satellite pairs, so beyond
benchmarking them zero-shot, one matcher is **fine-tuned across the domain gap**.
EfficientLoFTR is chosen as the student: it ships a complete training pipeline,
its matching happens in transformer attention layers (a clean target for
parameter-efficient adaptation), and it has *not* been aerial-pretrained — so it
has headroom. The teacher is the AerialExtreMatch RoMa checkpoint
(Sec. 4.4.3), the strongest zero-shot matcher available in this study.

**Why distillation rather than geometric ground truth.** The dataset's
GPS+yaw+height metadata yields only a *coarse planar* alignment between drone
image and satellite crop — sufficient to centre a patch, far too coarse to
supervise pixel-accurate correspondences. Instead, the teacher's matches are
distilled: RoMa recovers correct geometry on appearance-hard pairs where the
stock student fails, and a per-pair homography fit to its correspondences is a
good model for near-nadir aerial scenes — and consistent with how the benchmark
itself scores (a single transform per image, Sec. 4.4.1).

**Spatial train/validation/test bands.** Fine-tuning and evaluating on the same
nine flights requires a leakage-free split. Because consecutive drone frames
overlap heavily, a random split leaks: nearly identical frames land on both
sides. Each flight's frames are instead sorted along the wider-spread geographic
axis and sliced into contiguous bands — bottom to top: **train $0.60$ | buffer
$0.05$ | validation $0.10$ | test $0.25$**. The buffer band is discarded to
remove seam overlap between train and validation; the test band is never touched
during training or model selection and is reserved for the final comparison.
Fig.~\ref{fig:split-bands} shows the resulting bands.
The same splitting function also serves the CLIP fine-tuning of Sec. 4.6.2.

![Spatial within-flight split](figures/split_bands_08.png)

> **Fig.~\label{fig:split-bands}** Spatial within-flight split, shown for
> flight 08. Each point is one drone frame at its ground-truth position,
> colored by band (train | buffer | val | test), sliced along the flight's
> wider-spread geographic axis (here longitude). The flight lines cross the
> band boundaries, so it is the discarded buffer band — not frame order — that
> separates train from validation; the test band is spatially disjoint from
> everything used for training or model selection. The other eight flights are
> split identically. The retrieval fine-tuning (Sec. 4.6.2) uses the same
> slicing without a validation band (train 0.70, buffer 0.05); its test band
> is identical.

**Training-pair generation.** For every train-band drone image a satellite crop
is built exactly as at evaluation time (Secs. 4.3.3, 4.4.1), centred either on
the true position or on an evaluation-style perturbation of it. The teacher's
dense correspondences are filtered — certainty $\ge 0.5$, mutual one-to-one
deduplication, MAGSAC inliers ($5$ px) — and stored together with the crop and
the per-pair homography $H$ (drone → crop pixels) fit to them. Pairs with fewer
than $16$ correspondences, or whose $H$ deviates from the georeferenced prior
alignment by more than $64$ m, are discarded as unreliable labels.

**Supervision and adapters.** The student's upstream loss (coarse focal + fine
local-regression $L_2$) is reused unchanged; only the ground-truth warp is
replaced, going through the per-pair homography instead of depth and camera
pose. LoRA adapters (rank $8$, $\alpha=16$, dropout $0.05$) on the coarse
transformer's query/key/value and merge projections train under $1\,\%$ of the
parameters, with the convolutional backbone frozen; at evaluation the adapter
is merged and the backbone reparameterized into its inference form.

**Augmentation.** Both images receive independent mild photometric jitter
(gamma, contrast/brightness, occasional blur or noise); additionally the
satellite crop is perturbed by a small random similarity warp ($\pm8^{\circ}$
rotation, $\pm10\,\%$ scale, $\pm48$ px translation) $S$, with the label updated
to $H' = S\,H$ and content revealed by the warp's replicated border masked out
of the supervision.

**Optimization and model selection.** AdamW at learning rate $10^{-4}$ with a
$100$-step warmup and cosine decay, for up to $30$ epochs. After every epoch the
model is validated on the validation band with a metric that **mirrors the
benchmark**: predicted matches → the same 4-DOF robust fit of Sec. 4.4.1 →
centre projection → error in meters against the true position, gated at the
same $7$-inlier acceptance threshold. The checkpoint minimizing the median
validation error (ties broken by accuracy at $25$ m) is selected. The final
student–teacher–stock comparison is then run on the held-out test band only
(Sec. 4.7.1).

## 4.5 Match and Localization Visualization

To inspect behaviour qualitatively and to sanity-check that the geometry and GSD
calibration are correct, the pipeline can emit per-image diagnostic figures. Each
figure places the drone image and the satellite patch side by side, drawing the
surviving correspondences between them. On the satellite patch four overlays are
rendered:

- a **green cross and circle** at the true ground-truth location;
- a faint **grey ring** at the GPS prior, i.e. the crop centre, shown only when it
  differs from the ground truth, making the injected prior offset of Sec. 4.3.2 visible;
- a **yellow/red dot** at the predicted location (the drone centre projected through
  $H$), joined to the ground truth by a line annotated with the error in meters;
- **accuracy rings** at 20 m and 25 m around the ground truth, giving an immediate
  visual sense of whether a prediction falls within tolerance.

Rather than dumping a figure per image, the harness keeps only the best- and
worst-scoring cases per flight (ranked by inlier count, three each),
which surfaces both typical successes and instructive failure
modes for the discussion. Figures are written as JPEGs at quality 85.

## 4.6 Retrieval-Based Localization

The retrieval paradigm dispenses with geometry: the drone image and a grid of
satellite tiles are embedded into a shared feature space, and tiles are ranked by
cosine similarity to the query, so localization is only as fine-grained as the
tiling. The encoders themselves are introduced in the Background chapter
(Sec.~\ref{bg:clip}); this section describes the retrieval machinery and the
text-conditioned fine-tuning built on top of it.

### 4.6.1 Tile-gallery retrieval

The satellite map is tiled into a gallery with tile size $1024$ and stride $512$
(50 % overlap). Each tile and the query drone image are encoded and L2-normalized,
and tiles are ranked by cosine similarity to the query. The machinery is
encoder-agnostic; seven encoders are plugged into it: the general vision–language
and self-supervised backbones CLIP, MobileCLIP, and DINOv2, the geography-aware
embeddings GeoCLIP and SatCLIP, and the cross-view geo-localization models CAMP and
Sample4Geo with their published University-1652 checkpoints. Gallery embeddings are
cached per encoder and tiling configuration.

The **ground-truth tile** of a query is the gallery tile whose centre is nearest, in
metres, to the true position. Each query's results row records the rank of the
ground-truth tile, whether any top-$k$ tile centre lies within the distance
threshold of the true position, and the ground-truth rank among only the tiles
within radius $R$ of the noisy prior; the recall protocol of Sec. 4.7.2 is computed
from these columns.

### 4.6.2 Text-conditioned fine-tuning

On top of plain image retrieval, natural-language captions serve as a third
modality intended to be invariant to the drone↔satellite view change. A
vision–language model (`qwen3.5:9b`, served through Ollama) captions three targets:
the ground-truth satellite crops used as training positives, the drone images, and
the gallery tiles. The prompt asks for 10–15-word, comma-separated noun phrases
naming permanent physical features (road shape, building density, water bodies,
land cover, cardinal directions for key features) and forbids colour and brightness
words, since appearance cues are what fails to transfer between the two views; any
that the model emits regardless are stripped afterwards. Captions are written to
resumable per-flight caches, so the slow captioning step can be interrupted and
continued.

The backbone is then fine-tuned to pull the three modalities into a common space.
Low-rank adapters (LoRA; rank 8, $\alpha=16$, dropout 0.05) are inserted into the
attention projections (query/key/value/output) and the two MLP layers of every
transformer block, in both the vision and the text tower. The trainer is
backbone-agnostic across the CLIP and SigLIP/SigLIP2 families; input resolution,
normalization, and text padding are derived from each model's configuration. Each
training example is built from one train-band drone image: the image $D$, its
positive satellite view $S$ — a metric-isotropic crop (Sec. 4.3.3) centred on
the true position with no prior noise, kept north-up so its orientation and
cardinal captions match the gallery tiles — and the captions
$T_S$ (of the satellite crop) and $T_D$ (of the drone image itself). A modality
pair is scored by $\ell(\cdot,\cdot)$, the standard symmetric InfoNCE loss of CLIP
[CITE: CLIP]: each embedding must match its paired counterpart against all other
examples in the batch, averaged over both directions, with the temperature fixed at
the backbone's pretrained value throughout fine-tuning. Because consecutive drone
frames overlap spatially, in-batch pairs whose ground-truth positions lie within
100 m of each other are masked out of the negatives. The total loss couples four
modality pairings with per-pairing weights (Fig.~\ref{fig:trimodal-loss}),

$$
\mathcal{L}= w_{ds}\,\ell(D,S)+w_{dt}\,\ell(D,T_S)+w_{st}\,\ell(S,T_S)
+w_{ddt}\,\ell(D,T_D),
$$

all 1 by default, so the two image views are aligned directly while text acts as
a shared anchor. Optimization uses AdamW (learning rate $10^{-4}$, weight decay
0.01) with cosine annealing, for 10 epochs at batch size 64. Training and
evaluation use the spatial within-flight split of Sec. 4.4.4
(Fig.~\ref{fig:split-bands}) with a test fraction
of 0.25 and a guard band of 0.05 dropped at the seam (no validation band);
every flight contributes a training band and a disjoint held-out test band.

![Tri-modal loss mechanism](figures/trimodal_loss.png)

> **Fig.~\label{fig:trimodal-loss}** Tri-modal training mechanism. The two
> image views are embedded by the vision tower, the two captions by the text
> tower; the LoRA adapters are the only trainable weights. Each dashed edge is
> one symmetric InfoNCE term of the total loss $\mathcal{L}$, weighted by the
> $w$ shown; pairings without an edge ($S$–$T_D$, $T_S$–$T_D$) are not
> supervised.

At retrieval time, query and gallery representations can blend the image and text
embeddings,

$$
e = \alpha\, e_{\text{img}} + (1-\alpha)\, e_{\text{txt}}, \qquad
e \leftarrow e/\lVert e\rVert,
$$

where $\alpha$ is the image weight ($\alpha=1$ is image-only retrieval); gallery
tiles without a caption keep their pure image embedding. The gallery weight
$\alpha_g$ can be decoupled from the query weight, so that, for example, an
image-only query is retrieved against a text-fused gallery — a configuration that
needs no VLM at query time. Fusion reuses the gallery, ranking, and reporting
machinery of Sec. 4.6.1, so the protocol of Sec. 4.7.2 applies unchanged.

## 4.7 Evaluation Protocol

### 4.7.1 Geometric localization: accuracy at X meters

Geometric methods are scored by **accuracy at X meters** (A@Xm): the fraction of
*scored* images (all non-skipped images, $N$) whose localization error is within $X$ —
images that fail the acceptance gate of Sec. 4.4.1 count as failures,

$$
A@X\text{m}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\,e_i\le X\,\right],
\qquad X\in\{5,10,15,20,25,30\}\text{ m},
$$

where $e_i$ is the per-image localization error of Sec. 4.3.5. Alongside the
accuracy curve, the summary reports the acceptance rate (Sec. 4.4.1) and error statistics
(mean, median, RMSE, and 90th percentile) over the accepted images, so that precision and
coverage can be read separately. Two diagnostics complement the gated metric: an
**ungated** accuracy computed from the centre-projection error against the true GT for
any estimated transform, which shows how much the acceptance gate costs or
protects; and the **oracle solvability rate** — the fraction of images
whose true location lies inside the searched patch at all, since a prior offset large
enough to push the ground truth out of the crop makes the image unsolvable for every
method by construction (Sec. 4.3.2).

**Band-restricted comparison for fine-tuned methods.** A method fine-tuned on the
benchmark flights (Sec. 4.4.4) must not be scored on images it trained on. Whenever a
comparison involves a fine-tuned method, the per-image results of *all* compared
methods — fine-tuned and zero-shot alike — are filtered to the held-out spatial test
band before the metrics above are computed, so every method is scored on exactly the
same rows. The train-band scores are additionally reported as an overfitting
diagnostic: a fine-tuned method whose train-band advantage over its teacher far
exceeds its test-band advantage is memorising rather than generalising.

**Systematic-error decomposition.** As a diagnostic for residual error that no
matcher can remove, the mean error vector of each flight is decomposed into an
**along-track** component (parallel to the local flight direction) and a
**world-fixed** component (north/east). A bias that is consistent across two unrelated
matchers is attributable to the dataset itself — e.g. uncompensated gimbal pitch or
GPS timestamp lag (along-track), or a georeferencing offset of the satellite map
(world-fixed) — and is reported as a per-flight error floor rather than a method
failure.

### 4.7.2 Retrieval: Recall@k 

Retrieval methods are scored by **Recall@k**: the fraction of queries whose ground-truth
tile is ranked in the top $k$,

$$
R@k=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\,\text{rank}_i < k\,\right],
\qquad k\in\{1,5,10\},
$$

where $\text{rank}_i$ is the zero-based rank of the ground-truth tile (so
$\text{rank}_i<k$ means "in the top $k$"). All retrieval runs — zero-shot and
fine-tuned alike — use only the **test-band** frames of each flight
(Sec. 4.6.2, Fig.~\ref{fig:split-bands}) as queries, so zero-shot baselines and
fine-tuned models are always
compared on identical query sets. A prior-conditioned variant ranks
the ground-truth tile only among tiles within a radius $R$ of the noisy GPS prior,
reporting recall *given* a coarse location estimate; queries whose
ground-truth tile lies outside that radius are unsolvable in this setting and are
excluded from the variant's denominator. Because geometric and retrieval methods
produce different quantities — meters
versus tile rank — they are reported on their respective protocols and are not collapsed
into a single number.

## 4.8 Experimental Setup and Reproducibility

All reported runs are executed on a SLURM cluster inside an Apptainer container that
bakes in the matcher and embedding dependencies; the repository is bind-mounted
read-only at run time, so code changes need no image rebuild. Compute nodes are
**offline**: every model weight and cache is pre-staged, and all jobs run in fully
offline mode. Each method has a dedicated job script
that binds the dataset, weights, and caches, runs one pipeline, and archives the results
back. Jobs are placed on A100 partitions (80 GB for the largest backbones, 40 GB
otherwise) to avoid out-of-memory failures on smaller GPUs. Reproducibility rests on
fixed seeds — Python, NumPy, OpenCV, and (in the GPU workers) PyTorch are all seeded
to $0$ — the fixed RANSAC parameters, and the CRC-32-seeded priors of Sec. 4.3.2,
together with a stable results schema that the analysis scripts and the
result-archiving step both depend on; every run is bit-for-bit reproducible.

## 4.9 Method Comparison

<!-- TODO (deferred until results are in): decide and write the framing —
     whether the text-conditioned CLIP experiment is presented as the central
     contribution or the three families are weighted as a balanced benchmark —
     and any "main method" narrative. No prose until results land. -->
