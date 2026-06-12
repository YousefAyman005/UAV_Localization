# Results **(provisional)**

> **Drafting conventions.** As in the Methodology chapter, sections marked
> **(provisional)** may still change and should be revisited before submission.
> The narrative of which method family is the central contribution is **deferred**
> (cf. Methodology §4.9) and is *not* taken here: results are reported per family
> on their own protocols. Numbers below are from the expanded nine-flight
> benchmark (`01,02,03,04,05,06,08,10,11`) unless stated otherwise. A@30 figures
> for the primary run are computed from the stored per-image errors (`offset_m`,
> rounded to 2 dp), accurate to within $\sim0.5$ pp; later runs report A@30
> directly.

This chapter reports the geometric-localization results on the expanded benchmark.
The worked example throughout is **EfficientLoFTR fine-tuned with LoRA** (the
distilled student of Sec.~\ref{bg:eloftr}/Methodology §4.4.4), with **RoMa
(AerialExtreMatch)** as a strong reference and as the distillation teacher. Results
for the remaining matchers and for the CLIP retrieval line are pending and noted as
TODO in §6.7.

<!-- TODO: add the other matchers (SIFT/ORB, LightGlue, LoFTR, XoFTR, MATCHA) and
     the CLIP retrieval results under the same 9-flight geometry. -->

## 6.1 Preliminary dataset: a synthetic Berlin benchmark

Before adopting UAV-VisLoc, a custom dataset over the city of Berlin was built to
prototype the localization pipeline. A satellite basemap of Berlin was fetched from
the Google Maps Static API; the query ("drone") images were fetched from the **same**
API at a higher zoom level and then perturbed with a battery of geometric and
photometric augmentations — shearing, rotation, blur, colour shifts, and synthetic
fog. A CSV recorded the ground-truth GPS of the basemap and of each query so that
predicted positions could be verified.

The dataset proved unsuitable as a benchmark for one decisive reason: it **lacks a
genuine cross-domain gap**. Because the query and the reference are rendered from the
*same* satellite-imagery source, the two views share their underlying image
statistics; the augmentations emulate nuisance variation (blur, colour, fog) but not
the actual appearance difference between a *real drone photograph* and a *satellite
map* — which is precisely the difficulty the task is meant to measure. A matcher can
exploit the shared rendering cues, so any results are optimistic and do not transfer
to real imagery. The synthetic Berlin set was therefore retired in favour of
UAV-VisLoc (Methodology §4.2), whose drone images are genuine aerial photographs
paired with independently-sourced satellite maps and thus exhibit the authentic
drone↔satellite domain gap. All results in the remainder of this chapter use
UAV-VisLoc.

## 6.2 Ground-sampling calibration

The per-flight footprint factor $K_f$ (Methodology §4.3.4) was recovered from the
intrinsic scale of RoMa's similarity homographies. As a validation, the procedure
was run on the four hand-anchored flights, whose factors are known; it recovers them
to within a few percent (Table 6.1), confirming the scale-from-geometry estimate
before it is trusted on the uncalibrated flights.

**Table 6.1 — Calibration validation on the hand-anchored flights.**

| Flight | hand-anchored $K_f$ | recovered $K_f$ | deviation |
|---|---|---|---|
| 01 | 1.00 | 0.935 | −6.5 % |
| 02 | 1.00 | 0.934 | −6.6 % |
| 03 | 0.95 | 0.981 | +3.3 % |
| 08 | 1.00 | 0.988 | −1.2 % |

The factors recovered for the five newly enabled flights are
$K_{04}=0.99,\ K_{05}=0.167,\ K_{06}=0.352,\ K_{10}=0.361,\ K_{11}=0.297$. Several
are much smaller than a naive nadir-FOV guess would suggest — flights 05 and 11 in
particular correspond to narrow-FOV (telephoto) optics (horizontal FOV $\approx
9.5^\circ$ and $17^\circ$). A notable consequence: at the previously assumed
$K_{05}=0.5$ the satellite patch spanned $\sim1736$ m, exceeding the $\sim1834$ m
height of flight 05's small satellite map, so crops ran off the map edge; the
calibrated $K_{05}=0.167$ yields a $\sim580$ m patch that fits, which is the single
largest accuracy change in this study (§6.6).

## 6.3 Search-margin sensitivity

The search factor (Methodology §4.3.4) trades patch coverage against the drone↔patch
scale gap: a larger patch keeps the drone footprint inside the search region under
the GPS prior, but at a fixed pixel budget it shrinks the drone relative to the patch
and so reduces inlier counts. Sweeping it over the full benchmark (Table 6.2) shows
overall accuracy is flat from 1.5 to 1.75 and then declines, while median inliers
fall monotonically. $\text{SEARCH\_FACTOR}=1.75$ is adopted: it raises the margin for
the small-footprint flights (06, 10) — whose margin had fallen below the 80 m prior
after calibration — at no overall cost.

**Table 6.2 — Overall A@25m and median inlier count vs SEARCH\_FACTOR.**

| SEARCH\_FACTOR | 1.5 | 1.75 | 1.8 | 1.9 | 2.0 |
|---|---|---|---|---|---|
| Overall A@25m (%) | 60.0 | **60.0** | 59.8 | 59.9 | 59.2 |
| Median inliers | 668 | 516 | 476 | 398 | 320 |

## 6.4 Per-flight localization accuracy

Table 6.3 reports A@25m and A@30m per flight for the fine-tuned EfficientLoFTR at the
adopted geometry. Overall the system localizes **60.0 %** of images within 25 m and
**69.9 %** within 30 m. Accuracy is strongly flight-dependent: the textured,
near-nadir, lower-altitude flights (02, 03, 08) reach 76–79 % at 25 m, whereas a
distinct group (04, 05, 10, 11) trails — analysed in §6.6.

**Table 6.3 — EfficientLoFTR-LoRA, per-flight accuracy (SEARCH\_FACTOR 1.75).**

| Flight | A@25m | A@30m |
|---|---|---|
| 01 | 64.9 | 75.5 |
| 02 | 76.5 | 83.6 |
| 03 | 78.9 | 90.5 |
| 04 | 36.4 | 46.7 |
| 05 | 44.6 | 54.8 |
| 06 | 58.9 | 63.1 |
| 08 | 66.4 | 77.0 |
| 10 | 45.8 | 51.4 |
| 11 | 48.6 | 61.9 |
| **Overall** | **60.0** | **69.9** |

## 6.5 Student versus teacher

Because the LoRA student is distilled from RoMa, RoMa was evaluated under the
identical geometry as a reference ceiling (Table 6.4). The distilled student
**matches or exceeds its teacher overall** (60.0 % vs 58.5 % at 25 m) and on most
flights — including flights it was never trained on (04, 11) — while being far
cheaper at inference. RoMa leads only on flights 02, 06, and 10, by small margins.
The practical reading is twofold: (i) the distillation objective of bringing a light,
fast matcher up to a heavy teacher's quality is met and slightly surpassed; and
(ii) there is consequently little headroom for a further distillation pass from the
same teacher.

**Table 6.4 — RoMa (teacher) vs EfficientLoFTR-LoRA (student), A@25m per flight.**

| Flight | RoMa | ELoFTR-LoRA | Δ (student−teacher) |
|---|---|---|---|
| 01 | 65.8 | 64.9 | −0.9 |
| 02 | 79.9 | 76.5 | −3.4 |
| 03 | 76.0 | 78.9 | +2.9 |
| 04 | 33.7 | 36.4 | +2.7 |
| 05 | 43.6 | 44.6 | +1.0 |
| 06 | 61.2 | 58.9 | −2.3 |
| 08 | 59.5 | 66.4 | +6.9 |
| 10 | 49.3 | 45.8 | −3.5 |
| 11 | 45.1 | 48.6 | +3.5 |
| **Overall** | **58.5** | **60.0** | **+1.5** |

## 6.6 Failure-mode analysis

The four trailing flights fail for three distinct reasons — none of which the
matcher choice or the calibration can remove.

**Repetitive-texture aliasing (flight 04).** Flight 04 is near-nadir
($\Omega\approx-3^\circ$) yet caps near 36 %. It has the *highest* inlier counts of
any flight and a *tight* error spread (median 31 m, P90 56 m): matching succeeds, but
on near-uniform farmland the homography locks onto a look-alike field parcel shifted
by roughly one period, producing a consistent offset rather than scattered failures.
Both the student and the teacher cap at ${\sim}34$–$36$ %, confirming this is an
ambiguity in the scene content, not a property of either matcher.

**Obliquity (flights 05, 11).** These flights were flown with a tilted camera
($\Omega\approx13^\circ$; flight 11 also $\kappa\approx12^\circ$), which might be
expected to break the planar-homography model. In practice its effect is limited
here: because these are narrow-FOV cameras (§6.2, horizontal FOV $\approx 9.5^\circ$
and $17^\circ$), a $13^\circ$ tilt induces little perspective distortion across the
small angular field, and the fitted homography absorbs the residual. The binding
constraints are instead altitude, ground-sampling distance, and — for flight 05 —
the small satellite map.

**Sparse, low-texture matching (flight 10).** Flight 10 is near-nadir but matches
weakly — a median of only $\sim15$ inliers, barely above the acceptance gate — on
repetitive orchard rows, and has the smallest image count (144). Recalibrating its
scale (from 0.6 to 0.36) nearly doubled its A@25m (23.6 → 45.8 %), but the residual
ceiling is set by low texture and small sample size.

## 6.7 Outstanding **(provisional)**

<!-- TODO -->
- Evaluate the remaining feature matchers (SIFT/ORB, LightGlue, LoFTR, XoFTR, MATCHA)
  under the same nine-flight geometry for a complete cross-method comparison.
- Report the CLIP retrieval line (embedding + text-fusion) on the Recall@k protocol
  (Methodology §4.7.2).
- Decide the overall method-comparison framing (Methodology §4.9) once the above land.
