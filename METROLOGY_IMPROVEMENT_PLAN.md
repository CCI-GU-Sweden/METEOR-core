# METEOR-core: stability and image classification plan

This document describes a proposed implementation for more repeatable analysis of existing microscope metrology images. It is based on a review of the current repository, not on a completed quantitative comparison of the CZI files. The proposed algorithms and thresholds require validation against the existing acquisitions and the applicable Argolight target specifications before they are used for reporting instrument performance.

## Goals and boundaries

- Reduce analysis variation caused by small translations and rotations of the target, segmentation changes and Z-plane selection.
- Preserve genuine field-dependent optical effects and channel offsets. A target moving to a different part of the camera field can produce a real change; image processing cannot always distinguish it from instrument variation.
- Make failures and uncertainty visible instead of returning a plausible number from an unreliable fit.
- Classify images using explicit metadata and image content, with a manual override and an uncertain outcome.
- Use the images already in `tests/data/` and `upload/`; no new measurements are required for development. Digital perturbations test algorithm robustness, but cannot establish biological or instrument repeatability.
- Keep existing result names and API behavior where practical; version any change to a metric's scientific definition.

Star and PSF currently appear satisfactory according to user experience. Preserve them as regression baselines rather than redesigning them without evidence.

## Findings in the current code

| Priority | Finding | Consequence |
| --- | --- | --- |
| P0 | `RSF.py`: ring centers are detected after rotation and cropping of the middle plane, but used to crop the original stack. | Crops can miss the intended rings, especially when target position or angle changes. |
| P0 | `distortion_analysis.py`, `uniformity_analysis.py`, and `coregistration_analysis.py` select a Z plane by optimizing the metric being reported. | Optimistic selection bias and discontinuous plane changes; analyses of one image may use different planes. |
| P0 | `functions.py`: lattice matching uses nearest neighbors without one-to-one assignment; cropping and thresholding affect which features enter the fit. | Multiple detections can map to one node, and changes in visible features can shift the fitted grid. |
| P1 | Coregistration compares independently fitted ideal grids instead of directly matched observed centers. | Local channel differences may be suppressed; grid-index ambiguity can distort offsets. |
| P1 | Distortion reports a maximum residual. | A single erroneous centroid can dominate the reported value. Existing percentage denominators need a dimensional review. |
| P1 | Intensity searches thresholds and Z planes, partly using R² to stop; it sorts ROI intensities before regression. | Selection can favor an optimistic result, and sorting can conceal incorrect spatial assignment. |
| P1 | Intensity orientation transforms are incomplete; helper and public defaults use different spacings. | Rotation or calling path can change the result. Verify actual target geometry before choosing defaults. |
| P1 | Ring uniformity includes clipped ROIs and weights their mean by visible fraction. | Edges can be artificially dark, creating a position-dependent uniformity artifact. |
| P2 | `metrology.classify` uses filenames only; metadata is currently unused. | Misnamed or unnamed images can be sent to the wrong analysis or rejected. |
| P2 | Tests often check successful execution or finite outputs rather than invariance and accuracy. The intensity integration test allows failure. | Regressions and unstable metrics can pass. |

Additional engineering checks: `RSF.py` imports `joblib`, which is not declared as a direct dependency in `pyproject.toml`. Verify dependency declarations and audit handling of axis order, missing calibration, multi-scene images, channel selection and error result schemas. Silent fallback to 1 µm/pixel can produce misleading physical results; missing or invalid calibration should be a clear error for metrics that require it.

## Proposed architecture

Introduce shared components without requiring every existing analysis to change at once:

```text
image + metadata
  -> validate axes, calibration, channel and acquisition metadata
  -> classify pattern (or use explicit override)
  -> choose reference Z from image quality, independently of output metric
  -> detect pattern features in original image coordinates
  -> fit and match pattern, with quality diagnostics
  -> run metric-specific measurements on original pixels/coordinates
  -> return measurements + diagnostics + provenance + warnings
```

Suggested modules: `pattern_detection.py`, `focus.py`, `classification.py`, and `quality.py`. Keep `metrology_splitter` backward compatible via optional keyword arguments or a new API. Define coordinate conventions once: images are indexed `[y, x]`, while feature coordinates are `(x, y)` in original camera pixels; include crop offsets and transformation matrices whenever transformed images are used.

### Shared result diagnostics

Add optional fields to `extra_measurements` or a versioned `quality`/`provenance` object. Preserve existing measurement entries until downstream consumers have migrated.

```python
quality = {
    "status": "pass",  # pass, warning, fail
    "reasons": [],
    "selected_z": 12,
    "focus_score": 123.4,
    "matched_features": 390,
    "expected_features": 441,
    "match_residual_median_px": 0.18,
    "match_residual_p95_px": 0.46,
    "saturated_fraction": 0.0,
    "clipped_features": 0,
}
provenance = {
    "algorithm_version": "2",
    "parameters": {},
    "pixel_size_um": {"x": 0.1, "y": 0.1},
    "channel": 0,
    "classification_source": "image_content",
}
```

These values are illustrative, not acceptance thresholds. Define thresholds from the retrospective dataset and target geometry. A `fail` should not be reported as a successful measurement; a `warning` should remain visible to callers. Include raw point and ROI diagnostics where storage permits, without embedding large arrays in every result by default.

## 1. Correct RSF coordinate handling

In `RSF_analysis`, either detect in the original middle plane and crop the original stack using those centers, or apply the exact same rotation and crop to every Z plane before extracting ring volumes. Prefer original coordinates to avoid interpolation of measurement pixels. If a transformed image is useful for detection, use the inverse transform plus crop offset to map each detection back. Account for rotation around the image center, axis ordering and any resize behavior. Do not simply add the crop offset after rotation.

Implementation sketch:

```python
# Detection returns XY coordinates in the original stack's pixel frame.
centers_xy = detect_ring_centers_original_frame(stack[reference_z], metadata)
for x, y in centers_xy:
    volume = extract_bounded_volume(stack, x=x, y=y, radius_px=crop_radius)
    if volume is None:
        continue  # record a clipped-feature diagnostic
    # Existing RSF processing uses the extracted original-pixel volume.
```

Test with synthetic stacks containing rings at known coordinates and translated/rotated copies. Assert that extracted volumes contain the intended ring, including when the detection path uses a transformed view. Compare original CZI outputs before and after the fix; changes in previously incorrect results are expected. Also review the `RING_DIAMETER`/`radius_px` naming and crop dimensions against the physical target rather than changing the constants speculatively.

## 2. Select focus independently of the metric

Create a shared focus-selection function operating on a fixed reference channel and a valid pattern ROI. A candidate focus score can combine a robust edge or high-frequency measure with detection coverage and saturation checks. Evaluate alternatives on the existing stacks; Laplacian variance alone is noise-sensitive and may favor noise or saturation. Establish a deterministic tie-break rule and return the full score curve, selected plane and reason.

```python
def select_reference_plane(stack_zyx, score_plane):
    scores = np.asarray([score_plane(plane) for plane in stack_zyx], dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("No usable focus plane")
    # Deterministic first maximum; score_plane must reject unusable planes.
    index = int(np.argmax(np.where(finite, scores, -np.inf)))
    return index, scores
```

Use a common plane for comparable ring analyses when their acquisition and channels permit it. For multi-channel data, explicitly define the reference channel and document any channel-specific focus offsets. Single-plane images retain their plane. Report neighboring-plane metric variation as a sensitivity diagnostic where enough planes exist; it is not a confidence interval. Preserve current behavior behind a clearly marked legacy mode during comparison, not as the new default.

## 3. Robust ring detection and lattice registration

Detect candidate centers in original image coordinates using a method appropriate for the image scale and ring appearance. Estimate background and reject saturation, overlapping detections and clipped features. Estimate rotation, translation and spacing from many candidates, then fit a robust lattice with a fixed or bounded physical spacing based on verified target geometry. The appropriate model must be chosen deliberately:

- Rigid translation and rotation preserve magnification and distortion in the residuals.
- Allowing fitted scale removes global magnification error from the residual; report that scale separately.
- An affine fit can absorb shear and field distortion. Use it only if the intended metric explicitly excludes those effects, and document what has been removed.

Use one-to-one assignment with a distance gate, rather than independent nearest-neighbor matches. The following is a sketch; the gate must be calibrated, and both unmatched detections and nodes must be counted:

```python
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def match_nodes(nodes_xy, detections_xy, max_distance_px):
    if len(nodes_xy) == 0 or len(detections_xy) == 0:
        return np.empty((0, 2), dtype=int)
    cost = cdist(nodes_xy, detections_xy)
    # Dummy columns let a node remain unmatched instead of forcing a bad match.
    n = len(nodes_xy)
    dummy = np.full((n, n), max_distance_px)
    rows, cols = linear_sum_assignment(np.concatenate([cost, dummy], axis=1))
    keep = (cols < len(detections_xy))
    rows, cols = rows[keep], cols[keep]
    keep = cost[rows, cols] < max_distance_px
    return np.column_stack((rows[keep], cols[keep]))
```

The gated assignment above is a starting point, not a full robust fitting algorithm: evaluate competing initial orientations and grid shifts, refit on accepted pairs, check optimizer convergence and reject ambiguous solutions. An unmarked periodic lattice may not uniquely identify absolute target indices after translation by one pitch; report ambiguity and use camera coordinates or visible target landmarks where required. Avoid assuming a full 21×21 grid must always be visible. Report coverage, residual distribution, spacing and ambiguity.

For distortion, retain maximum displacement for compatibility, but also return median, p95 and a map with point quality. Define whether displacement is relative to a rigid, similarity or other fit, and compute physical X/Y distances using the appropriate calibrated pixel sizes. Revisit percentage metrics with an explicit dimensionally correct field-size denominator.

For coregistration, match detected centers across channels using the common lattice identities and a bounded displacement gate. Report global channel offset and local residual displacement separately. Do not remove global shift from the primary result unless the metric definition explicitly calls for it. Preserve channel identity and avoid silently analyzing only the first two channels.

For uniformity, measure fully visible comparable ROIs on original intensities after a documented background estimate. Mark clipped or missing cells as missing, not zero or area-weighted darker values. Report spatial coverage and the camera-frame region measured; field uniformity from different regions is not automatically comparable. Keep the legacy min/max metric if required, and add robust percentile summaries with clearly distinct names.

## 4. Redesign 4×4 intensity analysis

1. Confirm the target's actual pitch, element geometry, expected relative responses and orientation convention from its specification. The code's 9.5 and 10 µm defaults are inconsistent; neither should be declared correct without verification.
2. Select focus using the independent procedure above. Detect and register the pattern once; assign stable row/column identities with a confidence/ambiguity check. Orientation should cover all rotations and reflections relevant to the acquisition and must not assume the largest thresholded region is a reliable landmark without validation.
3. Measure original, non-normalized pixel values in consistent physically defined ROIs. Estimate local background using an annulus or nearby background regions while excluding neighboring features. Avoid clipping ROIs; record per-element signal, background, saturation and validity. Use fractional-pixel sampling only after validating its interpolation bias.
4. Compare observed values with the specified expected responses in their known positions. Fit an appropriate model, report slope, intercept, R² and per-element residuals; consider weighted fitting only with a defensible noise model. Do not sort values as a substitute for spatial identification. If no validated expected response is available, label the output as a descriptive pattern-intensity score rather than detector linearity.
5. Use quality criteria independent of the resulting R². Reject insufficient matches, clipping or saturation; report ambiguous orientation and low signal. Determine limits from data, not arbitrary examples.

Illumination changes as the target moves across the field can genuinely change the measured signal. An optional flat-field correction from an existing suitable uniformity image may be evaluated only for comparable optics, channel and acquisition settings; retain and report both corrected and uncorrected results and avoid circular normalization. There may be insufficient information in existing data to recover absolute detector linearity or separate illumination and detector effects.

## 5. Image classification and routing

Add an explicit analysis override, then consider trusted structured metadata, image evidence and finally filename as a weak hint. Never let a filename alone override contradictory strong evidence without a warning. Classification should return ranked candidates, evidence and uncertainty, not only a string. Candidate detectors can use:

| Candidate | Useful evidence | Important ambiguity |
| --- | --- | --- |
| 2D rings | Repeated roughly square lattice, expected pitch and ring/dot morphology | Partial field, out-of-focus image or another periodic target. |
| 4×4 intensity | A 4×4 layout with expected pitch, geometry and intensity structure | A small crop of another lattice; do not require a particular brightness order until specified. |
| Star | Radial spoke structure and rotational symmetry | A cropped or defocused star may be ambiguous. |
| PSF beads | Sparse compact objects, often with a suitable Z stack | Other isolated particles; Z dimension alone is insufficient. |
| Plastic field | Smooth, broadly distributed signal with little structured pattern | Defocused or low-signal images can look similar. |
| Crossing stairs | Recognizable crossing/stair geometry plus Z/T context | Z/T axes select a compatible analysis but do not establish pattern identity. |

Example interface:

```python
def classify_image(image, metadata, name=None, override=None):
    # Override is explicit and validated against supported categories.
    # Otherwise combine independent evidence; scores need dataset calibration.
    return {
        "category": "Unknown",
        "confidence": None,
        "candidates": [],
        "evidence": [],
        "requires_review": True,
    }
```

Only call a number a probability if calibrated on representative held-out data. Establish score and margin thresholds using existing labeled images; include unknown and deliberately misleading filenames. If uncertain or conflicting, return a structured result requesting manual selection rather than dispatching automatically. Extend the existing API carefully: `classify(name, metadata)` currently has no image argument, and some callers may depend on its string return and `metrology_splitter` returning `False` for unknown.

## 6. Validation with existing images

Create a manifest of files under `tests/data/` and `upload/` recording expected pattern, relevant metadata, channel, dimensions and any usable manual annotations. Check for repeated or near-duplicate acquisitions before splitting data; do not claim independent instrument repeats without confirming acquisition history. Keep image files and sensitive metadata handling consistent with repository policy.

Run the existing pipeline first and save a machine-readable baseline: status, all reported values, selected plane, runtime and warnings. Then compare each proposed change on the same files. Include failure rates and review overlays, not just averages. Close figures during batch processing. Distinguish actual repeat acquisitions from images of different objectives, microscopes or settings; those should not be pooled as identical repeats.

Apply deterministic digital perturbations to copies using recorded seeds and parameters:

- Small integer and subpixel translations and rotations, with and without edge clipping.
- Modest brightness, background and noise variations, and deliberately saturated examples.
- Z-plane perturbations or selection of neighboring acquired planes where available.
- Missing/extra detections and misleading filenames for classifier and matching tests.

Interpolation itself can change PSF, RSF and intensity. Measure the transformation-only effect and use integer shifts where possible; never interpret digital perturbation results as real microscope repeatability. For images that retain the same content, compare displacement in physical units, relative metric changes, selected planes, matched counts and failure rates. For clipped content or target movement across a real illumination field, require correct warnings and coverage reporting rather than unrealistic invariance. Choose numerical tolerances from baseline variability, expected measurement precision and intended use, and document them before treating tests as release gates.

### Tests to add

- Synthetic RSF stack: correct crop under translation and rotation, coordinate conversion, edge rejection.
- Lattice: one-to-one pairing, outliers, missing features, rotation, partial fields and ambiguous grid shifts.
- Focus: deterministic choice, invalid planes, noise/saturation cases, no selection by target metric.
- Distortion: known displacement field, unit conversion, outlier effect and robust summaries.
- Coregistration: known global translation plus local shift; confirm neither is silently fitted away.
- Uniformity: clipped ROIs are invalid, not artificially dark; camera-frame coverage is reported.
- Intensity: known element identities and values, orientation variants, background and saturation; verify spatial assignments and expected-response regression.
- Classifier: correct pattern, unknowns, conflicting filename/metadata, override, partial and degraded images.
- End-to-end CZI tests: assert meaningful known results or explicit quality failures, rather than accepting either success or error unconditionally. Preserve star and PSF regression results within justified tolerances.

## Implementation sequence and acceptance

1. Establish dataset manifest and baseline; record environment and dependencies.
2. Fix RSF coordinate frames and add focused synthetic tests.
3. Implement image validation, focus selection and shared one-to-one lattice matching with diagnostics.
4. Migrate distortion, coregistration and ring uniformity one at a time; compare old and new values and document definition changes.
5. Implement intensity element identification and measurements after target-specification verification.
6. Add classification with uncertainty and manual override; retain a compatibility path.
7. Run retrospective perturbation and regression tests, review overlays, set evidence-based quality gates, and update README and downstream result consumers.

Completion means that coordinate and matching tests pass, known failures are surfaced, results and provenance are reproducible for a fixed input and configuration, compatibility changes are documented, and the existing dataset has been reviewed with quantitative before/after comparisons. Any unresolved ambiguity in target specification, field correction or expected response must remain explicit in the output and documentation.
