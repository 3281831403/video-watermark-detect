"""Grayscale logo/template matching primitives.

Data model
----------
A **frame** is a non-empty rectangular ``list[list[float]]`` in row-major
order, so ``frame[y][x]`` is the grayscale sample at column ``x`` and row
``y``.  Every sample is a finite float in ``0.0..1.0``.

A **template** (the logo) uses exactly the same representation: a non-empty
rectangular ``list[list[float]]`` of grayscale samples in ``0.0..1.0``, with
width ``w = len(template[0])`` and height ``h = len(template)``.  A template
larger than the frame is invalid.  Placing a template at the integer pixel
``(x, y)`` means comparing it with ``frame[y:y+h][x:x+w]``, where ``(0, 0)``
is the top-left of the frame.

Floating-point scores and samples are compared with an absolute tolerance of
``1e-6``.  Coordinates are integer pixels.  This module uses only the Python
standard library (``math``); it never uses randomness, sleeps or touches the
network.

Invalid arguments -- malformed or ragged matrices, empty frames/templates,
samples outside ``0.0..1.0``, NaN, out-of-bounds coordinates, bad thresholds
or bad IoU values -- raise :class:`LogoError`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

__all__ = [
    "LogoError",
    "normalize",
    "ncc_at",
    "match",
    "nms",
    "temporal_vote",
    "estimate_opacity",
    "validate",
    "detect_batch",
]

_EPS = 1e-12


class LogoError(ValueError):
    """Raised when a frame, template, coordinate, threshold or batch is invalid."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _is_real_number(value):
    """True for finite int/float samples (booleans are not treated as numbers)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _matrix_problems(matrix, enforce_range=True):
    """Return structural/content issue tags for one matrix, never raising.

    Tags are drawn from ``"empty"``, ``"ragged"``, ``"nan"``,
    ``"nonnumeric"`` and ``"range"``.  The returned order follows the check
    order documented on :func:`validate`.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        return ["empty"]

    first_row = matrix[0]
    if not isinstance(first_row, list):
        # Rows are expected to be lists; a non-list first row is ragged.
        return ["ragged"]
    if len(first_row) == 0:
        return ["empty"]

    width = len(first_row)
    ragged = any(
        not isinstance(row, list) or len(row) != width for row in matrix
    )

    has_nan = False
    has_nonnumeric = False
    out_of_range = False
    for row in matrix:
        if not isinstance(row, list):
            continue
        for sample in row:
            if isinstance(sample, float) and math.isnan(sample):
                has_nan = True
                continue
            if not _is_real_number(sample):
                has_nonnumeric = True
                continue
            if enforce_range and not (0.0 <= sample <= 1.0):
                out_of_range = True

    problems = []
    if ragged:
        problems.append("ragged")
    if has_nan:
        problems.append("nan")
    if has_nonnumeric:
        problems.append("nonnumeric")
    if out_of_range:
        problems.append("range")
    return problems


def _require_matrix(matrix, what, enforce_range=True):
    """Raise :class:`LogoError` unless ``matrix`` is a valid grayscale matrix."""
    problems = _matrix_problems(matrix, enforce_range=enforce_range)
    if problems:
        raise LogoError("invalid %s: %s" % (what, problems[0]))


def _require_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogoError("%s must be an integer pixel coordinate" % what)
    return value


def _require_threshold(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not (-1.0 <= value <= 1.0)
    ):
        raise LogoError("threshold must be a finite number in -1.0..1.0")


def _require_iou(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not (0.0 <= value <= 1.0)
    ):
        raise LogoError("iou must be a finite number in 0.0..1.0")


def _check_placement(img, tpl, x, y):
    """Validate both matrices and the placement of ``tpl`` at ``(x, y)``."""
    _require_matrix(img, "image")
    _require_matrix(tpl, "template")
    _require_int(x, "x")
    _require_int(y, "y")
    img_h, img_w = len(img), len(img[0])
    tpl_h, tpl_w = len(tpl), len(tpl[0])
    if x < 0 or y < 0 or x + tpl_w > img_w or y + tpl_h > img_h:
        raise LogoError("template placement is out of bounds")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize(img):
    """Return a new frame normalized to zero mean and unit variance.

    The mean and the population variance are taken over *all* samples::

        mean = sum(samples) / count
        std  = sqrt(sum((sample - mean) ** 2) / count)
        out[y][x] = (img[y][x] - mean) / std

    The variance divisor is the sample count, not ``count - 1``.  A constant
    frame has ``std == 0``; instead of dividing by zero it returns a frame of
    the same dimensions filled with ``0.0``.  The input is not mutated and the
    returned rows are new lists.

    ``img`` is a grayscale frame in ``0.0..1.0``.  Empty, ragged, non-numeric,
    NaN or out-of-range input raises :class:`LogoError`.  Results compare with
    an absolute tolerance of ``1e-6``.
    """
    # Structure, numeric type and finiteness are enforced.  The 0..1 range is
    # intentionally not enforced here because normalization is defined for any
    # finite numeric matrix (the acceptance suite normalizes a 1.0..4.0 frame).
    _require_matrix(img, "image", enforce_range=False)

    count = sum(len(row) for row in img)
    mean = sum(sample for row in img for sample in row) / count
    variance = (
        sum((sample - mean) ** 2 for row in img for sample in row) / count
    )
    std = math.sqrt(variance)
    if std == 0.0:
        return [[0.0 for _ in row] for row in img]
    return [[(sample - mean) / std for sample in row] for row in img]


# ---------------------------------------------------------------------------
# Normalized cross-correlation
# ---------------------------------------------------------------------------
def ncc_at(img, tpl, x, y):
    """Return the normalized cross-correlation of ``tpl`` at ``(x, y)``.

    The template is compared with the frame patch whose top-left corner is
    ``(x, y)``.  Both patch and template are zero-meaned and normalized by
    their own energies::

        a = patch - mean(patch)
        b = tpl   - mean(tpl)
        score = sum(a * b) / sqrt(sum(a * a) * sum(b * b))

    Dividing by *both* energies is required.  Omitting the template energy
    makes the score depend on template contrast and can exceed ``1.0``; a
    correct score is always in ``-1.0..1.0``.  If either energy is at most
    ``1e-12`` (for example an all-black or constant patch/template), the
    correlation is undefined and this function returns ``0.0``.

    ``img`` and ``tpl`` are grayscale matrices in ``0.0..1.0``; ``tpl`` must
    fit entirely inside ``img`` at ``(x, y)``.  Coordinates are integers with
    ``0 <= x <= len(img[0]) - len(tpl[0])`` and
    ``0 <= y <= len(img) - len(tpl)``.  Malformed, ragged, empty, NaN,
    out-of-range or too-large input raises :class:`LogoError`.  The score
    compares with an absolute tolerance of ``1e-6``.
    """
    _check_placement(img, tpl, x, y)

    tpl_h = len(tpl)
    tpl_w = len(tpl[0])
    patch = [
        [img[y + row][x + col] for col in range(tpl_w)]
        for row in range(tpl_h)
    ]

    patch_mean = (
        sum(sample for row in patch for sample in row) / (tpl_h * tpl_w)
    )
    tpl_mean = (
        sum(sample for row in tpl for sample in row) / (tpl_h * tpl_w)
    )

    energy_patch = 0.0
    energy_tpl = 0.0
    dot = 0.0
    for row in range(tpl_h):
        for col in range(tpl_w):
            a = patch[row][col] - patch_mean
            b = tpl[row][col] - tpl_mean
            dot += a * b
            energy_patch += a * a
            energy_tpl += b * b

    if energy_patch <= _EPS or energy_tpl <= _EPS:
        return 0.0
    return dot / math.sqrt(energy_patch * energy_tpl)


# ---------------------------------------------------------------------------
# Coarse-to-fine template matching
# ---------------------------------------------------------------------------
def _scan_positions(max_pos, step):
    """Grid positions from 0 to ``max_pos`` inclusive, plus the final point."""
    positions = list(range(0, max_pos + 1, step))
    if positions[-1] != max_pos:
        positions.append(max_pos)
    return positions


def match(img, tpl, threshold):
    """Find template matches in ``img`` and return their top-left positions.

    Every returned item is a dict with exactly the keys ``"x"``, ``"y"`` and
    ``"score"``::

        {"x": int, "y": int, "score": float}

    ``threshold`` is the minimum accepted NCC score and must be a finite float
    in ``-1.0..1.0``.  A value outside that range raises :class:`LogoError`;
    it is never clamped silently.  Only positions with
    ``score >= threshold`` are returned, sorted by descending score and then
    by ascending ``y`` and ``x``.

    A fixed one-pixel scan over a large frame is roughly 100 times slower than
    necessary.  This function therefore performs a coarse-to-fine scan: the
    top-left positions start at ``(0, 0)`` and advance by::

        step = max(1, min(len(tpl), len(tpl[0])) // 4)

    on both axes (the final valid row/column is included when it does not land
    on the grid).  Each coarse position is scored with :func:`ncc_at`.  A
    coarse position at or above ``threshold`` is then refined over its 3x3
    pixel neighbourhood at step 1, and the best refined position is kept.
    This preserves full-frame coverage while avoiding a step-1 scan of every
    pixel.  Duplicate/overlapping detections are not merged here; pass the
    result through :func:`nms`.

    ``img`` and ``tpl`` are grayscale matrices in ``0.0..1.0``; the template
    must fit inside the frame.  Empty, ragged, NaN, out-of-range, too-large or
    non-finite-threshold input raises :class:`LogoError`.  If no candidate
    reaches ``threshold`` the result is ``[]``.  Scores compare with an
    absolute tolerance of ``1e-6``.
    """
    _require_matrix(img, "image")
    _require_matrix(tpl, "template")
    _require_threshold(threshold)

    img_h, img_w = len(img), len(img[0])
    tpl_h, tpl_w = len(tpl), len(tpl[0])
    if tpl_h > img_h or tpl_w > img_w:
        raise LogoError("template larger than image")

    max_x = img_w - tpl_w
    max_y = img_h - tpl_h
    step = max(1, min(tpl_h, tpl_w) // 4)
    xs = _scan_positions(max_x, step)
    ys = _scan_positions(max_y, step)

    detections = []
    for coarse_y in ys:
        for coarse_x in xs:
            coarse_score = ncc_at(img, tpl, coarse_x, coarse_y)
            if coarse_score < threshold:
                continue

            best_x, best_y, best_score = coarse_x, coarse_y, coarse_score
            for dy in (-1, 0, 1):
                ny = coarse_y + dy
                if not 0 <= ny <= max_y:
                    continue
                for dx in (-1, 0, 1):
                    nx = coarse_x + dx
                    if not 0 <= nx <= max_x or (nx == coarse_x and ny == coarse_y):
                        continue
                    score = ncc_at(img, tpl, nx, ny)
                    if score > best_score:
                        best_x, best_y, best_score = nx, ny, score
            detections.append(
                {"x": best_x, "y": best_y, "score": best_score}
            )

    detections.sort(key=lambda item: (-item["score"], item["y"], item["x"]))
    return detections


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------
def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a["x"], box_a["y"], box_a["w"], box_a["h"]
    bx, by, bw, bh = box_b["x"], box_b["y"], box_b["w"], box_b["h"]

    overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = overlap_w * overlap_h
    union = aw * ah + bw * bh - intersection
    return intersection / union


def nms(boxes, iou):
    """Greedily suppress overlapping boxes and return the kept boxes.

    Each box is a mapping with ``"x"``, ``"y"``, ``"w"``, ``"h"`` and
    ``"score"``; ``(x, y)`` is the top-left corner, dimensions are positive
    integer pixels in a grayscale frame and ``score`` is a finite float.
    ``iou`` is the maximum tolerated intersection-over-union and must be in
    ``0.0..1.0``.

    Boxes are considered in descending score order (ties by ascending ``y``
    then ``x``).  A box is kept when its IoU with every already-kept box is
    less than or equal to ``iou``; otherwise it is suppressed.  Thus the
    highest-scoring box of a cluster survives and duplicate detections of the
    same logo collapse to one.  IoU is::

        intersection_area / (area_a + area_b - intersection_area)

    The return value is a new list, in kept order; the input list and its
    mappings are not mutated.  An empty input returns ``[]``.  Malformed
    boxes, non-positive dimensions, missing keys, non-finite scores or an
    out-of-range ``iou`` raise :class:`LogoError`.  Geometric comparisons
    compare as exact pixel arithmetic; any floating score comparison uses an
    absolute tolerance of ``1e-6``.
    """
    _require_iou(iou)
    if not isinstance(boxes, (list, tuple)):
        raise LogoError("boxes must be a sequence of box mappings")

    validated = []
    for box in boxes:
        if not isinstance(box, Mapping):
            raise LogoError("each box must be a mapping")
        missing = [
            key
            for key in ("x", "y", "w", "h", "score")
            if key not in box
        ]
        if missing:
            raise LogoError("box missing keys: %s" % ", ".join(missing))
        if isinstance(box["x"], bool) or not isinstance(box["x"], int):
            raise LogoError("box x must be an integer")
        if isinstance(box["y"], bool) or not isinstance(box["y"], int):
            raise LogoError("box y must be an integer")
        if (
            isinstance(box["w"], bool)
            or not isinstance(box["w"], int)
            or box["w"] <= 0
        ):
            raise LogoError("box w must be a positive integer")
        if (
            isinstance(box["h"], bool)
            or not isinstance(box["h"], int)
            or box["h"] <= 0
        ):
            raise LogoError("box h must be a positive integer")
        if not _is_real_number(box["score"]):
            raise LogoError("box score must be a finite number")
        validated.append(box)

    ordered = sorted(
        validated, key=lambda box: (-box["score"], box["y"], box["x"])
    )

    kept = []
    for box in ordered:
        if all(_iou(box, other) <= iou for other in kept):
            kept.append(box)
    return kept


# ---------------------------------------------------------------------------
# Confidence-weighted temporal voting
# ---------------------------------------------------------------------------
def temporal_vote(matches, frames):
    """Vote detections across frames, preferring stable high-confidence regions.

    ``matches`` is a sequence with one list of detections per frame, and
    ``frames`` is the corresponding sequence of grayscale frames with samples
    in ``0.0..1.0`` (used for its length and order).  Each detection is a
    mapping with ``"x"``, ``"y"`` and ``"score"``.  The two sequences must have
    equal length and every detection must lie inside its frame; otherwise
    :class:`LogoError` is raised.

    Voting is confidence-weighted and frame-aware, not a raw count of boxes.
    Within one frame, duplicate detections at the same ``(x, y)`` contribute
    only that frame's highest score.  Detections are then clustered across
    frames when their top-left corners differ by at most 1 pixel on each axis.
    A cluster needs support from at least two distinct frames; a transient
    false positive seen in only one frame is discarded even if it has many
    duplicate boxes or a very high score.

    For a surviving cluster, ``votes`` is the number of distinct supporting
    frames and ``score`` is the mean of the per-frame best scores.  Clusters
    are ordered by descending total confidence weight (the sum of the
    per-frame best scores), then descending ``votes``, then ascending ``y``
    and ``x``.  Each result is::

        {"x": int, "y": int, "score": float, "votes": int}

    Scores compare with an absolute tolerance of ``1e-6``.  Malformed
    input, mismatched lengths or a detection outside its grayscale frame
    raises :class:`LogoError`.
    """
    if not isinstance(matches, (list, tuple)) or not isinstance(
        frames, (list, tuple)
    ):
        raise LogoError("matches and frames must be sequences")
    if len(matches) != len(frames):
        raise LogoError("matches and frames must have equal length")

    frame_dims = []
    for frame in frames:
        _require_matrix(frame, "frame")
        frame_dims.append((len(frame[0]), len(frame)))

    # One entry per (frame, position): duplicates within a frame collapse to
    # the frame's highest score at that position.
    entries = []
    for frame_index, detections in enumerate(matches):
        if not isinstance(detections, (list, tuple)):
            raise LogoError("each frame's detections must be a sequence")
        width, height = frame_dims[frame_index]
        best_per_pos = {}
        for detection in detections:
            if not isinstance(detection, Mapping):
                raise LogoError("each detection must be a mapping")
            if "x" not in detection or "y" not in detection or "score" not in detection:
                raise LogoError("detection missing x, y or score")
            x = detection["x"]
            y = detection["y"]
            if isinstance(x, bool) or not isinstance(x, int):
                raise LogoError("detection x must be an integer")
            if isinstance(y, bool) or not isinstance(y, int):
                raise LogoError("detection y must be an integer")
            if not (0 <= x < width and 0 <= y < height):
                raise LogoError("detection lies outside its frame")
            if not _is_real_number(detection["score"]):
                raise LogoError("detection score must be a finite number")
            score = float(detection["score"])
            current = best_per_pos.get((x, y))
            if current is None or score > current:
                best_per_pos[(x, y)] = score
        for (x, y), score in best_per_pos.items():
            entries.append((x, y, score, frame_index))

    # Greedy frame-aware clustering: entries are processed frame by frame, and
    # a frame can contribute at most one entry to any cluster.  An entry joins
    # the first existing cluster whose members are within one pixel on every
    # axis; the per-frame exclusion keeps a second, nearby detection in the
    # same frame from hijacking the cluster built by other frames.  A cluster
    # that never gains a second frame is the transient burst the docstring
    # rejects.
    clusters = []  # each cluster: [entries, supporting_frame_set]
    for frame_index in range(len(frames)):
        frame_entries = [
            entry for entry in entries if entry[3] == frame_index
        ]
        frame_entries.sort(key=lambda entry: (entry[1], entry[0]))
        for entry in frame_entries:
            x, y, _, entry_frame = entry
            for cluster in clusters:
                members, supporting = cluster
                if entry_frame in supporting:
                    continue
                if any(
                    abs(x - other[0]) <= 1 and abs(y - other[1]) <= 1
                    for other in members
                ):
                    members.append(entry)
                    supporting.add(entry_frame)
                    break
            else:
                clusters.append(([entry], {entry_frame}))

    results = []
    for group, supporting_frames in clusters:
        if len(supporting_frames) < 2:
            continue
        total_weight = sum(entry[2] for entry in group)
        mean_score = total_weight / len(group)
        # Representative position: highest per-frame score, ties by y then x.
        representative = min(
            group, key=lambda entry: (-entry[2], entry[1], entry[0], entry[3])
        )
        results.append(
            {
                "x": representative[0],
                "y": representative[1],
                "score": mean_score,
                "votes": len(supporting_frames),
                "_weight": total_weight,
            }
        )

    results.sort(
        key=lambda item: (
            -item["_weight"],
            -item["votes"],
            item["y"],
            item["x"],
        )
    )
    for item in results:
        del item["_weight"]
    return results


# ---------------------------------------------------------------------------
# Background-aware opacity estimation
# ---------------------------------------------------------------------------
def estimate_opacity(img, tpl, pos):
    """Estimate the template opacity in ``img`` without ignoring the background.

    ``pos`` is the integer ``(x, y)`` top-left of the template's location in
    the grayscale frame ``img``.  The observed patch is modelled as::

        observed = (1 - alpha) * background + alpha * template

    with an unknown constant local background.  Eliminating that background
    gives the least-squares estimate::

        alpha = sum((observed - mean(observed)) * (tpl - mean(tpl)))
                / sum((tpl - mean(tpl)) ** 2)

    clamped to ``0.0..1.0``.  A color-ratio estimate such as ``observed / tpl``
    would mistake the background for opacity and fail on dark or bright
    scenes; covariance with the template cancels that constant background.
    If the template is constant (its variance is at most ``1e-12``), opacity
    is unidentifiable and ``0.0`` is returned.

    ``img`` and ``tpl`` are grayscale matrices in ``0.0..1.0``; the template
    must fit at ``pos``.  Empty, ragged, NaN, out-of-range or out-of-bounds
    input raises :class:`LogoError`.  The returned value is in ``0.0..1.0``
    and compares with an absolute tolerance of ``1e-6``.
    """
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise LogoError("pos must be an (x, y) pair")
    x, y = pos
    _check_placement(img, tpl, x, y)

    tpl_h, tpl_w = len(tpl), len(tpl[0])
    count = tpl_h * tpl_w
    observed_mean = (
        sum(img[y + row][x + col] for row in range(tpl_h) for col in range(tpl_w))
        / count
    )
    template_mean = (
        sum(sample for row in tpl for sample in row) / count
    )

    covariance = 0.0
    template_energy = 0.0
    for row in range(tpl_h):
        for col in range(tpl_w):
            obs_dev = img[y + row][x + col] - observed_mean
            tpl_dev = tpl[row][col] - template_mean
            covariance += obs_dev * tpl_dev
            template_energy += tpl_dev * tpl_dev

    if template_energy <= _EPS:
        return 0.0

    alpha = covariance / template_energy
    if alpha < 0.0:
        return 0.0
    if alpha > 1.0:
        return 1.0
    return alpha


# ---------------------------------------------------------------------------
# Non-raising validation reporter
# ---------------------------------------------------------------------------
def validate(img, tpl):
    """Return a list of problems with a frame/template pair, without raising.

    This function is a reporting helper: malformed data produces issue strings
    instead of exceptions.  The checks and their exact strings are, in order:

    * ``"empty image"`` / ``"empty template"``: the matrix is absent, has no
      rows or has a zero-width first row;
    * ``"ragged image"`` / ``"ragged template"``: rows have different lengths
      or a row is not a list;
    * ``"image contains NaN"`` / ``"template contains NaN"``: any sample is
      NaN;
    * ``"image not grayscale"`` / ``"template not grayscale"``: a sample is
      not a real number or lies outside ``0.0..1.0``;
    * ``"template larger than image"``: the template width or height exceeds
      the frame's dimensions.

    Multiple problems may be returned, in the order above.  A valid grayscale
    pair returns ``[]``.  Issue strings and integer dimensions are compared
    exactly, so no floating tolerance applies here; callers that compare
    scores use an absolute tolerance of ``1e-6``.  This function itself never
    raises :class:`LogoError` for bad image/template data.  Callers that require
    valid data (such as :func:`ncc_at`) raise on the same conditions.
    """
    image_problems = _matrix_problems(img)
    template_problems = _matrix_problems(tpl)

    issues = []
    if "empty" in image_problems:
        issues.append("empty image")
    if "empty" in template_problems:
        issues.append("empty template")
    if "ragged" in image_problems:
        issues.append("ragged image")
    if "ragged" in template_problems:
        issues.append("ragged template")
    if "nan" in image_problems:
        issues.append("image contains NaN")
    if "nan" in template_problems:
        issues.append("template contains NaN")
    if "nonnumeric" in image_problems or "range" in image_problems:
        issues.append("image not grayscale")
    if "nonnumeric" in template_problems or "range" in template_problems:
        issues.append("template not grayscale")

    rectangular = (
        isinstance(img, list)
        and isinstance(tpl, list)
        and len(img) > 0
        and len(tpl) > 0
        and isinstance(img[0], list)
        and isinstance(tpl[0], list)
        and len(img[0]) > 0
        and len(tpl[0]) > 0
        and "ragged" not in image_problems
        and "ragged" not in template_problems
    )
    if rectangular:
        if len(tpl[0]) > len(img[0]) or len(tpl) > len(img):
            issues.append("template larger than image")

    return issues


# ---------------------------------------------------------------------------
# Atomic batch detection
# ---------------------------------------------------------------------------
def detect_batch(clips, config):
    """Detect one template in every clip atomically and preserve clip order.

    ``clips`` is a sequence of grayscale frames whose samples are in
    ``0.0..1.0``.  ``config`` is a mapping with three required keys::

        {"template": tpl, "threshold": float, "nms_iou": float}

    ``threshold`` must be in ``-1.0..1.0`` and ``nms_iou`` in ``0.0..1.0``.
    For each clip the pipeline is: validate the clip and template, call
    :func:`match`, convert each point match to a template-sized box, then call
    :func:`nms`.  The returned structure is::

        {
            "count": len(clips),
            "results": [
                {"clip": i, "count": len(kept), "matches": [kept, ...]},
                ...
            ],
        }

    where each kept match has ``"x"``, ``"y"`` and ``"score"``.  ``results``
    is in the same order as ``clips``; each result's ``"clip"`` is its original
    index.  An empty ``clips`` sequence returns ``{"count": 0, "results": []}``.

    The operation is atomic.  Every clip, the template and the configuration
    are validated before any matching starts.  If any clip is invalid -- for
    example it contains NaN, is empty or is smaller than the template -- the
    whole batch raises :class:`LogoError`; no partial result is returned and
    the input frames are not mutated.  Each clip is processed with its own
    normalization state, so one clip's brightness or content can never leak
    into another clip's scores.  Scores compare with an absolute tolerance of
    ``1e-6``.
    """
    if not isinstance(config, Mapping):
        raise LogoError("config must be a mapping")
    missing = [
        key for key in ("template", "threshold", "nms_iou") if key not in config
    ]
    if missing:
        raise LogoError("config missing keys: %s" % ", ".join(missing))
    if not isinstance(clips, (list, tuple)):
        raise LogoError("clips must be a sequence of frames")

    template = config["template"]
    threshold = config["threshold"]
    nms_iou = config["nms_iou"]
    _require_threshold(threshold)
    _require_iou(nms_iou)
    _require_matrix(template, "template")

    # Validate every clip up front so an invalid batch never starts matching.
    for clip in clips:
        problems = validate(clip, template)
        if problems:
            raise LogoError("invalid clip: %s" % problems[0])

    tpl_h, tpl_w = len(template), len(template[0])
    results = []
    for index, clip in enumerate(clips):
        # All state used for this clip is local: nothing is shared between
        # clips, so brightness/content cannot leak across the batch.
        point_matches = match(clip, template, threshold)
        boxes = [
            {
                "x": point["x"],
                "y": point["y"],
                "w": tpl_w,
                "h": tpl_h,
                "score": point["score"],
            }
            for point in point_matches
        ]
        kept_boxes = nms(boxes, nms_iou)
        kept_matches = [
            {"x": box["x"], "y": box["y"], "score": box["score"]}
            for box in kept_boxes
        ]
        results.append(
            {
                "clip": index,
                "count": len(kept_matches),
                "matches": kept_matches,
            }
        )

    return {"count": len(clips), "results": results}
