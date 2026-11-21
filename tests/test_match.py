"""Red tests for the video-watermark-detect bugfix task.

The data shapes used by this suite are::

    frame    = [[float, ...], ...]        # frame[y][x], grayscale 0..1
    template = [[float, ...], ...]        # same shape and range
    pos      = (x, y)                     # integer top-left pixel

All coordinates are pixels, all grayscale samples are in ``0.0..1.0`` and
every floating assertion uses an absolute tolerance of ``1e-6``.  These tests
are the acceptance baseline and must not be modified: fix
``logodetect/match.py`` instead.  Nothing here uses randomness, sleeps,
network access or third-party packages.
"""

import pytest

import logodetect.match as match_mod
from logodetect.match import (
    LogoError,
    detect_batch,
    estimate_opacity,
    match,
    ncc_at,
    nms,
    normalize,
    temporal_vote,
    validate,
)


def _frame(width, height, value=0.0):
    return [[float(value) for _ in range(width)] for _ in range(height)]


def _tpl(rows):
    return [list(row) for row in rows]


def _assert_matrix_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert actual_value == pytest.approx(expected_value, abs=1e-6)


def _assert_matches_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_match, expected_match in zip(actual, expected):
        assert actual_match["x"] == expected_match["x"]
        assert actual_match["y"] == expected_match["y"]
        assert actual_match["score"] == pytest.approx(
            expected_match["score"], abs=1e-6
        )


def test_normalize_uses_population_std():
    result = normalize([[1.0, 2.0], [3.0, 4.0]])

    _assert_matrix_close(
        result,
        [
            [-1.3416407864998738, -0.4472135954999579],
            [0.4472135954999579, 1.3416407864998738],
        ],
    )

    # A constant frame must not divide by a zero standard deviation.
    flat = normalize([[0.25, 0.25], [0.25, 0.25]])
    _assert_matrix_close(flat, [[0.0, 0.0], [0.0, 0.0]])


def test_ncc_at_normalizes_by_template_energy():
    # The bright patch is an affine rescaling of the template, so the only
    # correct normalized correlation is 1.0.  Dropping the template energy
    # from the denominator makes this 0.7071... instead.
    score = ncc_at([[0.6, 1.0]], [[0.0, 1.0]], 0, 0)

    assert -1.0 <= score <= 1.0
    assert score == pytest.approx(1.0, abs=1e-6)


def test_match_uses_coarse_stride_on_large_frame(monkeypatch):
    calls = []

    def fake_ncc(img, tpl, x, y):
        calls.append((x, y))
        return 0.0

    monkeypatch.setattr(match_mod, "ncc_at", fake_ncc)

    frame = _frame(512, 512, 0.0)
    template = _tpl([[0.0] * 40 for _ in range(40)])

    assert match(frame, template, 0.5) == []
    # A fixed step-1 scan would make 473 * 473 == 223729 calls.  A stride tied
    # to the 40x40 template must stay far below that.
    assert len(calls) < 5000


def test_nms_suppresses_overlapping_duplicates():
    boxes = [
        {"x": 10, "y": 20, "w": 5, "h": 5, "score": 0.95},
        {"x": 11, "y": 20, "w": 5, "h": 5, "score": 0.90},
        {"x": 40, "y": 20, "w": 5, "h": 5, "score": 0.80},
    ]

    kept = nms(boxes, 0.3)

    assert [(box["x"], box["y"]) for box in kept] == [(10, 20), (40, 20)]
    # The overlap must be suppressed, not deleted from the caller's list.
    assert len(boxes) == 3
    assert boxes[1]["x"] == 11


def test_temporal_vote_prefers_stable_region_over_transient_burst():
    frames = [_frame(4, 4, 0.0) for _ in range(3)]
    stable = {"x": 1, "y": 1, "score": 0.55}
    transient = {"x": 2, "y": 2, "score": 0.99}

    matches = [
        [dict(stable)] + [dict(transient) for _ in range(20)],
        [dict(stable)],
        [dict(stable)],
    ]

    result = temporal_vote(matches, frames)

    assert len(result) == 1
    assert result[0]["x"] == 1
    assert result[0]["y"] == 1
    assert result[0]["votes"] == 3
    assert result[0]["score"] == pytest.approx(0.55, abs=1e-6)


def test_estimate_opacity_is_background_aware():
    # With background 0.2 and alpha 0.5, a 0/1 template becomes 0.1/0.6.
    # A color ratio that ignores the background cannot recover alpha = 0.5.
    image = [[0.6, 0.1], [0.6, 0.1]]
    template = [[1.0, 0.0], [1.0, 0.0]]

    alpha = estimate_opacity(image, template, (0, 0))

    assert alpha == pytest.approx(0.5, abs=1e-6)
    assert 0.0 <= alpha <= 1.0


def test_validate_reports_template_larger_than_image():
    image = _frame(2, 2, 0.0)
    template = _tpl([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    assert validate(image, template) == ["template larger than image"]

    with pytest.raises(LogoError):
        ncc_at(image, template, 0, 0)


def test_validate_rejects_nan_instead_of_nan_scores():
    image = [[float("nan"), 0.0], [0.0, 0.0]]
    template = _tpl([[0.0, 1.0], [1.0, 0.0]])

    assert validate(image, template) == ["image contains NaN"]

    with pytest.raises(LogoError):
        ncc_at(image, template, 0, 0)


def test_detect_batch_isolates_each_clip():
    template = _tpl([[0.0, 1.0], [1.0, 0.0]])
    clip_a = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    clip_b = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    config = {"template": template, "threshold": 0.9, "nms_iou": 0.3}

    batch = detect_batch([clip_a, clip_b], config)
    alone = [detect_batch([clip], config)["results"][0]["matches"] for clip in (clip_a, clip_b)]

    assert batch["count"] == 2
    for index, expected in enumerate(alone):
        assert batch["results"][index]["clip"] == index
        _assert_matches_close(batch["results"][index]["matches"], expected)


def test_detect_batch_rejects_threshold_out_of_range():
    config = {
        "template": _tpl([[0.0, 1.0]]),
        "threshold": 1.5,
        "nms_iou": 0.3,
    }

    with pytest.raises(LogoError):
        detect_batch([_frame(4, 4, 0.0)], config)


def test_identical_template_at_origin_scores_one():
    image = _tpl([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    template = _tpl(image)

    score = ncc_at(image, template, 0, 0)

    assert score == pytest.approx(1.0, abs=1e-6)


def test_all_black_frame_scores_zero():
    image = _frame(2, 2, 0.0)
    template = _frame(2, 2, 0.0)

    assert ncc_at(image, template, 0, 0) == pytest.approx(0.0, abs=1e-6)


def test_match_rejects_threshold_out_of_range():
    with pytest.raises(LogoError):
        match(_frame(4, 4, 0.0), _tpl([[0.0, 1.0]]), 1.5)


def test_match_returns_empty_when_no_window_passes():
    frame = _frame(4, 4, 0.0)
    template = _tpl([[0.0, 1.0], [1.0, 0.0]])

    assert match(frame, template, 0.5) == []


def test_detect_batch_preserves_order_and_rolls_back():
    clips = [
        [[0.0, 1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0]],
    ]
    config = {
        "template": _tpl([[0.0, 1.0]]),
        "threshold": 0.9,
        "nms_iou": 0.3,
    }

    result = detect_batch(clips, config)

    assert result["count"] == 3
    assert [item["clip"] for item in result["results"]] == [0, 1, 2]
    assert [item["matches"][0]["x"] for item in result["results"]] == [0, 1, 2]

    before = [row[:] for row in clips[0]]
    bad_clip = [[0.0, float("nan"), 0.0, 0.0]]
    with pytest.raises(LogoError):
        detect_batch([clips[0], bad_clip], config)
    assert clips[0] == before
