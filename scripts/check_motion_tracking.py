"""Verify motion-aware plate tracking keeps fast vehicles on a single track.

Motivation: the temporal voter only confirms a plate after ``min_dwell`` reads
accumulated on *one* track, so a tracker that re-IDs a fast vehicle every frame
can never confirm anything -- the vehicle silently disappears from the output.
Overlap-only association breaks a track as soon as the box translates by more
than about a third of its own width between sampled frames, which highway-speed
vehicles exceed routinely.

Each scenario below prints the worst *overlap-only* IoU it produced. When that
number sits below the tracker's ``iou_threshold``, an overlap-only tracker
would have started a new track there -- which is exactly the regression this
script guards against.

Pure stdlib (no pytest, no OpenCV): run with

    python scripts/check_motion_tracking.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr_engine.tracking import IoUTracker, TemporalVoter, iou  # noqa: E402

IOU_THRESHOLD = 0.3
CENTER_DISTANCE = 1.5
VELOCITY_SMOOTHING = 0.5
MAX_SPEED_SCALE = 2.0

VOTER_WINDOW = 8
VOTER_MIN_DWELL = 5
VOTER_TOP_K = 3


def make_tracker():
    return IoUTracker(
        iou_threshold=IOU_THRESHOLD,
        max_age=30,
        max_center_distance=CENTER_DISTANCE,
        velocity_smoothing=VELOCITY_SMOOTHING,
        max_speed_scale=MAX_SPEED_SCALE,
    )


def straight_line(x0, y0, width, height, step_x, frames):
    """Per-frame boxes for a plate translating ``step_x`` px each frame."""
    return [
        (x0 + step_x * i, y0, x0 + step_x * i + width, y0 + height)
        for i in range(frames)
    ]


def test_fast_vehicle_holds_one_track():
    """A plate shifting 65% of its width per frame must stay on one track.

    Overlap-only IoU at that shift is 0.21 -- below the 0.3 threshold -- so the
    previous tracker opened a new track every frame here.
    """
    boxes = straight_line(100, 200, 80, 24, step_x=52, frames=12)
    tracker = make_tracker()
    seen = set()
    worst_overlap = 1.0
    previous = None
    for box in boxes:
        assigned = tracker.update([box])
        assert len(assigned) == 1, f"expected 1 assignment, got {len(assigned)}"
        tid = assigned[0][0]
        seen.add(tid)
        if previous is not None:
            worst_overlap = min(worst_overlap, iou(previous, box))
        previous = box
    assert len(seen) == 1, f"fast vehicle fragmented into {len(seen)} tracks: {seen}"
    assert worst_overlap < IOU_THRESHOLD, (
        "scenario no longer reproduces the overlap-only failure "
        f"(worst IoU {worst_overlap:.3f} >= {IOU_THRESHOLD}); "
        "tighten step_x so the test stays meaningful"
    )
    return f"fast vehicle: 1 track over 12 frames, worst overlap-only IoU {worst_overlap:.3f}"


def test_fast_vehicle_reaches_voter_dwell():
    """The real symptom: a fast vehicle must be able to confirm a plate.

    Blurred pre-convergence reads are withheld (that is what the read floor in
    the pipeline does), then clean reads arrive. With a stable track the voter
    reaches dwell and emits the plate; with a per-frame re-ID it never would.
    """
    boxes = straight_line(100, 200, 80, 24, step_x=52, frames=14)
    tracker = make_tracker()
    voter = TemporalVoter(window=VOTER_WINDOW, min_dwell=VOTER_MIN_DWELL,
                          top_k=VOTER_TOP_K)
    plate = "MH12AB1234"
    confirmed = None
    frames_on_one_track = set()
    for index, box in enumerate(boxes):
        for tid, matched, _predicted in tracker.update([box]):
            frames_on_one_track.add(tid)
            if index < 4:
                continue  # blurred pre-convergence frames, no trustworthy read
            result = voter.observe(tid, plate, 0.9, [0.9] * len(plate))
            if result is not None:
                confirmed = result
    assert len(frames_on_one_track) == 1, (
        f"fast vehicle fragmented into {len(frames_on_one_track)} tracks"
    )
    assert confirmed == plate, f"voter never confirmed the plate, got {confirmed!r}"
    return f"fast vehicle: voter confirmed {confirmed} after dwell on one track"


def test_slow_vehicle_unaffected():
    """A plate shifting 5% of its width per frame keeps one track and confirms."""
    boxes = straight_line(300, 100, 120, 30, step_x=6, frames=20)
    tracker = make_tracker()
    voter = TemporalVoter(window=VOTER_WINDOW, min_dwell=VOTER_MIN_DWELL,
                          top_k=VOTER_TOP_K)
    plate = "KA01MN4321"
    seen = set()
    confirmed = None
    for box in boxes:
        for tid, _matched, _predicted in tracker.update([box]):
            seen.add(tid)
            result = voter.observe(tid, plate, 0.8, [0.8] * len(plate))
            if result is not None:
                confirmed = result
    assert len(seen) == 1, f"slow vehicle fragmented into {len(seen)} tracks: {seen}"
    assert confirmed == plate, f"slow vehicle failed to confirm, got {confirmed!r}"
    return "slow vehicle: 1 track, confirmed (no regression)"


def test_two_vehicles_do_not_swap_tracks():
    """Two plates crossing at different depths must each keep their own ID."""
    frames = []
    for i in range(12):
        near = (200 + 18 * i, 300, 200 + 18 * i + 140, 340)
        far = (600 - 9 * i, 120, 600 - 9 * i + 46, 133)
        frames.append([near, far])
    tracker = make_tracker()
    near_ids, far_ids = set(), set()
    for index, boxes in enumerate(frames):
        for tid, box, _predicted in tracker.update(boxes):
            (near_ids if box[1] > 200 else far_ids).add(tid)
    assert len(near_ids) == 1, f"near plate fragmented: {near_ids}"
    assert len(far_ids) == 1, f"far plate fragmented: {far_ids}"
    assert not (near_ids & far_ids), "the two plates were merged into one track"
    return "two crossing vehicles: separate tracks, no ID theft"


def test_new_object_gets_new_id():
    """A plate appearing far from any prediction must not be force-matched."""
    tracker = make_tracker()
    first = tracker.update([(100, 100, 180, 124)])
    assert len(first) == 1
    origin_tid = first[0][0]
    second = tracker.update([(900, 700, 980, 724)])
    assert len(second) == 1
    assert second[0][0] != origin_tid, "distant plate was stolen by an existing track"
    assert tracker.active_track_count == 2
    return "distant new plate: correctly given a fresh track ID"


def test_stationary_plate_is_stable():
    """A jittering stationary plate must not fragment under the new gate."""
    jitter = [
        (500, 400, 580, 424), (502, 401, 582, 425), (499, 400, 579, 424),
        (501, 402, 581, 426), (500, 401, 580, 425), (503, 400, 583, 424),
    ]
    tracker = make_tracker()
    seen = set()
    for box in jitter:
        for tid, _matched, _predicted in tracker.update([box]):
            seen.add(tid)
    assert len(seen) == 1, f"stationary jitter fragmented into {len(seen)} tracks"
    return "stationary plate with detector jitter: 1 track"


def test_stale_track_expires():
    """max_age must still release per-track voter state after a disappearance.

    ``expired_track_ids`` reports the prune that the *current* ``update()``
    performed, which is how the pipelines consume it (they iterate it directly
    after ``update()`` to drop voter state). So collect it as it happens rather
    than after the loop.
    """
    tracker = make_tracker()
    first = tracker.update([(100, 100, 180, 124)])
    origin_tid = first[0][0]
    expired = []
    for _ in range(40):
        tracker.update([])
        expired.extend(tracker.expired_track_ids)
    assert tracker.active_track_count == 0, "abandoned track never expired"
    assert origin_tid in expired, f"expired track {origin_tid} not reported to the voter"
    return "abandoned track: expired and reported to the temporal voter"


def main():
    tests = [
        test_fast_vehicle_holds_one_track,
        test_fast_vehicle_reaches_voter_dwell,
        test_slow_vehicle_unaffected,
        test_two_vehicles_do_not_swap_tracks,
        test_new_object_gets_new_id,
        test_stationary_plate_is_stable,
        test_stale_track_expires,
    ]
    failures = 0
    for test in tests:
        try:
            print(f"  PASS  {test()}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
