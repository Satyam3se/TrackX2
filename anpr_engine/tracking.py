"""Motion-aware tracking + per-track temporal voting.

Ported from the `anpr-pipeline` project
(mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR). Each technique is
dependency-free and directly lifts low-resolution plate accuracy:

- ``iou`` / ``IoUTracker``: associate plate detections across video frames so
  reads of the *same physical plate* stay grouped. Association is
  **motion-aware** (see ``IoUTracker``) because a plain overlap test shatters
  fast vehicles into a new track every frame.
- ``TemporalVoter``: per-character-position, confidence-weighted majority vote
  over a track's top-K reads. The UFPR-SR-Plates benchmark (arXiv:2505.06393)
  showed this single trick lifts accuracy on low-res plates from ~31% to ~45%.

Both stages are gated on track *continuity*: the voter needs
``min_dwell`` reads on one track before it emits anything, so a tracker that
breaks a track every frame can never confirm a plate no matter how good the
OCR is. Motion-aware association is what makes short, fast passes confirmable.

Boxes are plain ``(x1, y1, x2, y2)`` int tuples -- no OpenCV/model coupling.
"""

from __future__ import annotations

from collections import defaultdict, deque

#: Affinity floor handed to a match accepted on centre distance rather than
#: overlap. Kept far below ``iou_threshold`` so any genuine overlap match always
#: outranks a proximity-only match in the global assignment.
_CENTER_MATCH_FLOOR = 0.05


def iou(a, b):
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    """``(cx, cy)`` of a ``(x1, y1, x2, y2)`` box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _extent(box):
    """``(w, h)`` of a box, floored at 1px so ratios stay finite."""
    return (max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1]))


class IoUTracker:
    """Assign stable track IDs to plate boxes across video frames.

    **Why plain IoU fails on fast vehicles.** With overlap-only association, a
    box that translates by more than roughly a third of its own size between
    two sampled frames drops below ``iou_threshold`` and is treated as a brand
    new object. On a 25fps feed sampled every 5th frame a plate covers its own
    width in well under 200ms at highway speed, so a fast vehicle is re-IDed
    every frame or two. The downstream ``TemporalVoter`` then never accumulates
    ``min_dwell`` reads on any single track, so the plate is never confirmed --
    fast vehicles silently vanish while slow ones read fine.

    **What this tracker does instead.**

    1. *Constant-velocity prediction.* Every track carries an EMA-smoothed
       centre velocity, and association is scored against where the track is
       expected to be this frame, not where it was last seen.
    2. *Centre-distance fallback.* A predicted box that a fast plate has already
       outrun no longer overlaps its own prediction cleanly, so a match is also
       accepted when the box centre lands within ``max_center_distance`` box
       diagonals of the prediction. The distance is normalised by the
       predicted box size, keeping the gate scale invariant.
    3. *Global best-first assignment.* All admissible (track, box) pairs are
       scored and consumed highest-first, instead of walking tracks in
       insertion order. A plate crossing in front of another no longer steals
       the wrong association just because it was detected first.
    4. *Velocity clamping.* One bad association cannot fling a prediction off
       screen: measured speed is capped at ``max_speed_scale`` box diagonals
       per frame.

    Tracks expire after ``max_age`` frames without a match. ``update()``
    returns ``[(track_id, box, predicted_box), ...]``: ``box`` is the raw
    detection (unchanged from the overlap-only behaviour, so it stays valid as
    a dict key and safe to crop), and ``predicted_box`` is where the track
    expected to see it, which callers can union with the detection to cover
    inter-frame motion when cropping.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        max_center_distance: float = 1.5,
        velocity_smoothing: float = 0.5,
        max_speed_scale: float = 2.0,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_center_distance < 0.0:
            raise ValueError("max_center_distance must be >= 0")
        if not 0.0 < velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in (0, 1]")
        if max_speed_scale <= 0.0:
            raise ValueError("max_speed_scale must be > 0")
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._max_center_distance = max_center_distance
        self._velocity_smoothing = velocity_smoothing
        self._max_speed_scale = max_speed_scale
        self._tracks = {}
        self._next_id = 1
        self._frame_idx = 0
        self._expired = []

    def _predict(self, track, gap):
        """Where ``track``'s box is expected ``gap`` frames from its last match."""
        cx, cy = _center(track['bbox'])
        w, h = _extent(track['bbox'])
        vx, vy = track['velocity']
        pcx = cx + vx * gap
        pcy = cy + vy * gap
        return (pcx - w / 2.0, pcy - h / 2.0, pcx + w / 2.0, pcy + h / 2.0)

    def _affinity(self, track, box, gap):
        """Score a (track, box) pairing; 0.0 means "not admissible"."""
        predicted = self._predict(track, gap)
        overlap = iou(predicted, box)
        if overlap >= self._iou_threshold:
            return overlap
        if self._max_center_distance <= 0.0:
            return 0.0
        pcx, pcy = _center(predicted)
        bcx, bcy = _center(box)
        pw, ph = _extent(predicted)
        diagonal = (pw * pw + ph * ph) ** 0.5
        distance = ((pcx - bcx) ** 2 + (pcy - bcy) ** 2) ** 0.5
        if distance / diagonal > self._max_center_distance:
            return 0.0
        return _CENTER_MATCH_FLOOR + overlap

    def _advance(self, track, box, gap):
        """Fold a matched detection into the track's box and velocity state."""
        tcx, tcy = _center(track['bbox'])
        bcx, bcy = _center(box)
        alpha = self._velocity_smoothing
        vx = (bcx - tcx) / gap
        vy = (bcy - tcy) / gap
        # Clamp a single frame's measured jump so one mis-association cannot
        # permanently fling the prediction off screen.
        bw, bh = _extent(track['bbox'])
        limit = self._max_speed_scale * (bw * bw + bh * bh) ** 0.5
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > limit > 0.0:
            vx *= limit / speed
            vy *= limit / speed
        track['velocity'] = (
            alpha * vx + (1.0 - alpha) * track['velocity'][0],
            alpha * vy + (1.0 - alpha) * track['velocity'][1],
        )
        track['bbox'] = box
        track['last_frame'] = self._frame_idx
        track['history'] += 1

    def update(self, boxes):
        """Assign boxes to tracks; return ``[(track_id, box, predicted), ...]``.

        ``predicted`` is the track's motion-compensated expectation for this
        frame (equal to the matched box for a brand new track).
        """
        self._frame_idx += 1
        boxes = list(boxes)
        pairs = []
        for tid, track in self._tracks.items():
            gap = max(1, self._frame_idx - track['last_frame'])
            for di, box in enumerate(boxes):
                score = self._affinity(track, box, gap)
                if score > 0.0:
                    pairs.append((score, tid, di, gap))

        # Highest affinity first, so the most confident pairing is consumed
        # before any weaker one can claim the same track or box.
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        assigned = []
        taken_tracks = set()
        taken_boxes = set()
        for _score, tid, di, gap in pairs:
            if tid in taken_tracks or di in taken_boxes:
                continue
            track = self._tracks[tid]
            predicted = self._predict(track, gap)
            self._advance(track, boxes[di], gap)
            taken_tracks.add(tid)
            taken_boxes.add(di)
            assigned.append((tid, boxes[di], predicted))

        for di, box in enumerate(boxes):
            if di in taken_boxes:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                'bbox': box,
                'velocity': (0.0, 0.0),
                'last_frame': self._frame_idx,
                'history': 1,
            }
            assigned.append((tid, box, box))

        stale = [
            tid for tid, t in self._tracks.items()
            if self._frame_idx - t['last_frame'] > self._max_age
        ]
        self._expired = stale
        for tid in stale:
            del self._tracks[tid]

        assigned.sort(key=lambda item: item[0])
        return assigned

    @property
    def active_track_count(self) -> int:
        """Number of currently tracked objects."""
        return len(self._tracks)

    @property
    def expired_track_ids(self) -> list:
        """Track IDs pruned by the last ``update()`` call because they aged out.

        Used by consumers to release per-track state (e.g. a temporal voter) so
        a re-entry of the same plate later starts a fresh confirmation window.
        """
        return list(self._expired)


def _vote(reads, *, top_k: int) -> str:
    """Confidence-weighted per-character-position majority of ``top_k`` reads.

    Each read is ``(text, confidence, char_confs)``. When a read carries
    aligned per-character confidences (``char_confs``) they are used to weight
    that character slot directly -- a low-confidence character contributes
    little, even inside a high-confidence overall read -- otherwise the read's
    flat mean confidence is used. Sorting for the top-K is still done on the
    scalar mean, which is a good proxy for overall read quality.
    """
    top = sorted(reads, key=lambda r: r[1], reverse=True)[:top_k]
    if not top:
        return ""
    max_len = max(len(text) for text, _, _ in top)
    chars = []
    for i in range(max_len):
        weights = defaultdict(float)
        for text, conf, char_confs in top:
            if i < len(text):
                weight = float(conf)
                if char_confs and i < len(char_confs):
                    weight = float(char_confs[i])
                weights[text[i]] += weight
        if weights:
            chars.append(max(weights.items(), key=lambda kv: kv[1])[0])
    return "".join(chars)


class TemporalVoter:
    """Per-track majority vote over recent OCR reads.

    ``observe(track_id, text, confidence, char_confs=None)`` accumulates the
    read and, once a track has at least ``min_dwell`` reads, returns the
    confirmed plate string (a re-computed majority every call). Returns
    ``None`` until then. Stale tracks are forgotten via ``forget()``.
    """

    def __init__(self, *, window: int = 8, min_dwell: int = 5, top_k: int = 3) -> None:
        if window < min_dwell:
            raise ValueError("window must be >= min_dwell")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._window = window
        self._min_dwell = min_dwell
        self._top_k = top_k
        self._tracks = defaultdict(lambda: None)
        self._history = defaultdict(lambda: deque(maxlen=self._window))

    def _track_deque(self, track_id: int) -> deque:
        return self._history[track_id]

    def observe(self, track_id: int, text: str, confidence: float,
                char_confs=None):
        """Record one read; return the confirmed plate once dwell is met.

        ``char_confs`` is an optional per-character confidence vector aligned
        to ``text`` (position-wise); when given it is used to weight each
        character in the majority vote.
        """
        reads = self._track_deque(track_id)
        reads.append((text, float(confidence), char_confs))
        if len(reads) < self._min_dwell:
            return None
        self._tracks[track_id] = _vote(list(reads), top_k=self._top_k)
        return self._tracks[track_id]

    def confirmed_for(self, track_id: int):
        """Last confirmed plate for a track (if dwell was met)."""
        h = self._tracks.get(track_id)
        return h if h else None

    def forget(self, track_id: int) -> None:
        """Drop all history for a track (e.g. when it expires)."""
        self._tracks.pop(track_id, None)
        self._history.pop(track_id, None)