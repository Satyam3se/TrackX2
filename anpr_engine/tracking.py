"""IoU-greedy tracking + per-track temporal voting.

Ported from the `anpr-pipeline` project
(mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR). Each technique is
dependency-free and directly lifts low-resolution plate accuracy:

- ``iou`` / ``IoUTracker``: associate plate detections across video frames by
  bounding-box overlap, so reads of the *same physical plate* stay grouped.
- ``TemporalVoter``: per-character-position, confidence-weighted majority vote
  over a track's top-K reads. The UFPR-SR-Plates benchmark (arXiv:2505.06393)
  showed this single trick lifts accuracy on low-res plates from ~31% to ~45%.

Boxes are plain ``(x1, y1, x2, y2)`` int tuples -- no OpenCV/model coupling.
"""

from __future__ import annotations

from collections import defaultdict, deque


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


class IoUTracker:
    """Assign stable track IDs to plate boxes across video frames.

    Greedy best-IoU assignment (a box can match at most one track). Tracks
    expire after ``max_age`` frames without a match. Returns
    ``[(track_id, (x1, y1, x2, y2)), ...]`` for the current frame.
    """

    def __init__(self, *, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._tracks = {}
        self._next_id = 1
        self._frame_idx = 0
        self._expired = []

    def update(self, boxes):
        """Assign/dedup boxes to tracks; return ``[(track_id, box), ...]``."""
        self._frame_idx += 1
        assigned = []
        unmatched = list(range(len(boxes)))

        for tid, track in list(self._tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for di in unmatched:
                v = iou(track['bbox'], boxes[di])
                if v > best_iou:
                    best_iou = v
                    best_idx = di
            if best_iou >= self._iou_threshold and best_idx >= 0:
                track['bbox'] = boxes[best_idx]
                track['last_frame'] = self._frame_idx
                track['history'] += 1
                assigned.append((tid, boxes[best_idx]))
                unmatched.remove(best_idx)

        for di in unmatched:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                'bbox': boxes[di],
                'last_frame': self._frame_idx,
                'history': 1,
            }
            assigned.append((tid, boxes[di]))

        stale = [
            tid for tid, t in self._tracks.items()
            if self._frame_idx - t['last_frame'] > self._max_age
        ]
        self._expired = stale
        for tid in stale:
            del self._tracks[tid]

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
    """Confidence-weighted per-character-position majority of ``top_k`` reads."""
    top = sorted(reads, key=lambda r: r[1], reverse=True)[:top_k]
    if not top:
        return ""
    max_len = max(len(text) for text, _ in top)
    chars = []
    for i in range(max_len):
        weights = defaultdict(float)
        for text, conf in top:
            if i < len(text):
                weights[text[i]] += conf
        if weights:
            chars.append(max(weights.items(), key=lambda kv: kv[1])[0])
    return "".join(chars)


class TemporalVoter:
    """Per-track majority vote over recent OCR reads.

    ``observe(track_id, text, confidence)`` accumulates the read and, once a
    track has at least ``min_dwell`` reads, returns the confirmed plate string
    (a re-computed majority every call). Returns ``None`` until then.
    Stale tracks are forgotten via ``forget()``.
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

    def observe(self, track_id: int, text: str, confidence: float):
        """Record one read; return the confirmed plate once dwell is met."""
        reads = self._track_deque(track_id)
        reads.append((text, float(confidence)))
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