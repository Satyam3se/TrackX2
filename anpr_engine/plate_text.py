"""Position-aware Indian license plate text correction & validation.

Lesson adapted from ``matthewearl/deep-anpr``: that project fixes the plate
*length and character semantics* in the network architecture (a fixed
``1 + 7 * len(CHARS)`` output) instead of cleaning free OCR text afterwards.
TrackX keeps EasyOCR, so instead of a destructive global character mapping
(e.g. ``S -> 5`` everywhere) we apply the same idea at the text level: every
character in an Indian RTO plate has a *fixed positional meaning* (letters vs
digits), so OCR confusions are resolved in context.

Example: ``KA05MS4321`` used to become ``KA05M54321`` with the old
``S -> 5`` mapping. This module keeps the ``S`` because position 6 IS a letter
slot.
"""

import re

__all__ = (
    'LETTER_SLOTS',
    'DIGIT_SLOTS',
    'PLATE_STRUCTURES',
    'VALID_STATES',
    'clean_plate_text',
    'is_valid_plate',
    'plate_confidence',
    'resolve_plate_structure',
)

#: Characters that a position can legally hold.
LETTER_SLOTS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGIT_SLOTS = '0123456789'

#: Typical Indian RTO layouts as (count, kind) block tuples.
#:   L/D = current 2-letter state, district digits, then letters + 4 digits.
PLATE_STRUCTURES = (
    ((2, 'L'), (2, 'D'), (1, 'L'), (4, 'D')),  # TS 07 J 9670      (9)
    ((2, 'L'), (2, 'D'), (2, 'L'), (4, 'D')),  # KA 05 MS 4321     (10)
    ((2, 'L'), (2, 'D'), (3, 'L'), (4, 'D')),  # DL 8C CAF 5001    (10, old)
    ((2, 'L'), (1, 'D'), (3, 'L'), (4, 'D')),  # TN 7 AB 1234      (10, old)
    ((2, 'L'), (2, 'D'), (0, 'L'), (4, 'D')),  # RJ 14 4526        (8, old)
)

#: OCR confusion maps, keyed by the *observed* char. ``ambiguous`` values are
#: acceptable alternatives with a small penalty.
_CONF = {
    # observed letter -> digit alternative
    'O': {'0'}, 'I': {'1'}, 'L': {'1'}, 'S': {'5'},
    'Z': {'2'}, 'B': {'8'}, 'G': {'6'}, 'Q': {'0'},
    'D': {'0'}, 'T': {'7'},
    # observed digit -> letter alternative
    '0': {'O'}, '1': {'I'}, '2': {'Z'}, '5': {'S'},
    '8': {'B'}, '6': {'G'}, '7': {'T'},
}

#: Expected slot layout for fixed-length plates keyed by total length.
#: ('L' letter, 'D' digit) -- taken from PlateVision-AI's ``get_expected_type``.
_SLOT_LAYOUT = {
    10: 'LLDDLLDDDD',
    9:  'LLDDLDDDD',
    8:  'LLDDDDDD',
}

#: Valid Indian state/district registration codes (from PlateVision-AI).
VALID_STATES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'KA', 'KL',
    'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS',
    'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'JK', 'LA', 'LD',
    'PY',
}

#: OCR padding bubbles that commonly wrap Indian plates (e.g. "INDIA" text).
#: Longest first so "INDIA" wins over "IND"/"IN" when both could match.
_PREFIX_NOISE = ('INDIA', '1ND1A', 'IND', '1ND', 'IN', '1N')

#: State codes ordered by approximate registration frequency. When a dropped
#: leading letter admits several valid state prefixes (``S07JS9670`` could be
#: ``AS``/``TS``), a tie is currently broken by string order -- AS won by pure
#: luck. Preferring the more common state makes the recovery more likely to be
#: right in practice.
_STATE_PREFERENCE = (
    'UP', 'MH', 'DL', 'KA', 'MP', 'TN', 'AP', 'RJ', 'TS', 'GJ',
    'KL', 'HR', 'PB', 'WB', 'BR', 'OD', 'CG', 'UK', 'JH', 'HP',
    'AS', 'GA', 'TR', 'JK', 'AN', 'CH', 'DD', 'DN', 'SK', 'MN',
    'ML', 'MZ', 'NL', 'AR', 'LA', 'LD', 'PY',
)

_RESOLVER = {'L': {}, 'D': {}}
for _ch, _alts in _CONF.items():
    if _ch.isalpha():
        _RESOLVER['L'][_ch] = {_ch} | _alts
        _RESOLVER['D'][_ch] = _alts  # letter observed in a digit slot
    else:
        _RESOLVER['D'][_ch] = {_ch} | _alts
        _RESOLVER['L'][_ch] = _alts  # digit observed in a letter slot
del _ch, _alts

#: Fallback legacy behaviour kept for plates nothing else can parse.
_LEGACY_MAPPING = {
    'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2',
}

#: Penalty applied to substring-scan candidates. Substrings are slices of a
#: noisier string that happen to look plate-shaped, so they must never beat a
#: full-length parse: ``KA05M54321``'s slot-fixed ``KA05MS4321`` (score -1)
#: would otherwise lose to the substring ``KA055432``'s clean -2. The penalty
#: still lets a substring rescue a plate when nothing else parses.
SUBSTRING_PENALTY = 3


def _sanitize(raw_text):
    """Uppercase and keep only alphanumerics."""
    return ''.join(ch for ch in str(raw_text).upper() if ch.isalnum()) if raw_text else ''


def _fix_with_slots(alnum, layout):
    """Fix ``alnum`` against a fixed L/D slot layout string (e.g. 'LLDDLLDDDD').

    Returns ``(fixed, cost)``. A char already matching its slot costs 0; an
    ambiguous flip costs 1; an impossible char fails (returns None).
    """
    if len(alnum) != len(layout):
        return None
    out = []
    cost = 0
    for ch, slot in zip(alnum, layout):
        charset = LETTER_SLOTS if slot == 'L' else DIGIT_SLOTS
        if ch in charset:
            out.append(ch)
        elif ch in _RESOLVER[slot]:
            out.append(sorted(_RESOLVER[slot][ch])[0])
            cost += 1
        else:
            return None
    return ''.join(out), cost


def _fix_with_structure(alnum, structure):
    """Map ``alnum`` onto a ``(count, kind)`` structure char-by-char."""
    out = []
    cost = 0
    pos = 0
    for count, kind in structure:
        charset = LETTER_SLOTS if kind == 'L' else DIGIT_SLOTS
        for _ in range(count):
            if pos >= len(alnum):
                return None
            ch = alnum[pos]
            pos += 1
            if ch in charset:
                out.append(ch)
            elif ch in _RESOLVER[kind]:
                out.append(sorted(_RESOLVER[kind][ch])[0])
                cost += 1
            else:
                return None
    if pos != len(alnum):
        return None
    return ''.join(out), cost


def _prepend_leading_letters(tail, layout, drop_offset):
    """Recover dropped leading letters of ``tail`` inside a slot ``layout``.

    OCR often swallows the first state character(s), e.g. reading
    ``TS07JS9670`` as ``S07JS9670``. We treat ``tail`` as layout suffixes and
    search the missing leading slots against the known state codes, breaking
    ties between equally-valid prefixes by registration frequency.

    Returns an iterable of ``(plate, score)``.
    """
    pref = {code: idx for idx, code in enumerate(_STATE_PREFERENCE)}
    max_rank = len(_STATE_PREFERENCE)
    prefixes = sorted(
        VALID_STATES, key=lambda code: pref.get(code, max_rank),
    )
    results = []
    for prepend in (1, 2):
        if len(tail) != len(layout) - prepend:
            continue
        if layout[:prepend] != 'L' * prepend:
            continue
        fixed = _fix_with_slots(tail, layout[prepend:])
        if fixed is None:
            continue
        body, cost = fixed
        for prefix in prefixes:
            if not body.startswith(prefix[prepend:]):
                continue
            plate = prefix[:prepend] + body
            # A tiny rank penalty breaks ties between equal-score prefixes so
            # the *common* state beats the alphabetically-first one.
            score = _score(plate, cost + 2 * prepend, drop_offset) \
                + pref.get(prefix, max_rank) * 0.01
            results.append((plate, score))
    return results


def _score(plate, cost, drop=0):
    """Total score for a candidate plate: fix cost + slot penalty - state bonus."""
    score = cost + 2 * drop
    if plate and plate[:2] in VALID_STATES:
        score -= 2  # known state code heavily preferred (PlateVision trick)
    return score


def resolve_plate_structure(raw_text):
    """Return ``(best_plate, score)`` or ``(None, None)``.

    Strategy (blends deep-anpr's fixed-layout idea with PlateVision-AI's
    correction): strip common "INDIA" padding, then
      1. try exact fixed-length layouts (8/9/10) with drop-forgiveness,
      2. try our flexible (count, kind) structures,
      3. recover swallowed leading state letters via VALID_STATES,
      4. scan substrings for a plate-shaped window inside noisy OCR text.
    The lowest-total-cost match wins.
    """
    alnum = _sanitize(raw_text)
    if not alnum:
        return None, None

    # Remove common "INDIA" / "IND" border padding that EasyOCR picks up.
    for prefix in _PREFIX_NOISE:
        if alnum.startswith(prefix) and len(alnum) > len(prefix) + 7:
            alnum = alnum[len(prefix):]
            break

    candidates = []  # (plate, score)

    # 1. Exact fixed-length layouts (the most reliable: PlateVision's method).
    layouts = list(_SLOT_LAYOUT.values())
    for layout in layouts:
        for drop in range(min(3, len(alnum) + 1)):
            tail = alnum[drop:]
            if len(tail) == len(layout):
                fixed = _fix_with_slots(tail, layout)
                if fixed:
                    candidates.append((fixed[0], _score(fixed[0], fixed[1], drop)))
            candidates.extend(_prepend_leading_letters(tail, layout, drop))

    # 2. Flexible (count, kind) structures.
    for structure in PLATE_STRUCTURES:
        total = sum(count for count, _ in structure)
        for drop in range(min(3, len(alnum) + 1)):
            if len(alnum) - drop != total:
                continue
            fixed = _fix_with_structure(alnum[drop:], structure)
            if fixed:
                candidates.append((fixed[0], _score(fixed[0], fixed[1], drop)))

    # 3. Substring scan: plates hidden inside longer OCR text.
    for m in re.finditer(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}', alnum):
        plate = m.group(0)
        candidates.append(
            (plate, _score(plate, 0, 0) + SUBSTRING_PENALTY),
        )

    if not candidates:
        return None, None
    best_plate, best_score = min(candidates, key=lambda c: (c[1], -len(c[0])))
    return best_plate, best_score


def is_valid_plate(plate):
    """Best-effort structural validity check for a normalised Indian plate."""
    if not plate:
        return False
    return re.fullmatch(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}', plate) is not None


def plate_confidence(plate, score):
    """Map an internal resolve score to a 0..1 confidence."""
    if not plate:
        return 0.0
    base = 0.92
    return max(0.1, base - 0.12 * score)


def clean_plate_text(raw_text):
    """Cleanse raw EasyOCR text into its most likely Indian plate string.

    Falls back to the legacy whole-string mapping (unchanged behaviour) when
    no known layout fits, so nothing that previously worked regresses.
    """
    if not raw_text:
        return ''

    resolved, score = resolve_plate_structure(raw_text)
    if resolved:
        return resolved

    # Legacy fallback: apply the historical letter->digit mapping verbatim.
    cleaned = _sanitize(raw_text)
    mapped = ''.join(_LEGACY_MAPPING.get(c, c) for c in cleaned)
    if len(mapped) == 9 and mapped[0].isdigit():
        mapped = 'T' + mapped
    return mapped