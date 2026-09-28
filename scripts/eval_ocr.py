"""Measure ANPR OCR character accuracy on a labelled plate-crop set.

Runs the *exact* production OCR path (``_preprocess_plate_crop`` ->
``_ocr_plate_image_preferred`` -> ``_clean_plate_text``) over a folder of
plate crops with a ``labels.csv`` (``filename,plate``) and reports exact-match
and per-character accuracy.

To make the task realistic (the pipeline exists to read *low-resolution*
plates), each crop is first downscaled to ``--width`` pixels wide, then the
pipeline is allowed to upscale it again -- exactly what happens on a distant
plate in a video frame.

Usage (from the repo root):
    python scripts/eval_ocr.py --width 96
    python scripts/eval_ocr.py --width 64 --limit 100
    python scripts/eval_ocr.py --dir data/syn_eval --width 96
"""

import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr_engine.vision import (
    _clean_plate_text,
    _ocr_plate_image_preferred,
)


def levenshtein(a, b):
    """Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def char_accuracy(pred, truth):
    """1 - normalised edit distance (0..1)."""
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(pred, truth) / len(truth))


def load_labels(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rows.append((row['filename'], row['plate'].strip().upper()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join('data', 'syn_eval'))
    ap.add_argument('--width', type=int, default=96,
                    help='Downscale crop to this width before OCR (0=off)')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    labels = load_labels(os.path.join(args.dir, 'labels.csv'))
    if args.limit:
        labels = labels[:args.limit]

    exact = 0
    total = 0
    acc_sum = 0.0
    conf_sum = 0.0
    missed = 0

    for fname, truth in labels:
        img = cv2.imread(os.path.join(args.dir, fname), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if args.width and img.shape[1] > args.width:
            scale = args.width / img.shape[1]
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA,
            )
        raw, conf, _chars = _ocr_plate_image_preferred(img)
        pred = _clean_plate_text(raw) if raw else ''
        total += 1
        if not pred:
            missed += 1
        if pred == truth:
            exact += 1
        acc_sum += char_accuracy(pred, truth)
        conf_sum += float(conf or 0.0)

    if not total:
        print('No samples evaluated.')
        return 1

    print(f'dir={args.dir} width={args.width} samples={total}')
    print(f'  exact match     : {exact}/{total} = {exact / total * 100:.1f}%')
    print(f'  char accuracy   : {acc_sum / total * 100:.1f}%')
    print(f'  empty reads     : {missed}/{total} = {missed / total * 100:.1f}%')
    print(f'  mean confidence : {conf_sum / total:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
