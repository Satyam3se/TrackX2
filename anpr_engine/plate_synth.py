"""Synthetic Indian license plate generator.

Lesson adapted from ``matthewearl/deep-anpr``: that project renders plates
with a TrueType font, applies random 3D affine warp, and composites them over
photographic backgrounds to synthesize unlimited labelled training data,
instead of collecting+labelling real plates by hand.

TrackX version:
  * Renders valid Indian RTO plate layouts (not fixed 7-char UK format).
  * Backgrounds are sampled from your own camera footage (any directory with
    images/videos, e.g. ``Videos/``), falling back to procedural noise.
  * Two output modes:

      crops    -- standalone plate crops + ``labels.csv`` (``filename,plate``)
                  for training/fine-tuning a plate OCR/classifier.

      scenes   -- full images with plates pasted at random scale/location plus
                  YOLOv8 label files (class ``0``) + ``data.yaml``, for
                  augmenting ``license_plate_detector.pt`` training.

Usage::

    python anpr_engine/plate_synth.py --count 500 --out syn_plates --mode crops
    python anpr_engine/plate_synth.py --count 300 --out syn_scenes --mode scenes --bg-dir Videos

Drop your own plate font(s) into ``fonts/`` (any ``*.ttf``) for the best
character rendering; otherwise a bundled system font is used.
"""

import argparse
import math
import os
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = 'fonts'
OUTPUT_SHAPE = (64, 128)  # (height, width) of generated plate crops
SCENE_SIZE = 640          # square canvas for --mode scenes
PLATE_LETTERS = string.ascii_uppercase
PLATE_DIGITS = string.digits

#: Invalid-before-letter pairs are not enforced; keep the random generator lean.
#: Layouts mirror plate_text.PLATE_STRUCTURES with common sizes weighted first.
STRUCTURES = (
    ((2, 'L'), (2, 'D'), (2, 'L'), (4, 'D')),  # KA 05 MS 4321  (10)  -- common
    ((2, 'L'), (2, 'D'), (1, 'L'), (4, 'D')),  # TS 07 J 9670   (9)
    ((2, 'L'), (2, 'D'), (0, 'L'), (4, 'D')),  # RJ 14 4526     (8, old)
    ((2, 'L'), (1, 'D'), (3, 'L'), (4, 'D')),  # TN 7 AB 1234   (10, old)
)


def generate_code():
    """Generate a random, structurally valid Indian RTO plate string."""
    structure = random.choice(STRUCTURES)
    chars = []
    for count, kind in structure:
        pool = PLATE_LETTERS if kind == 'L' else PLATE_DIGITS
        chars.extend(random.choice(pool) for _ in range(count))
    return ''.join(chars)


def find_font(font_dir=FONT_DIR):
    """Locate the best available TrueType font for rendering plates."""
    candidates = []
    if os.path.isdir(font_dir):
        candidates = [os.path.join(font_dir, f) for f in os.listdir(font_dir)
                      if f.lower().endswith(('.ttf', '.otf'))]
    if not candidates:
        system_candidates = [
            # Windows
            r'C:\Windows\Fonts\Arial.ttf',
            r'C:\Windows\Fonts\arial.ttf',
            r'C:\Windows\Fonts\Arialbd.ttf',
            r'C:\Windows\Fonts\impact.ttf',
            # Linux / macOS
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/Library/Fonts/Arial Bold.ttf',
            '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        ]
        candidates = [p for p in system_candidates if os.path.isfile(p)]
    if not candidates:
        raise RuntimeError(
            'No TrueType font found. Put any plate-like *.ttf into ./fonts/'
        )
    return candidates[0]


def render_plate(code, font_path, font_height=32):
    """Render a plate string into a (HxW) float image + rounded-rect mask.

    Mirrors deep-anpr's ``generate_plate``: per-character glyphs are drawn
    with PIL then row-packed with random padding and spacing.
    """
    char_ims = {}
    for c in code:
        font_size = font_height * 4
        font = ImageFont.truetype(font_path, font_size)
        h = font.getbbox(max(c, 'M'))[3]
        w = font.getbbox(c)[2]
        im = Image.new('L', (max(w, 4), h + 4), 0)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, fill=255, font=font)
        scale = float(font_height) / (h + 4)
        im = im.resize(
            (max(1, int(w * scale)), font_height),
            Image.LANCZOS,
        )
        char_ims[c] = np.asarray(im, dtype=np.float32) / 255.0

    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color = 1.0 if random.random() < 0.85 else 0.0
    plate_color = 0.0 if text_color else 1.0

    text_mask = np.zeros(out_shape)
    x = h_padding
    y = v_padding
    for c in code:
        ci = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + ci.shape[0], ix:ix + ci.shape[1]] = ci
        x += ci.shape[1] + spacing

    plate = (np.ones(out_shape) * plate_color * (1.0 - text_mask) +
             np.ones(out_shape) * text_color * text_mask)
    return plate, rounded_rect(out_shape, radius)


def rounded_rect(shape, radius):
    """Binary mask of a rectangle with rounded corners (np.ndarray)."""
    out = np.ones(shape)
    for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
        yy = (shape[0] - 1) if y else 0
        xx = (shape[1] - 1) if x else 0
        cv2.circle(out, (xx, yy), radius, 0.0, -1)
    return out


def euler_to_mat(yaw, pitch, roll):
    """Rotation matrix from Euler angles (adapted from deep-anpr)."""
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[1., 0., 0.], [0., c, -s], [0., s, c]]) * M
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) * M
    return M


def make_affine_transform(from_shape, to_shape,
                          min_scale=0.15, max_scale=0.6,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    """Build a random 3D-morphed affine matrix that keeps the plate in bounds.

    Adapted from deep-anpr's ``make_affine_transform``.
    """
    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    scale = max(min_scale, min(max_scale, scale))
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w], [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                           np.min(M * corners, axis=1))
    scale *= np.min(to_size / skewed_size)

    trans = (np.random.random((2, 1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.
    M = euler_to_mat(yaw, pitch, roll)[:2, :2] * scale
    M = np.hstack([M, trans + center_to - M * center_from])
    return M


def numpy_random_like_0_5():
    return np.random.random((2, 1))


def get_background_frames(bg_dir, limit=24):
    """Sample background frames from a dir of images/videos. Empty -> []."""
    frames = []
    if not bg_dir or not os.path.isdir(bg_dir):
        return frames
    vids = [f for f in os.listdir(bg_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    imgs = [f for f in os.listdir(bg_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imgs)
    for name in imgs:
        if len(frames) >= limit:
            break
        p = os.path.join(bg_dir, name)
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            frames.append(im)
    random.shuffle(vids)
    for name in vids:
        if len(frames) >= limit:
            break
        cap = cv2.VideoCapture(os.path.join(bg_dir, name))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randrange(0, total))
        ok, f = cap.read()
        cap.release()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return frames


def make_procedural_bg(shape):
    """Random gradient + noise background used when no footage is found."""
    h, w = shape
    grad = np.linspace(0.2, 0.9, w, dtype=np.float32)[None, :]
    grad = np.repeat(grad, h, axis=0)
    grad += np.random.normal(0, 0.08, (h, w)).astype(np.float32)
    return np.clip(grad, 0.0, 1.0)


def generate_plate_crop(code, font_path, bg):
    """Warp a rendered plate over a random crop of ``bg``; return crop + plate bbox.

    Returns ``(crop, (x1, y1, x2, y2))`` where the bbox marks the plate region
    in crop coordinates.
    """
    plate, plate_mask = render_plate(code, font_path)
    bg_h, bg_w = bg.shape[:2]

    # Pick a random window big enough for the plate.
    max_h = min(bg_h, 160)
    max_w = min(bg_w, 320)
    window_h = random.randint(60, max_h)
    window_w = random.randint(120, max_w)
    top = random.randint(0, bg_h - window_h)
    left = random.randint(0, bg_w - window_w)
    bg_window = bg[top:top + window_h, left:left + window_w]
    bg_window = cv2.resize(bg_window, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    M = make_affine_transform(
        from_shape=plate.shape,
        to_shape=bg_window.shape,
        min_scale=0.4,
        max_scale=0.9,
        scale_variation=1.0,
        rotation_variation=1.0,
        translation_variation=0.6,
    )

    scaled_plate = cv2.warpAffine(plate, M,
                                  (bg_window.shape[1], bg_window.shape[0]))
    scaled_mask = cv2.warpAffine(plate_mask, M,
                                 (bg_window.shape[1], bg_window.shape[0]))
    out = scaled_plate * scaled_mask + bg_window * (1.0 - scaled_mask)

    noise = np.random.normal(scale=0.05, size=out.shape)
    out = np.clip(out + noise, 0.0, 1.0)

    # Plate bbox: project the plate rectangle corners through the affine M.
    h, w = plate.shape
    corn = np.array([[0., 0.], [w, 0.], [w, h], [0, h]], dtype=np.float32)
    proj = cv2.transform(corn.reshape(1, -1, 2), M).reshape(-1, 2)
    x1, y1 = int(np.min(proj[:, 0])), int(np.min(proj[:, 1]))
    x2, y2 = int(np.max(proj[:, 0])), int(np.max(proj[:, 1]))
    return (out * 255.0).astype(np.uint8), (x1, y1, x2, y2)


def generate_scene(code, font_path, bg):
    """Paste a plate into a full ``SCENE_SIZE`` canvas; return (scene, bbox).

    bbox is ``[x1, y1, x2, y2]`` in scene pixel coordinates.
    """
    plate, plate_mask = render_plate(code, font_path)
    scene_w = scene_h = SCENE_SIZE
    if bg is None:
        bg_canvas = make_procedural_bg((scene_h, scene_w))
    else:
        h, w = bg.shape[:2]
        scale = max(scene_w / w, scene_h / h)
        bg_canvas = cv2.resize(bg, (int(w * scale), int(h * scale)))
        top = random.randint(0, max(0, bg_canvas.shape[0] - scene_h))
        left = random.randint(0, max(0, bg_canvas.shape[1] - scene_w))
        bg_canvas = bg_canvas[top:top + scene_h, left:left + scene_w]
    bg_canvas = cv2.resize(bg_canvas, (scene_w, scene_h))

    M = make_affine_transform(
        from_shape=plate.shape,
        to_shape=bg_canvas.shape,
        min_scale=0.05,
        max_scale=0.35,
        scale_variation=0.8,
        rotation_variation=1.0,
        translation_variation=1.0,
    )

    warped = cv2.warpAffine(plate, M, (scene_w, scene_h))
    mask = cv2.warpAffine(plate_mask, M, (scene_w, scene_h))
    out = warped * mask + bg_canvas * (1.0 - mask)

    noise = np.random.normal(scale=0.03, size=out.shape)
    out = np.clip(out + noise, 0.0, 1.0)

    h, w = plate.shape
    corn = np.array([[0., 0.], [w, 0.], [w, h], [0, h]], dtype=np.float32)
    proj = cv2.transform(corn.reshape(1, -1, 2), M).reshape(-1, 2)
    x1, y1 = int(np.min(proj[:, 0])), int(np.min(proj[:, 1]))
    x2, y2 = int(np.max(proj[:, 0])), int(np.max(proj[:, 1]))
    return (out * 255.0).astype(np.uint8), (x1, y1, x2, y2)


def write_scene_labels(scene, bbox, label_path):
    """Write a YOLOv8-format label line for a single class-0 plate."""
    H, W = scene.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    with open(label_path, 'w') as f:
        f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')


def main():
    parser = argparse.ArgumentParser(description='TrackX synthetic Indian plate generator')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--out', default='syn_plates')
    parser.add_argument('--mode', choices=('crops', 'scenes'), default='crops')
    parser.add_argument('--bg-dir', default='Videos',
                        help='Directory of real footage/images for backgrounds')
    parser.add_argument('--font', default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    font_path = args.font or find_font()
    bgs = get_background_frames(args.bg_dir)
    print(f'[i] font     : {font_path}')
    print(f'[i] backgrounds: {len(bgs)} frame(s) from {args.bg_dir or "(procedural)"}')

    labels = []
    for i in range(args.count):
        code = generate_code()
        bg = random.choice(bgs) if bgs else None

        if args.mode == 'crops':
            crop, _ = generate_plate_crop(code, font_path, bg)
            fname = f'{i:06d}_{code}.png'
            cv2.imwrite(os.path.join(args.out, fname), crop)
            labels.append((fname, code))
        else:
            scene, bbox = generate_scene(code, font_path, bg)
            fname = f'{i:06d}_{code}.jpg'
            cv2.imwrite(os.path.join(args.out, fname), scene)
            write_scene_labels(scene, bbox,
                               os.path.join(args.out, f'{i:06d}.txt'))
            labels.append((fname, code))

    with open(os.path.join(args.out, 'labels.csv'), 'w') as f:
        f.write('filename,plate\n')
        for fname, code in labels:
            f.write(f'{fname},{code}\n')

    if args.mode == 'scenes':
        abs_out = os.path.abspath(args.out).replace('\\', '/')
        with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
            f.write(f'path: {abs_out}\n'
                    f'train: {abs_out}\n'
                    f'val: {abs_out}\n'
                    f'nc: 1\n'
                    f'names: ["license_plate"]\n')

    print(f'[OK] Wrote {args.count} plates + labels to {args.out}/')


if __name__ == '__main__':
    main()