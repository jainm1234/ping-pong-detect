import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BG = DATA / "backgrounds"
PREVIEW = ROOT / "runs" / "preview"
GROUP = "orange"
COUNTS = {"train": 800, "val": 200}
SEEDS = {"train": 11, "val": 23}
WIDTH, HEIGHT = 1280, 720
MIN_SIDE = 12
SHEETS = ["orange_balls_a.jpg", "orange_balls_b.jpg", "orange_balls_c.jpg", "orange_commons.jpg"]

# (share, smallest diameter, largest diameter) in output pixels
SIZE_CLASSES = [(0.32, 12, 30), (0.36, 30, 100), (0.24, 100, 260), (0.08, 260, 560)]
DISTRACTORS = {
    "basketball": 0.13,
    "tennis": 0.11,
    "soccer": 0.09,
    "baseball": 0.07,
    "beach": 0.06,
    "fruit": 0.07,
    "light": 0.15,
    "bokeh": 0.07,
    "smiley": 0.08,
    "clock": 0.07,
    "face": 0.10,
}


def unit(*v):
    v = np.array(v, np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-8)


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


def hsv_bgr(h, s, v):
    pixel = np.uint8([[[int(h) % 180, int(np.clip(s, 0, 255)), int(np.clip(v, 0, 255))]]])
    return cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255


def log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def grain(rng, n, sigma, amount):
    noise = rng.normal(0, 1, (n, n)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), max(0.3, sigma))
    return noise / (noise.std() + 1e-6) * amount


def sphere(size, supersample=2):
    n = max(16, int(round(size * supersample)))
    coords = (np.arange(n, dtype=np.float32) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(coords, -coords)
    r = np.sqrt(x * x + y * y)
    z = np.sqrt(np.clip(1 - r * r, 0, 1))
    alpha = np.clip((1 - r) * n / 2, 0, 1)
    return np.dstack([x, y, z]), alpha


def light_dir(rng):
    return unit(rng.uniform(-1, 1), rng.uniform(-0.3, 1), rng.uniform(0.35, 1.2))


def shade(normal, albedo, light, ambient, wrap, ks, shininess):
    diffuse = np.clip((normal @ light + wrap) / (1 + wrap), 0, 1)
    color = albedo * (ambient + (1 - ambient) * diffuse)[..., None]
    half = unit(light + np.array([0, 0, 1], np.float32))
    spec = ks * np.clip(normal @ half, 0, 1) ** shininess
    return color + spec[..., None]


def finish(color, alpha, width, height=None):
    height = height or width
    color = cv2.resize(np.clip(color, 0, 1.2), (width, height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    return color.astype(np.float32), np.clip(alpha, 0, 1).astype(np.float32)


def logo_patch(rng):
    patch = np.zeros((160, 320), np.uint8)
    kind = rng.integers(4)
    font = int(rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]))

    def text(label, center_y, max_w):
        scale = 4.0
        thickness = int(rng.integers(5, 11))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        scale *= min(1.0, max_w / tw, 70 / th)
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.putText(patch, label, (160 - tw // 2, center_y + th // 2), font, scale, 255, thickness, cv2.LINE_AA)

    def stars(count, center_y, radius):
        for i in range(count):
            cx = 160 + (i - (count - 1) / 2) * radius * 2.3
            points = []
            for k in range(10):
                r = radius if k % 2 == 0 else radius * 0.42
                angle = -math.pi / 2 + k * math.pi / 5
                points.append((cx + r * math.cos(angle), center_y + r * math.sin(angle)))
            cv2.fillPoly(patch, [np.int32(points)], 255, cv2.LINE_AA)

    words = ["40+", "ABS", "PRO", "3 STAR", "TRAINING", "CLUB", "D40+", "SPIN", "ACE", "TT", "PREMIUM", "MATCH"]
    if kind == 0:
        text(str(rng.choice(words)), 80, 300)
    elif kind == 1:
        stars(int(rng.integers(1, 4)), 80, 34)
    elif kind == 2:
        stars(int(rng.integers(1, 4)), 45, 26)
        text(str(rng.choice(words)), 120, 260)
    else:
        cv2.ellipse(patch, (160, 80), (120, 55), float(rng.uniform(-20, 20)), 200, 520, 255, 12, cv2.LINE_AA)
        text(str(rng.choice(words)), 80, 170)
    return patch.astype(np.float32) / 255


def apply_logo(albedo, normal, rng):
    patch = logo_patch(rng)
    ph, pw = patch.shape
    center = unit(rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), 1)
    tangent = unit(np.cross([0, 1, 0], center))
    bitangent = np.cross(center, tangent)
    angle = rng.uniform(-0.7, 0.7)
    t = tangent * math.cos(angle) + bitangent * math.sin(angle)
    b = -tangent * math.sin(angle) + bitangent * math.cos(angle)
    half = rng.uniform(0.3, 0.65)
    u = normal @ t / half
    v = normal @ b / (half * ph / pw)
    map_x = ((u + 1) / 2 * (pw - 1)).astype(np.float32)
    map_y = ((1 - v) / 2 * (ph - 1)).astype(np.float32)
    ink = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    ink *= (normal @ center > 0.2).astype(np.float32)
    ink_color = [
        np.array([0.05, 0.05, 0.05]),
        np.array([0.35, 0.12, 0.05]),
        np.array([0.08, 0.08, 0.75]),
        np.array([0.9, 0.9, 0.9]),
        np.array([0.25, 0.25, 0.25]),
    ][int(rng.choice(5, p=[0.5, 0.15, 0.15, 0.1, 0.1]))].astype(np.float32)
    strength = rng.uniform(0.55, 0.95)
    weight = (ink * strength)[..., None]
    return albedo * (1 - weight) + ink_color * weight


def render_orange(rng, size):
    normal, alpha = sphere(size)
    n = normal.shape[0]
    hue = rng.triangular(4, 12, 19)
    base = hsv_bgr(hue, rng.uniform(190, 255), rng.uniform(215, 255))
    albedo = np.broadcast_to(base, normal.shape).copy()
    albedo *= 1 + grain(rng, n, n * rng.uniform(0.03, 0.1), rng.uniform(0.004, 0.02))[..., None]
    if rng.random() < 0.5:
        seam = random_rotation(rng)[0]
        width = rng.uniform(0.006, 0.018)
        band = np.exp(-((normal @ seam) ** 2) / (2 * width * width))
        albedo *= (1 - rng.uniform(0.08, 0.25) * band)[..., None]
    if rng.random() < 0.6:
        albedo = apply_logo(albedo, normal, rng)
    if rng.random() < 0.1:
        scuffs = np.clip(grain(rng, n, n * 0.02, 1.0) - 2.0, 0, None)
        albedo *= (1 - 0.12 * np.clip(scuffs, 0, 1))[..., None]
    finish_kind = rng.random()
    if finish_kind < 0.4:
        ks, shininess = rng.uniform(0.0, 0.12), rng.uniform(3, 10)
    elif finish_kind < 0.8:
        ks, shininess = rng.uniform(0.12, 0.4), rng.uniform(12, 40)
    else:
        ks, shininess = rng.uniform(0.4, 0.85), rng.uniform(40, 140)
    color = shade(normal, albedo, light_dir(rng), rng.uniform(0.35, 0.6), rng.uniform(0.3, 0.8), ks, shininess)
    color *= (0.88 + 0.12 * normal[..., 2] ** 0.4)[..., None]
    return finish(color, alpha, size)


def cut_sheets():
    sprites = []
    for name in SHEETS:
        image = cv2.imread(str(BG / "sprites" / name))
        if image is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        mask = (((hue < 32) | (hue > 165)) & (sat > 60) & ((sat > 140) | (val > 200))).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[1] // 150 * 10000 + cv2.boundingRect(c)[0]):
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            if area < image.shape[0] * image.shape[1] * 0.01:
                continue
            _, enclosing = cv2.minEnclosingCircle(contour)
            if area / (math.pi * enclosing * enclosing) < 0.75:
                continue
            moments = cv2.moments(hull)
            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            radius = math.sqrt(area / math.pi) * 0.965
            r = int(math.ceil(radius))
            x0, y0 = int(cx) - r, int(cy) - r
            if x0 < 0 or y0 < 0 or x0 + 2 * r > image.shape[1] or y0 + 2 * r > image.shape[0]:
                continue
            crop = image[y0 : y0 + 2 * r, x0 : x0 + 2 * r].astype(np.float32) / 255
            yy, xx = np.mgrid[: 2 * r, : 2 * r].astype(np.float32) + 0.5
            dist = np.sqrt((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2)
            alpha = np.clip(radius - dist, 0, 1)
            sprites.append((f"{Path(name).stem}_{len(sprites):02d}", crop, alpha))
    return sprites


def photo_orange(rng, sprite, size):
    _, crop, alpha = sprite
    side = crop.shape[0]
    matrix = cv2.getRotationMatrix2D((side / 2, side / 2), float(rng.uniform(0, 360)), 1.0)
    crop = cv2.warpAffine(crop, matrix, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    alpha = cv2.warpAffine(alpha, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.5:
        crop, alpha = crop[:, ::-1], alpha[:, ::-1]
    hsv = cv2.cvtColor((np.clip(crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    hsv[..., 2] *= rng.uniform(0.85, 1.1)
    crop = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    color, alpha = finish(crop, np.ascontiguousarray(alpha), size)
    normal, _ = sphere(size, 1)
    relight = 1 + rng.uniform(0, 0.35) * (normal @ light_dir(rng) - 0.3)
    relight = cv2.resize(relight.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return color * relight[..., None], alpha


def render_patterned(rng, size, kind):
    normal, alpha = sphere(size)
    n = normal.shape[0]
    frame = random_rotation(rng)
    a, b, c = frame
    ks, shininess, ambient, wrap = 0.05, 8, rng.uniform(0.25, 0.45), rng.uniform(0.1, 0.4)
    if kind == "basketball":
        albedo = np.broadcast_to(hsv_bgr(rng.uniform(7, 14), rng.uniform(190, 240), rng.uniform(120, 185)), normal.shape).copy()
        albedo *= 1 + grain(rng, n, max(0.5, n * 0.006), 0.08)[..., None]
        width = 0.025
        lines = np.zeros(normal.shape[:2], np.float32)
        lines = np.maximum(lines, np.abs(normal @ a) < width)
        lines = np.maximum(lines, np.abs(normal @ b) < width)
        lines = np.maximum(lines, np.abs(np.abs(normal @ c) - 0.72) < width)
        albedo *= (1 - 0.9 * lines)[..., None]
    elif kind == "tennis":
        albedo = np.broadcast_to(hsv_bgr(rng.uniform(30, 40), rng.uniform(170, 235), rng.uniform(190, 250)), normal.shape).copy()
        albedo *= 1 + grain(rng, n, max(0.5, n * 0.004), 0.12)[..., None]
        theta = np.arctan2(normal @ c, normal @ b)
        seam = np.abs(normal @ a - 0.45 * np.cos(2 * theta)) < 0.035
        albedo[seam] = albedo[seam] * 0.2 + 0.8
    elif kind == "baseball":
        albedo = np.broadcast_to(hsv_bgr(20, rng.uniform(10, 35), rng.uniform(215, 245)), normal.shape).copy()
        theta = np.arctan2(normal @ c, normal @ b)
        dist = np.abs(normal @ a - 0.45 * np.cos(2 * theta))
        stitches = (dist < 0.06) & (dist > 0.025) & (np.sin(theta * 34) > 0.2)
        albedo[stitches] = np.array([0.15, 0.15, 0.8], np.float32)
    elif kind == "soccer":
        albedo = np.full(normal.shape, rng.uniform(0.85, 0.97), np.float32)
        phi = (1 + 5 ** 0.5) / 2
        verts = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                verts += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
        verts = np.array([unit(*v) for v in verts]) @ frame
        patches = (normal @ verts.T).max(axis=-1) > 0.935
        albedo[patches] = 0.08
        ks, shininess = 0.25, 30
    elif kind == "beach":
        palette = [hsv_bgr(0, 230, 230), hsv_bgr(28, 230, 250), hsv_bgr(110, 220, 220), np.ones(3, np.float32), hsv_bgr(60, 200, 200), hsv_bgr(14, 230, 250)]
        rng.shuffle(palette)
        theta = np.arctan2(normal @ c, normal @ b)
        segment = ((theta + math.pi) / (2 * math.pi) * 6).astype(int) % 6
        albedo = np.stack(palette)[segment]
        albedo[np.abs(normal @ a) > 0.93] = 0.95
        ks, shininess = 0.45, 50
    else:
        hue = rng.uniform(9, 17)
        albedo = np.broadcast_to(hsv_bgr(hue, rng.uniform(210, 255), rng.uniform(200, 245)), normal.shape).copy()
        pits = grain(rng, n, max(0.6, n * 0.008), 1.0)
        albedo *= (1 + 0.06 * pits)[..., None]
        stem = normal @ unit(rng.uniform(-0.5, 0.5), rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.8)) > 0.985
        albedo[stem] = np.array([0.1, 0.3, 0.2], np.float32)
        ks, shininess = rng.uniform(0.15, 0.35), rng.uniform(12, 30)
        color = shade(normal, albedo, light_dir(rng), ambient, wrap, ks, shininess)
        color *= (1 + 0.05 * pits)[..., None]
        squash = rng.uniform(0.9, 1.0)
        return finish(color, alpha, size, max(4, int(size * squash)))
    color = shade(normal, albedo, light_dir(rng), ambient, wrap, ks, shininess)
    return finish(color, alpha, size)


def render_smiley(rng, size):
    normal, alpha = sphere(size)
    n = normal.shape[0]
    hue = rng.choice([28, 28, 28, 0, 100, 150, 60])
    base = hsv_bgr(hue, rng.uniform(170, 240), rng.uniform(210, 255))
    flat = np.zeros((n, n), np.uint8)
    eye_y = int(n * rng.uniform(0.33, 0.4))
    for side in (-1, 1):
        cv2.ellipse(flat, (int(n / 2 + side * n * 0.17), eye_y), (max(1, int(n * 0.06)), max(1, int(n * 0.1))), 0, 0, 360, 255, -1)
    cv2.ellipse(flat, (n // 2, int(n * 0.55)), (int(n * 0.28), int(n * 0.2)), 0, 15, 165, 255, max(1, int(n * 0.05)))
    albedo = np.broadcast_to(base, normal.shape).copy()
    albedo[flat > 0] = 0.08
    color = shade(normal, albedo, light_dir(rng), 0.6, 0.6, 0.1, 10)
    return finish(color, alpha, size)


def render_clock(rng, size):
    n = max(32, size * 2)
    face = np.full((n, n, 3), rng.uniform(0.85, 0.98), np.float32)
    rim = tuple(float(v) for v in rng.choice([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.1, 0.2, 0.5], [0.15, 0.3, 0.6]]))
    center = (n // 2, n // 2)
    cv2.circle(face, center, int(n * 0.47), rim, max(1, int(n * 0.06)), cv2.LINE_AA)
    for k in range(12):
        angle = k * math.pi / 6
        r1, r2 = n * 0.36, n * 0.42
        p1 = (int(n / 2 + r1 * math.cos(angle)), int(n / 2 + r1 * math.sin(angle)))
        p2 = (int(n / 2 + r2 * math.cos(angle)), int(n / 2 + r2 * math.sin(angle)))
        cv2.line(face, p1, p2, (0.1, 0.1, 0.1), max(1, int(n * 0.015)), cv2.LINE_AA)
    for length, width in ((0.22, 0.03), (0.33, 0.018)):
        angle = rng.uniform(0, 2 * math.pi)
        tip = (int(n / 2 + n * length * math.cos(angle)), int(n / 2 + n * length * math.sin(angle)))
        cv2.line(face, center, tip, (0.08, 0.08, 0.08), max(1, int(n * width)), cv2.LINE_AA)
    alpha = np.zeros((n, n), np.float32)
    cv2.circle(alpha, center, int(n * 0.5) - 1, 1.0, -1, cv2.LINE_AA)
    yy = np.linspace(1, 0, n, dtype=np.float32)[:, None]
    face *= (0.8 + 0.2 * yy)[..., None]
    squash = rng.uniform(0.55, 1.0)
    return finish(face, alpha, max(4, int(size * squash)), size)


def load_faces():
    faces = []
    for path in sorted((BG / "faces").glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None and image.shape[2] == 4:
            faces.append((image[..., :3].astype(np.float32) / 255, image[..., 3].astype(np.float32) / 255))
    return faces


def render_face(rng, size, faces):
    if not faces:
        return render_smiley(rng, size)
    color, alpha = faces[int(rng.integers(len(faces)))]
    h, w = color.shape[:2]
    if rng.random() < 0.5:
        color, alpha = color[:, ::-1], alpha[:, ::-1]
    return finish(np.ascontiguousarray(color), np.ascontiguousarray(alpha), max(4, int(size * w / h)), size)


def motion_blur(color, alpha, length, angle):
    k = int(length) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2] = 1
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0), (k, k))
    kernel /= kernel.sum() + 1e-8
    pad = k // 2 + 1
    premult = np.pad(color * alpha[..., None], ((pad, pad), (pad, pad), (0, 0)))
    alpha = np.pad(alpha, pad)
    premult = cv2.filter2D(premult, -1, kernel)
    alpha = cv2.filter2D(alpha, -1, kernel)
    return premult / np.maximum(alpha, 1e-4)[..., None], alpha, pad


def defocus(color, alpha, sigma):
    pad = int(sigma * 3) + 1
    premult = np.pad(color * alpha[..., None], ((pad, pad), (pad, pad), (0, 0)))
    alpha = np.pad(alpha, pad)
    premult = cv2.GaussianBlur(premult, (0, 0), sigma)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return premult / np.maximum(alpha, 1e-4)[..., None], alpha, pad


def match_light(rng, image, color, cx, cy, size):
    r = int(max(12, size * 1.5))
    patch = image[max(0, int(cy) - r) : int(cy) + r, max(0, int(cx) - r) : int(cx) + r]
    if patch.size == 0:
        return color
    mean = patch.reshape(-1, 3).mean(axis=0)
    gray = float(mean.mean())
    gain = float(np.interp(gray, [15, 120, 240], [0.4, 0.85, 1.08])) * rng.uniform(0.85, 1.12)
    tint = np.clip((mean / max(gray, 1.0)) ** 0.25, 0.85, 1.15)
    return color * gain * tint


def paste(image, color, alpha, x0, y0):
    h, w = alpha.shape
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(WIDTH, x0 + w), min(HEIGHT, y0 + h)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    a = alpha[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, None]
    c = color[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
    region = image[iy0:iy1, ix0:ix1]
    image[iy0:iy1, ix0:ix1] = region * (1 - a) + np.clip(c, 0, 1.3) * 255 * a
    ys, xs = np.nonzero(a[..., 0] > 0.3)
    if len(xs) == 0:
        return None
    return ix0 + xs.min(), iy0 + ys.min(), ix0 + xs.max() + 1, iy0 + ys.max() + 1


def add_shadow(image, x, y, size, strength):
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    cv2.ellipse(mask, (int(x), int(y)), (max(2, int(size * 0.45)), max(1, int(size * 0.13))), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, size * 0.15))
    image *= (1 - strength * mask)[..., None]


def add_light(rng, image, cx, cy, size):
    palette = [(200, 230, 255), (60, 150, 255), (255, 245, 235), (80, 220, 255), (60, 60, 255), (120, 255, 140), (255, 180, 120)]
    color = np.array(palette[int(rng.integers(len(palette)))], np.float32)
    r = int(size * 3) + 4
    yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
    dist = np.sqrt(xx * xx + yy * yy)
    core = np.clip(size / 2 - dist + 1, 0, 1)
    halo = np.exp(-(dist / (size * rng.uniform(0.6, 1.6))) ** 2)
    glow = core * rng.uniform(0.7, 1.0) + halo * rng.uniform(0.2, 0.7)
    x0, y0 = int(cx) - r, int(cy) - r
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(WIDTH, x0 + 2 * r), min(HEIGHT, y0 + 2 * r)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    g = glow[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, None]
    region = image[iy0:iy1, ix0:ix1]
    image[iy0:iy1, ix0:ix1] = region + (np.array([255, 255, 255], np.float32) * 0.6 + color * 0.4 - region * 0.3) * g
    c = core[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, None]
    image[iy0:iy1, ix0:ix1] = np.maximum(image[iy0:iy1, ix0:ix1], (color * 0.9 + 25) * c)


def add_bokeh(rng, image, cx, cy, size):
    hue = rng.choice([100, 110, 60, 150, 170, 25, 0])
    color = hsv_bgr(hue, rng.uniform(40, 200), 255) * 255
    r = int(size / 2) + 4
    yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
    dist = np.sqrt(xx * xx + yy * yy)
    disk = np.clip(size / 2 - dist, 0, 1) * (0.7 + 0.3 * (dist / (size / 2 + 1e-6)) ** 4)
    disk = cv2.GaussianBlur(disk, (0, 0), max(0.6, size * 0.03)) * rng.uniform(0.25, 0.6)
    x0, y0 = int(cx) - r, int(cy) - r
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(WIDTH, x0 + 2 * r), min(HEIGHT, y0 + 2 * r)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    d = disk[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, None]
    image[iy0:iy1, ix0:ix1] += color * d


def load_backgrounds(split):
    tables = json.loads((BG / "tables.json").read_text()) if (BG / "tables.json").exists() else {}
    rejected_path = BG / "rejected.txt"
    rejected = set(rejected_path.read_text().split()) if rejected_path.exists() else set()
    entries = []
    for path in sorted((BG / split).glob("*.jpg")):
        if f"{split}/{path.name}" not in rejected:
            entries.append((path, tables.get(f"{split}/{path.name}")))
    for path in sorted((BG / "commons").glob("*.jpg")):
        index = int(path.stem.split("_")[1])
        if f"commons/{path.name}" not in rejected and (index % 5 == 4) == (split == "val"):
            entries.append((path, None))
    return entries


def background(rng, entry):
    path, table = entry
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    aspect = WIDTH / HEIGHT
    zoom = rng.uniform(0.85, 1.0) if table else rng.uniform(0.6, 1.0)
    if w / h > aspect:
        ch = h * zoom
        cw = ch * aspect
    else:
        cw = w * zoom
        ch = cw / aspect
    x0 = rng.uniform(0, w - cw)
    y0 = rng.uniform(0, h - ch)
    crop = image[int(y0) : int(y0 + ch), int(x0) : int(x0 + cw)]
    out = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA if cw > WIDTH else cv2.INTER_CUBIC)
    flip = rng.random() < 0.5
    if flip:
        out = out[:, ::-1]
    surface = None
    if table:
        scale = WIDTH / cw
        quad = []
        for qx, qy in table["quad"]:
            px = (qx * w - x0) * scale
            py = (qy * h - y0) * scale
            quad.append((WIDTH - px if flip else px, py))
        surface = (quad, table["near"] * scale, table["far"] * scale)
    return np.ascontiguousarray(out).astype(np.float32), surface


def on_surface(rng, surface):
    quad, near, far = surface
    nl, nr, fl, fr = quad
    u, v = rng.uniform(0.08, 0.92), rng.uniform(0.05, 0.95)
    near_pt = (nl[0] * (1 - u) + nr[0] * u, nl[1] * (1 - u) + nr[1] * u)
    far_pt = (fl[0] * (1 - u) + fr[0] * u, fl[1] * (1 - u) + fr[1] * u)
    x = near_pt[0] * (1 - v) + far_pt[0] * v
    y = near_pt[1] * (1 - v) + far_pt[1] * v
    diameter = (near * (1 - v) + far * v) * rng.uniform(0.85, 1.15)
    return x, y, max(MIN_SIDE, diameter)


def ball_size(rng):
    shares = np.array([c[0] for c in SIZE_CLASSES])
    _, low, high = SIZE_CLASSES[int(rng.choice(len(SIZE_CLASSES), p=shares / shares.sum()))]
    return log_uniform(rng, low, high)


def free_spot(rng, size, taken):
    for _ in range(30):
        margin = size * 0.5 if size < 260 else size * 0.15
        x = rng.uniform(margin, WIDTH - margin) if WIDTH > 2 * margin else WIDTH / 2
        y = rng.uniform(margin, HEIGHT - margin) if HEIGHT > 2 * margin else HEIGHT / 2
        if all((x - tx) ** 2 + (y - ty) ** 2 > ((size + ts) * 0.6) ** 2 for tx, ty, ts in taken):
            return x, y
    return None


def place_ball(rng, image, ball, x, y, size, ground):
    color, alpha = ball
    color = match_light(rng, image, color, x, y, size)
    pad = 0
    roll = rng.random()
    blur_chance = 0.5 if size < 150 else 0.2
    if roll < blur_chance:
        length = min(70, max(3, size * rng.uniform(0.2, 1.8)))
        color, alpha, pad = motion_blur(color, alpha, length, rng.uniform(0, 180))
    elif roll < blur_chance + 0.12:
        color, alpha, pad = defocus(color, alpha, max(0.6, size * rng.uniform(0.01, 0.05)))
    h, w = alpha.shape
    x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    ys, xs = np.nonzero(alpha > 0.3)
    full = (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)
    seen = (max(0, full[0]), max(0, full[1]), min(WIDTH, full[2]), min(HEIGHT, full[3]))
    seen_w, seen_h = seen[2] - seen[0], seen[3] - seen[1]
    if seen_w < MIN_SIDE * 0.8 or seen_h < MIN_SIDE * 0.8:
        return None
    if seen_w * seen_h < 0.4 * (full[2] - full[0]) * (full[3] - full[1]):
        return None
    if ground is not None:
        add_shadow(image, x, ground, size, rng.uniform(0.2, 0.5) / (1 + (ground - y) / max(size, 1) * 0.4))
    return paste(image, color, alpha, x0, y0)


def make_ball(rng, size, sprites):
    if sprites and rng.random() < 0.45:
        return photo_orange(rng, sprites[int(rng.integers(len(sprites)))], size)
    return render_orange(rng, size)


def make_distractor(rng, kind, size, faces):
    if kind in ("basketball", "tennis", "baseball", "soccer", "beach", "fruit"):
        return render_patterned(rng, size, kind)
    if kind == "smiley":
        return render_smiley(rng, size)
    if kind == "clock":
        return render_clock(rng, size)
    return render_face(rng, size, faces)


def add_distractors(rng, image, count, taken, faces):
    kinds = list(DISTRACTORS)
    weights = np.array([DISTRACTORS[k] for k in kinds])
    for _ in range(count):
        kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
        low = {"basketball": 40, "fruit": 40, "clock": 30, "smiley": 24, "face": 40}.get(kind, 14)
        size = log_uniform(rng, low, 320)
        spot = free_spot(rng, size, taken)
        if spot is None:
            continue
        x, y = spot
        taken.append((x, y, size))
        if kind == "light":
            add_light(rng, image, x, y, max(6, size * 0.4))
            continue
        if kind == "bokeh":
            add_bokeh(rng, image, x, y, size)
            continue
        size = int(round(size))
        color, alpha = make_distractor(rng, kind, size, faces)
        color = match_light(rng, image, color, x, y, size)
        if rng.random() < 0.25:
            color, alpha, _ = motion_blur(color, alpha, min(50, max(3, size * rng.uniform(0.2, 1.0))), rng.uniform(0, 180))
        h, w = alpha.shape
        paste(image, color, alpha, int(round(x - w / 2)), int(round(y - h / 2)))


def camera(rng, image):
    image = np.clip(image, 0, 255) / 255
    image = image ** rng.uniform(0.7, 1.45)
    image = (image - 0.5) * rng.uniform(0.75, 1.25) + 0.5 + rng.uniform(-0.1, 0.1)
    temp = rng.uniform(-0.12, 0.12)
    image *= np.array([1 - temp, 1 + rng.uniform(-0.04, 0.04), 1 + temp], np.float32)
    noise = rng.uniform(0, 0.015)
    if rng.random() < 0.18:
        image *= rng.uniform(0.3, 0.6)
        noise = rng.uniform(0.02, 0.05)
    if rng.random() < 0.3:
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        dist = ((xx - WIDTH / 2) / WIDTH) ** 2 + ((yy - HEIGHT / 2) / HEIGHT) ** 2
        image *= (1 - rng.uniform(0.2, 0.6) * dist * 2)[..., None]
    roll = rng.random()
    if roll < 0.12:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.6, 2.0))
    elif roll < 0.2:
        k = int(rng.integers(5, 16)) | 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1 / k
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0), (k, k))
        image = cv2.filter2D(image, -1, kernel / (kernel.sum() + 1e-8))
    if rng.random() < 0.12:
        scale = rng.uniform(0.4, 0.7)
        small = cv2.resize(image, (int(WIDTH * scale), int(HEIGHT * scale)), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    image = image + rng.normal(0, noise, image.shape).astype(np.float32)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def make_image(rng, entry, sprites, faces):
    image, surface = background(rng, entry)
    empty = rng.random() < 0.1
    balls = 0 if empty else int(rng.choice([1, 2, 3], p=[0.75, 0.18, 0.07]))
    distractors = int(rng.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.17, 0.08]))
    if empty and rng.random() < 0.7:
        distractors = max(distractors, int(rng.integers(1, 4)))
    taken = []
    plans = []
    for _ in range(balls):
        ground = None
        if surface and rng.random() < 0.7:
            x, ground, size = on_surface(rng, surface)
            lift = 0.0 if rng.random() < 0.4 else size * log_uniform(rng, 0.5, 8)
            y = ground - size * 0.5 - lift
            if y < size * 0.5:
                y, ground = ground - size * 0.5, ground
        else:
            size = ball_size(rng)
            spot = free_spot(rng, size, taken)
            if spot is None:
                continue
            x, y = spot
            if rng.random() < 0.25:
                ground = y + size * 0.5
        if any((x - tx) ** 2 + (y - ty) ** 2 < ((size + ts) * 0.6) ** 2 for tx, ty, ts in taken):
            continue
        taken.append((x, y, size))
        plans.append((x, y, size, ground))
    add_distractors(rng, image, distractors, taken, faces)
    boxes = []
    for x, y, size, ground in plans:
        side = max(MIN_SIDE, int(round(size)))
        box = place_ball(rng, image, make_ball(rng, side, sprites), x, y, side, ground)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if x2 - x1 < MIN_SIDE * 0.8 or y2 - y1 < MIN_SIDE * 0.8:
            continue
        boxes.append(box)
    lines = [
        f"0 {(x1 + x2) / 2 / WIDTH:.6f} {(y1 + y2) / 2 / HEIGHT:.6f} {(x2 - x1) / WIDTH:.6f} {(y2 - y1) / HEIGHT:.6f}\n"
        for x1, y1, x2, y2 in boxes
    ]
    return camera(rng, image), lines


def preview(split, count=24):
    images = sorted((DATA / "images" / split / GROUP).glob("*.jpg"))
    rng = np.random.default_rng(5)
    picks = [images[i] for i in sorted(rng.choice(len(images), size=min(count, len(images)), replace=False))]
    tiles = []
    for path in picks:
        image = cv2.imread(str(path))
        label = DATA / "labels" / split / GROUP / f"{path.stem}.txt"
        for line in label.read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = int((cx - bw / 2) * WIDTH), int((cy - bh / 2) * HEIGHT)
            x2, y2 = int((cx + bw / 2) * WIDTH), int((cy + bh / 2) * HEIGHT)
            cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 3)
        tiles.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    PREVIEW.mkdir(parents=True, exist_ok=True)
    per_sheet = 12
    for sheet in range(0, len(tiles), per_sheet):
        group = tiles[sheet : sheet + per_sheet]
        group += [np.zeros_like(tiles[0])] * (per_sheet - len(group))
        rows = [np.hstack(group[r * 3 : r * 3 + 3]) for r in range(per_sheet // 3)]
        cv2.imwrite(str(PREVIEW / f"{split}_{sheet // per_sheet}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    sprites = cut_sheets()
    faces = load_faces()
    print(f"photo balls {len(sprites)} faces {len(faces)}")
    for split, count in COUNTS.items():
        rng = np.random.default_rng(SEEDS[split])
        pool = [s for i, s in enumerate(sprites) if (i % 4 == 3) == (split == "val")]
        entries = load_backgrounds(split)
        out_images = DATA / "images" / split / GROUP
        out_labels = DATA / "labels" / split / GROUP
        for folder in (out_images, out_labels):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
        total = 0
        for i in range(count):
            image, lines = make_image(rng, entries[int(rng.integers(len(entries)))], pool, faces)
            name = f"orange_{i:04d}"
            cv2.imwrite(str(out_images / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_labels / f"{name}.txt").write_text("".join(lines))
            total += len(lines)
        print(f"{split}: {count} images, {total} balls, {len(entries)} backgrounds, {len(pool)} photo balls")
        preview(split)
    for cache in (DATA / "labels").rglob("*.cache"):
        cache.unlink()


if __name__ == "__main__":
    main()
