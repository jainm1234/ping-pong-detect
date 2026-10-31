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
COUNTS = {"orange": {"train": 800, "val": 200}, "pov": {"train": 240, "val": 60}}
TABLE_W, TABLE_L, BALL_MM = 1525.0, 2740.0, 40.0
SEEDS = {"train": 11, "val": 23}
WIDTH, HEIGHT = 1280, 720
MIN_SIDE = 12
SHEETS = ["orange_balls_a.jpg", "orange_balls_b.jpg", "orange_balls_c.jpg", "orange_commons.jpg"]

# (share, smallest diameter, largest diameter) in output pixels
SIZE_CLASSES = [(0.32, 12, 30), (0.36, 30, 100), (0.24, 100, 260), (0.08, 260, 560)]
DISTRACTORS = {
    "light": 0.28,
    "clock": 0.16,
    "sign": 0.2,
    "plate": 0.13,
    "wheel": 0.13,
    "button": 0.1,
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
    hue = rng.triangular(8, 13, 19)
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
            sprites.append((crop, alpha))
    return sprites


def is_orange(sprite):
    crop, alpha = sprite
    hsv = cv2.cvtColor((np.clip(crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    inside = alpha > 0.9
    return np.median(hsv[..., 0][inside]) >= 9 and np.median(hsv[..., 2][inside]) >= 190


def photo_orange(rng, sprite, size):
    crop, alpha = sprite
    side = crop.shape[0]
    matrix = cv2.getRotationMatrix2D((side / 2, side / 2), float(rng.uniform(0, 360)), 1.0)
    crop = cv2.warpAffine(crop, matrix, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    alpha = cv2.warpAffine(alpha, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    hsv = cv2.cvtColor((np.clip(crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-2, 2)) % 180
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    hsv[..., 2] *= rng.uniform(0.85, 1.1)
    crop = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    color, alpha = finish(crop, np.ascontiguousarray(alpha), size)
    normal, _ = sphere(size, 1)
    relight = 1 + rng.uniform(0, 0.35) * (normal @ light_dir(rng) - 0.3)
    relight = cv2.resize(relight.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return color * relight[..., None], alpha


def render_flat(rng, size, kind):
    n = max(32, size * 2)
    c = n // 2
    r = int(n * 0.48)
    canvas = np.zeros((n, n, 3), np.float32)
    if kind == "sign":
        if rng.random() < 0.55:
            red = hsv_bgr(rng.uniform(175, 180), rng.uniform(200, 250), rng.uniform(170, 230))
            cv2.circle(canvas, (c, c), r, red.tolist(), -1, cv2.LINE_AA)
            if rng.random() < 0.5:
                cv2.rectangle(canvas, (int(c - r * 0.62), int(c - r * 0.16)), (int(c + r * 0.62), int(c + r * 0.16)), (0.95, 0.95, 0.95), -1)
            else:
                cv2.circle(canvas, (c, c), int(r * 0.78), (0.95, 0.95, 0.95), -1, cv2.LINE_AA)
                label = str(rng.choice(["20", "30", "40", "50", "60", "P"]))
                scale = n / 90
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, max(1, n // 40))
                cv2.putText(canvas, label, (c - tw // 2, c + th // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0.05, 0.05, 0.05), max(1, n // 40), cv2.LINE_AA)
        else:
            cv2.circle(canvas, (c, c), r, hsv_bgr(rng.uniform(105, 115), 230, rng.uniform(150, 210)).tolist(), -1, cv2.LINE_AA)
            cv2.circle(canvas, (c, c), int(r * 0.93), (0.95, 0.95, 0.95), max(1, n // 50), cv2.LINE_AA)
            tip, tail = (c, int(c - r * 0.6)), (c, int(c + r * 0.6))
            cv2.arrowedLine(canvas, tail, tip, (0.95, 0.95, 0.95), max(2, n // 12), cv2.LINE_AA, tipLength=0.4)
    elif kind == "plate":
        base = rng.choice([[0.95, 0.95, 0.95], [0.85, 0.9, 0.93], [0.7, 0.8, 0.6], [0.8, 0.65, 0.6], [0.3, 0.3, 0.3]])
        cv2.circle(canvas, (c, c), r, (np.array(base) * 0.9).tolist(), -1, cv2.LINE_AA)
        cv2.circle(canvas, (c, c), int(r * 0.68), np.array(base).tolist(), -1, cv2.LINE_AA)
        if rng.random() < 0.4:
            cv2.circle(canvas, (c, c), int(r * 0.9), hsv_bgr(rng.choice([60, 100, 120, 150]), 160, 160).tolist(), max(1, n // 60), cv2.LINE_AA)
    elif kind == "wheel":
        cv2.circle(canvas, (c, c), r, (0.07, 0.07, 0.07), -1, cv2.LINE_AA)
        rim = float(rng.uniform(0.45, 0.8))
        cv2.circle(canvas, (c, c), int(r * 0.62), (rim, rim, rim), -1, cv2.LINE_AA)
        spokes = int(rng.integers(5, 9))
        start = rng.uniform(0, 2 * math.pi)
        for k in range(spokes):
            angle = start + k * 2 * math.pi / spokes
            hole = (int(c + r * 0.38 * math.cos(angle)), int(c + r * 0.38 * math.sin(angle)))
            cv2.circle(canvas, hole, max(1, int(r * 0.13)), (0.1, 0.1, 0.1), -1, cv2.LINE_AA)
        cv2.circle(canvas, (c, c), max(1, int(r * 0.14)), (rim * 0.7,) * 3, -1, cv2.LINE_AA)
    else:
        hue = float(rng.choice([60, 90, 110, 125, 145, 165]))
        color = hsv_bgr(hue, rng.uniform(0, 230), rng.uniform(40, 240))
        cv2.circle(canvas, (c, c), r, color.tolist(), -1, cv2.LINE_AA)
        cv2.circle(canvas, (c, c), int(r * 0.85), (color * 0.8).tolist(), max(1, n // 40), cv2.LINE_AA)
        holes = int(rng.choice([2, 4]))
        for k in range(holes):
            angle = k * 2 * math.pi / holes + math.pi / 4
            hole = (int(c + r * 0.25 * math.cos(angle)), int(c + r * 0.25 * math.sin(angle)))
            cv2.circle(canvas, hole, max(1, int(r * 0.08)), (color * 0.35).tolist(), -1, cv2.LINE_AA)
    alpha = np.zeros((n, n), np.float32)
    cv2.circle(alpha, (c, c), r, 1.0, -1, cv2.LINE_AA)
    yy = np.linspace(1, 0, n, dtype=np.float32)[:, None]
    canvas *= (rng.uniform(0.75, 0.9) + rng.uniform(0.1, 0.25) * yy)[..., None]
    canvas *= 1 + grain(rng, n, max(0.5, n * 0.01), 0.03)[..., None]
    squash = rng.uniform(0.4, 1.0) if kind in ("plate", "wheel") else rng.uniform(0.7, 1.0)
    return finish(canvas, alpha, size, max(4, int(size * squash)))


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


def line_kernel(length, angle):
    k = int(length) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2] = 1
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0), (k, k))
    return kernel / (kernel.sum() + 1e-8)


def blur_sprite(color, alpha, pad, blur):
    premult = blur(np.pad(color * alpha[..., None], ((pad, pad), (pad, pad), (0, 0))))
    alpha = blur(np.pad(alpha, pad))
    return premult / np.maximum(alpha, 1e-4)[..., None], alpha


def motion_blur(color, alpha, length, angle):
    kernel = line_kernel(length, angle)
    return blur_sprite(color, alpha, kernel.shape[0] // 2 + 1, lambda x: cv2.filter2D(x, -1, kernel))


def defocus(color, alpha, sigma):
    return blur_sprite(color, alpha, int(sigma * 3) + 1, lambda x: cv2.GaussianBlur(x, (0, 0), sigma))


def match_light(rng, image, color, cx, cy, size, floor=0.6):
    r = int(max(12, size * 1.5))
    patch = image[max(0, int(cy) - r) : int(cy) + r, max(0, int(cx) - r) : int(cx) + r]
    if patch.size == 0:
        return color
    mean = patch.reshape(-1, 3).mean(axis=0)
    gray = float(mean.mean())
    gain = float(np.interp(gray, [15, 120, 240], [floor, max(floor, 0.85), 1.08])) * rng.uniform(0.85, 1.12)
    tint = np.clip((mean / max(gray, 1.0)) ** 0.2, 0.92, 1.08)
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
    r = int(size * 8) + 4
    yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
    dist = np.sqrt(xx * xx + yy * yy)
    core = np.clip(size / 2 - dist + 1, 0, 1)
    halo = np.exp(-dist / (size * rng.uniform(1.0, 2.5))) * np.clip(1 - dist / r, 0, 1) ** 2
    glow = np.clip(core + halo * rng.uniform(0.5, 0.9), 0, 1)
    x0, y0 = int(cx) - r, int(cy) - r
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(WIDTH, x0 + 2 * r), min(HEIGHT, y0 + 2 * r)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    g = glow[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, None]
    region = image[iy0:iy1, ix0:ix1]
    image[iy0:iy1, ix0:ix1] = region + (np.array([255, 255, 255], np.float32) * 0.6 + color * 0.4 - region * 0.3) * g
    c = core[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, None]
    image[iy0:iy1, ix0:ix1] = np.maximum(image[iy0:iy1, ix0:ix1], (color * 0.9 + 25) * c)


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


def crowded(x, y, size, taken):
    return any((x - tx) ** 2 + (y - ty) ** 2 < ((size + ts) * 0.6) ** 2 for tx, ty, ts in taken)


def free_spot(rng, size, taken):
    for _ in range(30):
        margin = size * 0.5 if size < 260 else size * 0.15
        x = rng.uniform(margin, WIDTH - margin) if WIDTH > 2 * margin else WIDTH / 2
        y = rng.uniform(margin, HEIGHT - margin) if HEIGHT > 2 * margin else HEIGHT / 2
        if not crowded(x, y, size, taken):
            return x, y
    return None


def put_ball(image, color, alpha, x, y, shadow=None):
    h, w = alpha.shape
    x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    ys, xs = np.nonzero(alpha > 0.3)
    full = (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)
    seen_w = min(WIDTH, full[2]) - max(0, full[0])
    seen_h = min(HEIGHT, full[3]) - max(0, full[1])
    if seen_w < MIN_SIDE * 0.8 or seen_h < MIN_SIDE * 0.8:
        return None
    if seen_w * seen_h < 0.4 * (full[2] - full[0]) * (full[3] - full[1]):
        return None
    if shadow is not None:
        add_shadow(image, x, *shadow)
    return paste(image, color, alpha, x0, y0)


def yolo_lines(boxes):
    return [
        f"0 {(x1 + x2) / 2 / WIDTH:.6f} {(y1 + y2) / 2 / HEIGHT:.6f} {(x2 - x1) / WIDTH:.6f} {(y2 - y1) / HEIGHT:.6f}\n"
        for x1, y1, x2, y2 in boxes
    ]


def make_ball(rng, size, sprites):
    if sprites and rng.random() < 0.45:
        return photo_orange(rng, sprites[int(rng.integers(len(sprites)))], size)
    return render_orange(rng, size)


def make_distractor(rng, kind, size):
    if kind == "clock":
        return render_clock(rng, size)
    return render_flat(rng, size, kind)


def add_distractors(rng, image, count, taken, light_top=None):
    kinds = list(DISTRACTORS)
    weights = np.array([DISTRACTORS[k] for k in kinds])
    for _ in range(count):
        kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
        low = {"light": 20, "clock": 30, "button": 20, "plate": 60}.get(kind, 40)
        size = log_uniform(rng, low, 320)
        spot = free_spot(rng, size, taken)
        if spot is None:
            continue
        x, y = spot
        if kind == "light" and light_top is not None and y > light_top:
            continue
        taken.append((x, y, size))
        if kind == "light":
            add_light(rng, image, x, y, min(24, max(5, size * 0.12)))
            continue
        size = int(round(size))
        color, alpha = make_distractor(rng, kind, size)
        color = match_light(rng, image, color, x, y, size)
        if rng.random() < 0.25:
            color, alpha = motion_blur(color, alpha, min(50, max(3, size * rng.uniform(0.2, 1.0))), rng.uniform(0, 180))
        h, w = alpha.shape
        paste(image, color, alpha, int(round(x - w / 2)), int(round(y - h / 2)))


def camera(rng, image):
    image = np.clip(image, 0, 255) / 255
    image = image ** rng.uniform(0.75, 1.25)
    image = (image - 0.5) * rng.uniform(0.75, 1.25) + 0.5 + rng.uniform(-0.1, 0.1)
    temp = rng.uniform(-0.07, 0.07)
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
        image = cv2.filter2D(image, -1, line_kernel(rng.integers(5, 16), rng.uniform(0, 180)))
    if rng.random() < 0.12:
        scale = rng.uniform(0.4, 0.7)
        small = cv2.resize(image, (int(WIDTH * scale), int(HEIGHT * scale)), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    image = image + rng.normal(0, noise, image.shape).astype(np.float32)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def make_image(rng, entry, sprites):
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
        if crowded(x, y, size, taken):
            continue
        taken.append((x, y, size))
        plans.append((x, y, size, ground))
    light_top = min(surface[0][2][1], surface[0][3][1]) - 20 if surface else None
    add_distractors(rng, image, distractors, taken, light_top)
    boxes = []
    for x, y, size, ground in plans:
        side = max(MIN_SIDE, int(round(size)))
        color, alpha = make_ball(rng, side, sprites)
        color = match_light(rng, image, color, x, y, side)
        roll = rng.random()
        blur_chance = 0.5 if side < 150 else 0.2
        if roll < blur_chance:
            color, alpha = motion_blur(color, alpha, min(70, max(3, side * rng.uniform(0.2, 1.8))), rng.uniform(0, 180))
        elif roll < blur_chance + 0.12:
            color, alpha = defocus(color, alpha, max(0.6, side * rng.uniform(0.01, 0.05)))
        shadow = None
        if ground is not None:
            shadow = (ground, side, rng.uniform(0.2, 0.5) / (1 + (ground - y) / side * 0.4))
        box = put_ball(image, color, alpha, x, y, shadow)
        if box is not None:
            boxes.append(box)
    return camera(rng, image), yolo_lines(boxes)


def load_pov(split):
    meta = json.loads((BG / "pov" / "pov.json").read_text())
    return [(BG / "pov" / name, m["quad"]) for name, m in sorted(meta.items()) if m["split"] == split]


def pov_background(rng, entry):
    path, quad = entry
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    zoom = rng.uniform(0.7, 1.0)
    cw, ch = w * zoom, h * zoom
    x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
    crop = image[int(y0) : int(y0 + ch), int(x0) : int(x0 + cw)]
    out = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    flip = rng.random() < 0.5
    points = []
    for qx, qy in quad:
        px, py = (qx * w - x0) * WIDTH / cw, (qy * h - y0) * HEIGHT / ch
        points.append((WIDTH - px if flip else px, py))
    if flip:
        out = out[:, ::-1]
        points = [points[1], points[0], points[3], points[2]]
    table = np.float32([[0, 0], [TABLE_W, 0], [0, TABLE_L], [TABLE_W, TABLE_L]])
    homography = cv2.getPerspectiveTransform(table, np.float32(points))
    return np.ascontiguousarray(out).astype(np.float32), homography


def project(homography, x, y):
    return cv2.perspectiveTransform(np.float32([[[x, y]]]), homography)[0, 0]


def pov_spot(rng, homography):
    for _ in range(30):
        x = rng.uniform(-300, TABLE_W + 300)
        zone = rng.random()
        if zone < 0.4:
            y = -log_uniform(rng, 50, 1600)
        elif zone < 0.75:
            y = rng.uniform(0, TABLE_L / 2)
        else:
            y = rng.uniform(TABLE_L / 2, TABLE_L + 600)
        on_table = 0 <= x <= TABLE_W and 0 <= y <= TABLE_L
        z = 0.0 if on_table and rng.random() < 0.25 else log_uniform(rng, 40, 900 if y < 0 else 700)
        gx, gy = project(homography, x, y)
        px_per_mm = np.linalg.norm(project(homography, x + 200, y) - project(homography, x - 200, y)) / 400
        size = BALL_MM * px_per_mm * rng.uniform(0.95, 1.05)
        cy = gy - (z + BALL_MM / 2) * px_per_mm
        if 0 < gx < WIDTH and 0 < cy < HEIGHT and size < 400:
            toward = project(homography, x, y - 300) - np.array([gx, gy])
            return gx, cy, size, gy if on_table else None, z * px_per_mm, toward
    return None


def make_pov_image(rng, entry, sprites):
    image, homography = pov_background(rng, entry)
    empty = rng.random() < 0.08
    balls = 0 if empty else int(rng.choice([1, 2], p=[0.88, 0.12]))
    taken = []
    plans = []
    for _ in range(balls):
        spot = pov_spot(rng, homography)
        if spot is None:
            continue
        x, y, size, ground, lift, toward = spot
        if crowded(x, y, size, taken):
            continue
        taken.append((x, y, size))
        plans.append(spot)
    far_y = min(project(homography, 0, TABLE_L)[1], project(homography, TABLE_W, TABLE_L)[1])
    add_distractors(rng, image, int(rng.choice([0, 1], p=[0.75, 0.25])) if not empty else 1, taken, far_y - 20)
    boxes = []
    for x, y, size, ground, lift, toward in plans:
        side = max(MIN_SIDE, int(round(size)))
        color, alpha = make_ball(rng, side, sprites)
        color = match_light(rng, image, color, x, y, side, floor=0.7)
        if rng.random() < 0.55:
            angle = -math.degrees(math.atan2(toward[1] - rng.uniform(0, 1) * np.linalg.norm(toward), toward[0]))
            length = min(90, max(3, side * rng.uniform(0.3, 1.5 if side < 25 else 2.5)))
            color, alpha = motion_blur(color, alpha, length, angle + rng.uniform(-25, 25))
        shadow = None
        if ground is not None:
            shadow = (ground, side, rng.uniform(0.25, 0.5) / (1 + lift / side * 0.15))
        box = put_ball(image, color, alpha, x, y, shadow)
        if box is not None:
            boxes.append(box)
    return camera(rng, image), yolo_lines(boxes)


def preview(split, group, count=24):
    images = sorted((DATA / "images" / split / group).glob("*.jpg"))
    rng = np.random.default_rng(5)
    picks = [images[i] for i in sorted(rng.choice(len(images), size=min(count, len(images)), replace=False))]
    tiles = []
    for path in picks:
        image = cv2.imread(str(path))
        label = DATA / "labels" / split / group / f"{path.stem}.txt"
        for line in label.read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = int((cx - bw / 2) * WIDTH), int((cy - bh / 2) * HEIGHT)
            x2, y2 = int((cx + bw / 2) * WIDTH), int((cy + bh / 2) * HEIGHT)
            cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 3)
        tiles.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    PREVIEW.mkdir(parents=True, exist_ok=True)
    per_sheet = 12
    for sheet in range(0, len(tiles), per_sheet):
        page = tiles[sheet : sheet + per_sheet]
        page += [np.zeros_like(tiles[0])] * (per_sheet - len(page))
        rows = [np.hstack(page[r * 3 : r * 3 + 3]) for r in range(per_sheet // 3)]
        cv2.imwrite(str(PREVIEW / f"{split}_{group}_{sheet // per_sheet}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def output_folders(split, group):
    folders = (DATA / "images" / split / group, DATA / "labels" / split / group)
    for folder in folders:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    return folders


def write_group(rng, split, group, make, entries, pool):
    out_images, out_labels = output_folders(split, group)
    count, total = COUNTS[group][split], 0
    for i in range(count):
        image, lines = make(rng, entries[int(rng.integers(len(entries)))], pool)
        name = f"{group}_{i:04d}"
        cv2.imwrite(str(out_images / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_labels / f"{name}.txt").write_text("".join(lines))
        total += len(lines)
    print(f"{split} {group}: {count} images, {total} balls, {len(entries)} backgrounds, {len(pool)} photo balls")
    preview(split, group)


def main():
    sprites = cut_sheets()
    print(f"photo balls {len(sprites)}")
    for split, seed in SEEDS.items():
        rng = np.random.default_rng(seed)
        pool = [s for i, s in enumerate(sprites) if is_orange(s) and (i % 4 == 3) == (split == "val")]
        write_group(rng, split, "orange", make_image, load_backgrounds(split), pool)
        write_group(rng, split, "pov", make_pov_image, load_pov(split), pool)
    for cache in (DATA / "labels").rglob("*.cache"):
        cache.unlink()


if __name__ == "__main__":
    main()
