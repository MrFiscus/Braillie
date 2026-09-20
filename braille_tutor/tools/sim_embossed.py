"""Simulated photos of embossed braille, with known ground truth, for scoring the readers.

Real embossed dots on white paper show up as a highlight on the side facing the light and a shadow on the other side, with
low contrast. This renders that: a height field of domes, side lighting, paper grain, uneven illumination, blur and JPEG.
"""
from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

# dots of each letter a-z, in standard braille (dot numbers)
LETTER_DOTS = dict(zip("abcdefghijklmnopqrstuvwxyz", (
    "1 12 14 145 15 124 1245 125 24 245 13 123 134 1345 135 1234 12345 1235 234 2345 136 1236 2456 1346 13456 1356").split()))
COL_ROW = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2)}  # dot -> (column, row) inside a cell


def random_text_rows(n_lines: int, cells_per_line: int, seed: int = 0) -> list:
    """Lines of random letters with word gaps (None = blank cell): [[dotset or None, ...], ...]."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_lines):
        row = []
        while len(row) < cells_per_line:
            row += [frozenset(int(d) for d in LETTER_DOTS[rng.choice("abcdefghijklmnopqrstuvwxyz")]) for _ in range(rng.randint(2, 6))]
            row.append(None)
        rows.append(row[:cells_per_line])
    return rows


def render_relief(rows: list, dot_mm: float, pitch_mm: float, line_mm: float, px_per_mm: float, size_mm: tuple,
                  origin_mm: tuple = (10, 10), contrast: float = 35.0, seed: int = 0, light=(-1.0, -1.0)) -> tuple:
    """(gray image, truth) where truth is [(row, col, dotset)] for every non-blank cell. Light comes from the upper left."""
    w, h = int(size_mm[0] * px_per_mm), int(size_mm[1] * px_per_mm)
    height = np.zeros((h, w), np.float32)
    r_px = 0.5 * 0.6 * dot_mm * px_per_mm  # dot diameter is ~60% of the dot spacing in standard braille
    truth = []
    for ri, row in enumerate(rows):
        for ci, dots in enumerate(row):
            if not dots:
                continue
            truth.append((ri, ci, frozenset(dots)))
            for d in dots:
                col, rr = COL_ROW[d]
                cx = (origin_mm[0] + ci * pitch_mm + col * dot_mm) * px_per_mm
                cy = (origin_mm[1] + ri * line_mm + rr * dot_mm) * px_per_mm
                cv2.circle(height, (int(round(cx)), int(round(cy))), int(round(r_px)), 1.0, -1, cv2.LINE_AA)
    height = cv2.GaussianBlur(height, (0, 0), max(1.0, 0.35 * r_px))  # rounded domes
    gx, gy = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    lx, ly = np.array(light, np.float32) / np.linalg.norm(light)
    shade = -(gx * lx + gy * ly)
    shade = shade / (np.abs(shade).max() + 1e-6)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    illum = 1.0 + 0.10 * np.sin(xx / w * 2.0) + 0.12 * (yy / h - 0.5) - 0.10 * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2))
    paper = 225.0 + rng.normal(0, 2.5, (h, w)).astype(np.float32)
    paper = cv2.GaussianBlur(paper, (0, 0), 0.8)
    img = (paper + contrast * shade - 0.35 * contrast * height) * illum
    return np.clip(img, 0, 255).astype(np.uint8), truth


def camera_view(gray: np.ndarray, out_size=(1280, 960), quad=None, blur: float = 1.0, noise: float = 2.0, jpeg: int = 80,
                seed: int = 0) -> tuple:
    """Photograph the flat image: perspective warp onto a dark desk, blur, sensor noise, JPEG. Returns (BGR image, T) where
    T maps the flat image's pixels to camera pixels."""
    h, w = gray.shape
    quad = np.float32(quad if quad is not None else [[160, 60], [1080, 100], [1120, 900], [110, 860]])
    T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
    rng = np.random.default_rng(seed)
    desk = cv2.GaussianBlur(np.full(out_size[::-1], 55, np.float32) + rng.normal(0, 5, out_size[::-1]), (0, 0), 2)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T, out_size)
    warped = cv2.warpPerspective(gray, T, out_size, flags=cv2.INTER_AREA)
    img = np.where(mask > 0, warped, desk).astype(np.float32)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR), T


def render_relief_points(points_mm: list, dot_r_mm: float, px_per_mm: float, size_mm: tuple, contrast: float = 30.0,
                         seed: int = 0, light=(-1.0, -1.0)) -> np.ndarray:
    """Embossed dots at arbitrary (x, y) millimetre positions, same lighting model as render_relief."""
    w, h = int(size_mm[0] * px_per_mm), int(size_mm[1] * px_per_mm)
    height = np.zeros((h, w), np.float32)
    r_px = dot_r_mm * px_per_mm
    for x, y in points_mm:
        cv2.circle(height, (int(round(x * px_per_mm)), int(round(y * px_per_mm))), int(round(r_px)), 1.0, -1, cv2.LINE_AA)
    height = cv2.GaussianBlur(height, (0, 0), max(1.0, 0.35 * r_px))
    gx, gy = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    lx, ly = np.array(light, np.float32) / np.linalg.norm(light)
    shade = -(gx * lx + gy * ly)
    shade = shade / (np.abs(shade).max() + 1e-6)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    illum = 1.0 + 0.10 * np.sin(xx / w * 2.0) + 0.12 * (yy / h - 0.5) - 0.10 * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2))
    paper = cv2.GaussianBlur(225.0 + rng.normal(0, 2.5, (h, w)).astype(np.float32), (0, 0), 0.8)
    return np.clip((paper + contrast * shade - 0.35 * contrast * height) * illum, 0, 255).astype(np.uint8)
