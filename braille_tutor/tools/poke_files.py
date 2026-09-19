"""Read the guide dots out of a poke_*.png FILE, and turn them into what poking + flipping the sheet would give."""
from __future__ import annotations

import cv2
import numpy as np

MM = 300 / 25.4


def guide_dots(path: str) -> list:
    """Centres (x, y in mm on the printed paper) of the filled guide circles in a poke template PNG. Text and the pin-prick
    crosses are excluded by size and shape."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dots = []
    for i in range(1, n):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if 25 <= w <= 60 and 25 <= h <= 60 and abs(w - h) <= 4 and area > 0.7 * w * h * 0.785:  # a filled disc about 3 mm across
            dots.append((cents[i][0] / MM, cents[i][1] / MM))
    return dots


def after_flip(dots_mm: list, paper_w_mm: float = 210.0) -> list:
    """Where those dots are on the FACE once the sheet is poked and flipped over left-to-right: the template is mirrored."""
    return [(paper_w_mm - x, y) for x, y in dots_mm]
