"""Save the four page markers as PNGs plus one printable A4 sheet. Run: python make_markers.py"""
from pathlib import Path

import cv2
import numpy as np

from page import ARUCO_DICT

OUT = Path("markers")
MARKER_PX = 400  # marker size in the individual PNGs, before the border
BORDER_PX = 100  # white quiet zone around each PNG (ArUco needs one to be detected)
DPI = 300
MM = DPI / 25.4
SHEET_PX = (int(210 * MM), int(297 * MM))  # A4, (w, h)
SHEET_MARKER_MM = 40  # printed marker side; the white border adds 10 mm on each side
LABELS = {0: "top left", 1: "top right", 2: "bottom right", 3: "bottom left"}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    sheet = np.full((SHEET_PX[1], SHEET_PX[0]), 255, np.uint8)
    size = int(SHEET_MARKER_MM * MM)
    pad = int(10 * MM)
    cell = size + 2 * pad
    for i in range(4):
        img = cv2.aruco.generateImageMarker(d, i, MARKER_PX)
        img = cv2.copyMakeBorder(img, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(OUT / f"marker_{i}.png"), img)
        # 2x2 grid on the sheet, well apart so you can cut them out
        x0 = int(20 * MM) + (i % 2) * (cell + int(20 * MM))
        y0 = int(30 * MM) + (i // 2) * (cell + int(30 * MM))
        big = cv2.aruco.generateImageMarker(d, i, size)
        sheet[y0 + pad : y0 + pad + size, x0 + pad : x0 + pad + size] = big
        cv2.putText(sheet, f"id {i} - {LABELS[i]}", (x0, y0 + cell + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.imwrite(str(OUT / "sheet.png"), sheet)
    print(f"wrote {OUT}/marker_0..3.png and {OUT}/sheet.png (print at 100% scale, A4)")


if __name__ == "__main__":
    main()
