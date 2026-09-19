"""Live camera check for the A-Z sheet: draws the layout over the camera image, click to see the cell under the cursor.

Run: python check_sheet.py [--camera N|URL] [--calib calibration.json] [--cells cells.json]
Keys: s = save check.png, q = quit. If the drawn dots sit on your real dots, markers, scale and layout all agree.
--cells checks a scanned/edited cells.json (boxes and dot numbers) instead of the built-in A-Z sheet.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

import make_sheet
from detect import (GREEN, RED, annotate, draw_hud, draw_page_outline, load_cells, nearest_cell, open_camera,
                    page_source_from_args, page_status)
from page import to_image, to_page, visible_markers


def draw_overlay(frame: np.ndarray, H: np.ndarray, cells: list) -> np.ndarray:
    """Project every layout dot onto the camera image (green) and label each cell with its letter."""
    out = frame.copy()
    for c in cells:
        for x, y in make_sheet.dot_points(c):
            cv2.circle(out, tuple(int(v) for v in to_image(H, x, y)), 5, (0, 220, 0), 2)
        lx, ly = to_image(H, c["x"], c["y"] + 2 * make_sheet.DOT_MM + 4)
        letter = make_sheet.ROWS[c["row"]][c["col"]].upper()
        cv2.putText(out, letter, (int(lx) - 8, int(ly)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def describe(H: np.ndarray, cells: list, px: int, py: int, letters: bool = True) -> str:
    """What the quiz would say if the fingertip were at image pixel (px, py)."""
    x, y = to_page(H, px, py)
    c = nearest_cell(cells, x, y)
    dots = "" if c is None else "dots " + "".join(map(str, sorted(c["dots"])))
    what = "no cell" if c is None else (f"{make_sheet.ROWS[c['row']][c['col']].upper()} ({dots})" if letters else dots)
    return f"({x:.0f}, {y:.0f}) mm -> {what}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default=None, help="index, URL or video file (default: first that works)")
    ap.add_argument("--calib", help="calibration.json from calibrate.py, instead of the four markers")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"), help="find the page edges automatically")
    ap.add_argument("--cells", help="cells.json to check instead of the built-in A-Z sheet")
    a = ap.parse_args()
    cap, click = open_camera(a.camera), []
    cells = load_cells(a.cells) if a.cells else make_sheet.sheet_cells()
    page_src = page_source_from_args(a)
    cv2.namedWindow("check")
    cv2.setMouseCallback("check", lambda ev, x, y, *_: click.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, msg, page_ok = page_status(frame, page_src)
        if H is None:
            view = frame.copy()
        else:
            view = annotate(frame, cells, H) if a.cells else draw_overlay(frame, H, cells)
        for i, (mx, my) in visible_markers(frame).items():  # show exactly which markers the camera found
            cv2.circle(view, (int(mx), int(my)), 12, (255, 0, 255), 3)
            cv2.putText(view, f"id {i}", (int(mx) + 14, int(my) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        if a.auto_page and H is not None:
            draw_page_outline(view, H, *a.auto_page)
        lines = [(msg, GREEN if page_ok else RED)]
        if H is not None and click:
            cv2.circle(view, click[-1], 6, (255, 0, 0), -1)
            lines.append((describe(H, cells, *click[-1], letters=not a.cells), (255, 128, 0)))
        draw_hud(view, lines)
        cv2.imshow("check", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            cv2.imwrite("check.png", view)
            print("saved check.png")
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
