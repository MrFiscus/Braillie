"""Live camera check for the A-Z sheet: draws the layout over the camera image, click to see the cell under the cursor.

Run: python check_sheet.py [--camera N|URL] [--calib calibration.json] [--cells cells.json]
Keys: s = save check.png, u = unlock and read again, d = show/hide the live reading while the page is registered, q = quit. If the drawn dots sit on your real dots, markers, scale and layout all agree.
--cells checks a scanned/edited cells.json (boxes and dot numbers) instead of the built-in A-Z sheet.
"""
from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np

import make_sheet
from sheets import SHEET_NAMES, get_sheet
from detect import (GREEN, RED, _Detector, annotate, braille_status, cells_from_boxes, draw_detections, draw_hud,
                    draw_page_outline, load_cells, locked_check_line, nearest_cell, open_camera,
                    page_source_from_args, page_status)
from page import to_image, to_page, visible_markers
from vote import CellLocker


def draw_overlay(frame: np.ndarray, H: np.ndarray, cells: list, labels: Optional[dict] = None, states: Optional[dict] = None) -> np.ndarray:
    """Project every layout dot onto the camera image and label each cell with its letter.

    states maps (row, col) to "locked" (read correctly and held: solid green), "wrong" (red) or "reading" (amber); without it
    everything is plain green."""
    out = frame.copy()
    palette = {"locked": (0, 200, 0), "wrong": (0, 0, 255), "reading": (0, 165, 255)}
    for c in cells:
        state = (states or {}).get((c["row"], c["col"]))
        color = palette.get(state, (0, 220, 0))
        for x, y in make_sheet.dot_points(c):
            cv2.circle(out, tuple(int(v) for v in to_image(H, x, y)), 5, color, -1 if state == "locked" else 2)
        lx, ly = to_image(H, c["x"], c["y"] + 2 * make_sheet.DOT_MM + 4)
        letter = labels[(c["row"], c["col"])] if labels else make_sheet.ROWS[c["row"]][c["col"]].upper()
        cv2.putText(out, letter, (int(lx) - 8, int(ly)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color if state else (0, 0, 255), 2)
    return out


def describe(H: np.ndarray, cells: list, px: int, py: int, letters: bool = True, labels: Optional[dict] = None) -> str:
    """What the quiz would say if the fingertip were at image pixel (px, py)."""
    x, y = to_page(H, px, py)
    c = nearest_cell(cells, x, y)
    dots = "" if c is None else "dots " + "".join(map(str, sorted(c["dots"])))
    what = "no cell" if c is None else (f"{(labels or {}).get((c['row'], c['col'])) or make_sheet.ROWS[c['row']][c['col']].upper()} ({dots})" if letters else dots)
    return f"({x:.0f}, {y:.0f}) mm -> {what}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default=None, help="index, URL or video file (default: first that works)")
    ap.add_argument("--calib", help="calibration.json from calibrate.py, instead of the four markers")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"), help="find the page edges automatically")
    ap.add_argument("--markers-only", action="store_true", help="require all four markers in every frame")
    ap.add_argument("--paper", action="store_true", help="no markers: find the printed A4 sheet's own edges")
    ap.add_argument("--sheet", choices=SHEET_NAMES, default="alphabet", help="which printed sheet's layout to overlay")
    ap.add_argument("--no-detect", action="store_true", help="don't run the detector (faster; no live reading on screen)")
    ap.add_argument("--cells", help="cells.json to check instead of the built-in A-Z sheet")
    a = ap.parse_args()
    cap, click = open_camera(a.camera), []
    sheet = get_sheet(a.sheet)
    labels = {k: s.short for k, s in sheet.names.items()}
    cells = load_cells(a.cells) if a.cells else sheet.cells
    page_src = page_source_from_args(a)
    det = None if a.no_detect else _Detector(0.15, "auto", sheet=None if a.cells else cells)  # reads the sheet in the background (by observation when the page is located)
    show_reading = False  # hidden while the page is registered, so the good layout overlay stays clean; press d to show it
    locker, results, seen_version = CellLocker(), [], -1  # cells that read right are held, so a shaky frame cannot flicker them
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
            states = None
            if results:
                states = {(e["row"], e["col"]): "locked" if r["locked"] else ("wrong" if r["dots"] != e["dots"] else "reading")
                          for r, e in zip(results, cells)}
            view = annotate(frame, cells, H) if a.cells else draw_overlay(frame, H, cells, labels, states)
        for i, (mx, my) in visible_markers(frame).items():  # show exactly which markers the camera found
            cv2.circle(view, (int(mx), int(my)), 12, (255, 0, 255), 3)
            cv2.putText(view, f"id {i}", (int(mx) + 14, int(my) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        if hasattr(page_src, "size_mm") and H is not None:
            draw_page_outline(view, H, *page_src.size_mm, page_src.origin)
        lines = [(msg, GREEN if page_ok else RED)]
        if det is not None:
            det.submit(frame, H if page_ok else None)
            if det.sheet is not None and det.observed and det.version != seen_version:
                seen_version = det.version
                results = locker.update_known(det.observed, cells)  # locks a cell once it reads as the sheet says
            if show_reading or H is None:
                n_read = draw_detections(view, det.boxes)  # cyan boxes + the letters it read
            else:
                n_read = len(cells_from_boxes(det.boxes, np.eye(3)))  # still counted, just not drawn
            lines.append(braille_status(det, n_read))
            if results and H is not None:
                lines.append(locked_check_line(results, cells))
        if H is not None and click:
            cv2.circle(view, click[-1], 6, (255, 0, 0), -1)
            lines.append((describe(H, cells, *click[-1], letters=not a.cells, labels=labels), (255, 128, 0)))
        draw_hud(view, lines)
        cv2.imshow("check", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            cv2.imwrite("check.png", view)
            print("saved check.png")
        elif k == ord("u"):
            locker.reset()  # read everything again from scratch (a different sheet, or re-poked dots)
            results = []
        elif k == ord("d"):
            show_reading = not show_reading  # keep the reading on screen even while the page is registered, or hide it
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
