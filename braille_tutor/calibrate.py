"""Marker-free calibration for a FIXED camera and a FIXED page. Writes calibration.json.

Click the four corners of a rectangle whose real size you measured with a ruler, in this order: top left,
top right, bottom right, bottom left (e.g. an index card on the page, or four marks on the board).
Run: python calibrate.py --width 120 --height 60 [--camera N | --image photo.jpg] [--no-track]
For the A-Z sheet: python calibrate.py --sheet --no-track   (click the four corners of the paper)
Keys: space = freeze the live view, y = save, r = redo the clicks, q = quit.
The frozen frame is also saved as a reference photo, so later runs can follow the camera as it moves (tracker.py).
This needs a page with plenty of texture; with a blank or sparse page, keep the camera still or use the markers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from detect import open_camera
from page import homography_from_corners, save_homography, to_image


def draw_preview(view: np.ndarray, H: np.ndarray, w_mm: float, h_mm: float, origin: tuple = (0.0, 0.0),
                 step: int = 10) -> None:
    """Draw a millimeter grid over the calibrated rectangle so you can see if it matches the page."""
    x0, y0 = origin
    for x in np.arange(x0, x0 + w_mm + 0.1, step):
        cv2.polylines(view, [np.int32([to_image(H, x, y0), to_image(H, x, y0 + h_mm)])], False, (0, 200, 0), 1)
    for y in np.arange(y0, y0 + h_mm + 0.1, step):
        cv2.polylines(view, [np.int32([to_image(H, x0, y), to_image(H, x0 + w_mm, y)])], False, (0, 200, 0), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=float, help="mm between the top-left and top-right clicks")
    ap.add_argument("--height", type=float, help="mm between the top-left and bottom-left clicks")
    ap.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="page position (mm) of the top-left click; default 0 0")
    ap.add_argument("--sheet", action="store_true", help="the A-Z sheet: click the four corners of the A4 paper")
    ap.add_argument("--no-track", action="store_true", help="do not save a reference photo (fixed calibration only)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--image", help="calibrate on a photo instead of the camera")
    ap.add_argument("--out", default="calibration.json")
    a = ap.parse_args()
    if a.sheet:  # A4 paper; layout coordinates start at the top-left marker centre, which is ORIGIN mm in from the corner
        import make_sheet
        a.width, a.height, a.origin = make_sheet.A4_MM[0], make_sheet.A4_MM[1], (-make_sheet.ORIGIN[0], -make_sheet.ORIGIN[1])
    if a.width is None or a.height is None:
        ap.error("give --width and --height (mm), or --sheet")
    cap = None if a.image else open_camera(a.camera)
    frozen = cv2.imread(a.image) if a.image else None
    if a.image and frozen is None:
        raise SystemExit(f"cannot read {a.image}")
    pts: list = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and frozen is not None and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    live = None
    while True:
        if frozen is None:
            ok, live = cap.read()
            if not ok:
                break
            view, msg = live.copy(), "space: freeze the view, then click 4 corners"
        else:
            view, msg = frozen.copy(), "click: top left, top right, bottom right, bottom left"
        for i, p in enumerate(pts):
            cv2.circle(view, p, 6, (255, 0, 0), -1)
            cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        H = None
        if len(pts) == 4:
            H = homography_from_corners(pts, a.width, a.height, tuple(a.origin))
            draw_preview(view, H, a.width, a.height, tuple(a.origin))
            msg = "y: save   r: redo"
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(20) & 0xFF
        if k == ord(" ") and frozen is None:
            frozen = live
        elif k == ord("r"):
            pts.clear()
        elif k == ord("y") and H is not None:
            save_homography(H, a.out)
            ref = Path(a.out).with_suffix(".png")
            ref.unlink(missing_ok=True)  # a stale reference from an earlier run would turn tracking on
            if a.no_track:
                print(f"saved {a.out} (fixed calibration: the camera and page must not move).")
            else:
                cv2.imwrite(str(ref), frozen)  # reference photo: lets tracker.py follow the camera
                print(f"saved {a.out} and {ref} (page tracking on; use --no-track for a fixed calibration).")
            print(f"Use it with: python detect.py --live --calib {a.out}")
            break
        elif k == ord("q"):
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
