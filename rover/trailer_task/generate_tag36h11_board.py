#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate an AprilTag tag36h11 grid board for camera calibration.")
    parser.add_argument("--output", default=str(here / "calibration" / "tag36h11_board_6x4_30mm.png"))
    parser.add_argument("--metadata", default="", help="Defaults to OUTPUT with .yaml suffix.")
    parser.add_argument("--cols", type=positive_int, default=6)
    parser.add_argument("--rows", type=positive_int, default=4)
    parser.add_argument("--tag-size-mm", type=positive_float, default=30.0)
    parser.add_argument("--gap-mm", type=positive_float, default=8.0)
    parser.add_argument("--margin-mm", type=positive_float, default=10.0)
    parser.add_argument("--dpi", type=positive_float, default=300.0)
    parser.add_argument("--first-id", type=int, default=0)
    return parser.parse_args()


def aruco_dictionary():
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return
    lines = []
    for key, value in data.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    metadata = Path(args.metadata).expanduser().resolve() if args.metadata else output.with_suffix(".yaml")
    output.parent.mkdir(parents=True, exist_ok=True)

    tag_m = args.tag_size_mm / 1000.0
    gap_m = args.gap_mm / 1000.0
    dictionary = aruco_dictionary()
    board = cv2.aruco.GridBoard_create(args.cols, args.rows, tag_m, gap_m, dictionary, firstMarker=args.first_id)

    px_per_mm = args.dpi / 25.4
    board_w_mm = args.cols * args.tag_size_mm + (args.cols - 1) * args.gap_mm
    board_h_mm = args.rows * args.tag_size_mm + (args.rows - 1) * args.gap_mm
    out_w = int(round((board_w_mm + 2 * args.margin_mm) * px_per_mm))
    out_h = int(round((board_h_mm + 2 * args.margin_mm) * px_per_mm))
    margin_px = int(round(args.margin_mm * px_per_mm))

    image = cv2.aruco.drawPlanarBoard(board, (out_w, out_h), marginSize=margin_px, borderBits=1)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(output), image)

    ids = np.asarray(board.ids).reshape(-1).astype(int).tolist()
    dump_yaml(
        metadata,
        {
            "dictionary": "DICT_APRILTAG_36h11",
            "cols": int(args.cols),
            "rows": int(args.rows),
            "tag_size_m": float(tag_m),
            "tag_size_mm": float(args.tag_size_mm),
            "gap_m": float(gap_m),
            "gap_mm": float(args.gap_mm),
            "margin_mm": float(args.margin_mm),
            "dpi": float(args.dpi),
            "first_id": int(args.first_id),
            "ids": ids,
            "image_path": str(output),
            "board_width_mm": float(board_w_mm),
            "board_height_mm": float(board_h_mm),
            "image_width_px": int(out_w),
            "image_height_px": int(out_h),
        },
    )
    print(f"Wrote board: {output}")
    print(f"Wrote metadata: {metadata}")
    print(f"Board physical size without margin: {board_w_mm:.1f} x {board_h_mm:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
