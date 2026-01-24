import argparse
import os
from typing import Optional

import numpy as np
import cv2


def load_sequence_npy(path: str) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
    elif data.ndim == 2:
        data = data[None, ...]
    return data.astype(np.float32)


def disparity_to_depth(disp: np.ndarray, scale: Optional[float], eps: float) -> np.ndarray:
    disp = np.abs(disp)
    if scale is None:
        depth = 1.0 / (disp + eps)
    else:
        depth = scale / (disp + eps)
    return depth


def normalize_to_uint8_framewise(data: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
    # data: (T, H, W)
    T, H, W = data.shape
    out = np.zeros_like(data, dtype=np.uint8)
    for i in range(T):
        frame = data[i]
        finite = np.isfinite(frame) & (frame > 0)
        if not np.any(finite):
            out[i] = np.zeros((H, W), dtype=np.uint8)
            continue
        vmin = np.percentile(frame[finite], lower_pct)
        vmax = np.percentile(frame[finite], upper_pct)
        if vmax <= vmin:
            out[i] = np.zeros((H, W), dtype=np.uint8)
            continue
        clipped = np.clip(frame, vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin)
        out[i] = (np.clip(norm * 255, 0, 255)).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert disparity/depth npy to depth video with 95pct upper clipping")
    parser.add_argument("--disp", type=str, default="", help="Path to disparity npy")
    parser.add_argument("--depth", type=str, default="", help="Path to depth npy")
    parser.add_argument("--scale", type=float, default=None, help="Depth scale (optional)")
    parser.add_argument("--fps", type=int, default=10, help="Output video fps")
    parser.add_argument(
        "--out",
        type=str,
        default="./outputs/depth_from_disp_clip95.mp4",
        help="Output video path",
    )
    parser.add_argument("--lower-pct", type=float, default=5.0, help="Lower percentile for clipping (default 5)")
    parser.add_argument("--upper-pct", type=float, default=95.0, help="Upper percentile for clipping (default 95)")
    parser.add_argument("--eps", type=float, default=1e-6, help="EPS for disparity->depth")
    args = parser.parse_args()

    if args.depth:
        if not os.path.exists(args.depth):
            raise FileNotFoundError(args.depth)
        depth = load_sequence_npy(args.depth)
    elif args.disp:
        if not os.path.exists(args.disp):
            raise FileNotFoundError(args.disp)
        disp = load_sequence_npy(args.disp)
        depth = disparity_to_depth(disp, args.scale, eps=args.eps)
    else:
        raise ValueError("Please provide --disp or --depth")

    # depth: (T,H,W)
    frames, height, width = depth.shape

    # normalize per-frame with percentile clipping
    depth_u8_seq = normalize_to_uint8_framewise(depth, args.lower_pct, args.upper_pct)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )

    for i in range(frames):
        frame = depth_u8_seq[i]
        frame_bgr = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
        writer.write(frame_bgr)

    writer.release()
    print(
        f"保存深度视频: {args.out} (frames={frames}, size={width}x{height}, upper_pct={args.upper_pct})"
    )


if __name__ == "__main__":
    main()
