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


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros_like(data, dtype=np.uint8)
    vmin = data[finite].min()
    vmax = data[finite].max()
    if vmax <= vmin:
        return np.zeros_like(data, dtype=np.uint8)
    norm = (data - vmin) / (vmax - vmin)
    return (norm * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert disparity/depth npy to depth video")
    parser.add_argument("--disp", type=str, default="", help="Path to disparity npy")
    parser.add_argument("--depth", type=str, default="", help="Path to depth npy")
    parser.add_argument("--scale", type=float, default=None, help="Depth scale (optional)")
    parser.add_argument("--fps", type=int, default=10, help="Output video fps")
    parser.add_argument(
        "--out",
        type=str,
        default="./outputs/depth_from_disp.mp4",
        help="Output video path",
    )
    args = parser.parse_args()

    if args.depth:
        if not os.path.exists(args.depth):
            raise FileNotFoundError(args.depth)
        depth = load_sequence_npy(args.depth)
    elif args.disp:
        if not os.path.exists(args.disp):
            raise FileNotFoundError(args.disp)
        disp = load_sequence_npy(args.disp)
        depth = disparity_to_depth(disp, args.scale, eps=1e-6)
    else:
        raise ValueError("Please provide --disp or --depth")

    depth_vis = normalize_to_uint8(depth)
    frames, height, width = depth_vis.shape

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )

    for i in range(frames):
        frame = depth_vis[i]
        frame_bgr = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
        writer.write(frame_bgr)

    writer.release()
    print(f"保存深度视频: {args.out} (frames={frames}, size={width}x{height})")


if __name__ == "__main__":
    main()
