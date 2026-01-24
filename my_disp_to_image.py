import argparse
import os
from typing import Optional

import cv2
import numpy as np


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


def normalize_framewise(data: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
	t, h, w = data.shape
	out = np.zeros((t, h, w), dtype=np.uint8)
	for i in range(t):
		frame = data[i]
		finite = np.isfinite(frame)
		if not np.any(finite):
			continue
		vmin = np.percentile(frame[finite], lower_pct)
		vmax = np.percentile(frame[finite], upper_pct)
		if vmax <= vmin:
			continue
		clipped = np.clip(frame, vmin, vmax)
		norm = (clipped - vmin) / (vmax - vmin)
		out[i] = (np.clip(norm * 255, 0, 255)).astype(np.uint8)
	return out


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert disparity npy to images")
	parser.add_argument("--disp", type=str, required=True, help="Path to disparity npy")
	parser.add_argument("--out-dir", type=str, required=True, help="Output directory for images")
	parser.add_argument("--scale", type=float, default=None, help="Depth scale (optional)")
	parser.add_argument("--eps", type=float, default=1e-6, help="EPS for disparity->depth")
	parser.add_argument("--lower-pct", type=float, default=1.0, help="Lower percentile for normalization")
	parser.add_argument("--upper-pct", type=float, default=99.0, help="Upper percentile for normalization")
	parser.add_argument(
		"--cmap",
		type=str,
		default="viridis_r",
		choices=["viridis", "viridis_r", "inferno", "magma", "plasma"],
		help="Colormap",
	)
	args = parser.parse_args()

	disp = load_sequence_npy(args.disp)
	depth = disparity_to_depth(disp, args.scale, eps=args.eps)

	depth_u8 = normalize_framewise(depth, args.lower_pct, args.upper_pct)

	os.makedirs(args.out_dir, exist_ok=True)

	cmap_map = {
		"viridis": cv2.COLORMAP_VIRIDIS,
		"viridis_r": cv2.COLORMAP_VIRIDIS,
		"inferno": cv2.COLORMAP_INFERNO,
		"magma": cv2.COLORMAP_MAGMA,
		"plasma": cv2.COLORMAP_PLASMA,
	}
	cmap = cmap_map[args.cmap]

	for i in range(depth_u8.shape[0]):
		gray = depth_u8[i]
		if args.cmap == "viridis_r":
			gray = 255 - gray
		color = cv2.applyColorMap(gray, cmap)
		out_path = os.path.join(args.out_dir, f"disp_{i:03d}.png")
		cv2.imwrite(out_path, color)
		print(f"frame {i:03d} -> {out_path}")


if __name__ == "__main__":
	main()
# python my_disp_to_image.py --disp ./outputs/bidastereo_real/depth_sample_045input_0.npy --out-dir /openbayes/home/bidastereo/outputs/disp_images/depth_sample_045input_0
# python my_disp_to_image.py --disp ./outputs/bidastereo_real/depth_sample_045pred_0.npy --out-dir /openbayes/home/bidastereo/outputs/disp_images/depth_sample_045pred_0
# python my_disp_to_image.py --disp ./outputs/bidastereo_real/depth_sample_050input_0.npy --out-dir /openbayes/home/bidastereo/outputs/disp_images/depth_sample_050input_0
# python my_disp_to_image.py --disp ./outputs/bidastereo_real/depth_sample_050pred_0.npy --out-dir /openbayes/home/bidastereo/outputs/disp_images/depth_sample_050pred_0