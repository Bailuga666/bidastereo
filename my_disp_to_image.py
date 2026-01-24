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
	parser = argparse.ArgumentParser(description="Convert disparity/depth npy to images or video")
	parser.add_argument("--disp", type=str, default="", help="Path to disparity npy")
	parser.add_argument("--depth", type=str, default="", help="Path to depth npy (optional, if provided, used directly)")
	parser.add_argument("--out-dir", type=str, required=True, help="Output directory for images or video path when --video is set")
	parser.add_argument("--video", action="store_true", help="If set, write a single video instead of images (out-dir treated as file path)")
	parser.add_argument("--scale", type=float, default=None, help="Depth scale (optional)")
	parser.add_argument("--eps", type=float, default=1e-6, help="EPS for disparity->depth")
	parser.add_argument("--lower-pct", type=float, default=1, help="Lower percentile for normalization")
	parser.add_argument("--upper-pct", type=float, default=99.0, help="Upper percentile for normalization")
	parser.add_argument(
		"--cmap",
		type=str,
		default="viridis",
		choices=["viridis", "viridis_r", "inferno", "magma", "plasma"],
		help="Colormap",
	)
	args = parser.parse_args()

	# load either depth or disparity
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

	# normalize
	depth_u8 = normalize_framewise(depth, args.lower_pct, args.upper_pct)

	cmap_map = {
		"viridis": cv2.COLORMAP_VIRIDIS,
		"viridis_r": cv2.COLORMAP_VIRIDIS,
		"inferno": cv2.COLORMAP_INFERNO,
		"magma": cv2.COLORMAP_MAGMA,
		"plasma": cv2.COLORMAP_PLASMA,
	}
	cmap = cmap_map[args.cmap]

	# output images or video
	if args.video:
		# args.out_dir is a file path for video
		out_file = args.out_dir
		frames, height, width = depth_u8.shape
		writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
		for i in range(frames):
			gray = depth_u8[i]
			if args.cmap == "viridis_r":
				gray = 255 - gray
			color = cv2.applyColorMap(gray, cmap)
			writer.write(color)
		writer.release()
		print(f"Saved video: {out_file} (frames={frames})")
	else:
		os.makedirs(args.out_dir, exist_ok=True)
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
# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_000pred_0.npy --out-dir ./outputs/disp_images/sample_000pred
# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_001pred_0.npy --out-dir ./outputs/disp_images/sample_001pred
# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_045pred_0.npy --out-dir ./outputs/disp_images/sample_045pred
# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_050pred_0.npy --out-dir ./outputs/disp_images/sample_050pred

# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_000input_0.npy --out-dir ./outputs/disp_images/sample_000input
# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_001input_0.npy --out-dir ./outputs/disp_images/sample_001input
# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_045input_0.npy --out-dir ./outputs/disp_images/sample_045input
# python my_disp_to_image.py --depth ./outputs/bidastereo_real/depth_sample_050input_0.npy --out-dir ./outputs/disp_images/sample_050input