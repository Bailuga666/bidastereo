"""
脚本：从单目视频批量创建伪双目序列（images, masks, depths, annotations）
用法示例：
python scripts/create_real_sequences.py \
  --pair /path/to/mono1.mp4:mydata1 \
  --pair /path/to/mono2.mp4:mydata2 \
  --out-root ./dynamic_replica_data/real \
  --crop 256 --shift 10
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def save_depth_as_16png(depth_arr: np.ndarray, path: Path):
    # depth_arr: float32 or float16 2D
    depth_f16 = depth_arr.astype(np.float16)
    depth_u16 = depth_f16.view(np.uint16)
    depth_img = Image.fromarray(depth_u16, mode="I;16")
    depth_img.save(str(path))


def create_sequence_from_video(video_path, seq_name, out_root, crop, shift, dummy_depth=1.0):
    out_root = Path(out_root)
    images_dir = out_root / seq_name / "test" / "images"
    masks_dir = out_root / seq_name / "test" / "masks"
    depths_dir = out_root / seq_name / "test" / "depths"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width < crop or height < crop:
        raise ValueError(f"Video too small {width}x{height} for crop {crop}x{crop}")

    x0 = (width - crop) // 2
    y0 = (height - crop) // 2

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cropped = frame[y0 : y0 + crop, x0 : x0 + crop]
        left = frame_cropped
        right = np.zeros_like(frame_cropped)
        if shift >= crop:
            # 如果 shift 太大，右图全部黑
            pass
        else:
            right[:, shift:] = frame_cropped[:, :-shift]

        left_path = images_dir / f"left_{frame_count:03d}.png"
        right_path = images_dir / f"right_{frame_count:03d}.png"
        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)

        # masks: 全白 L
        mask = Image.new("L", (crop, crop), 255)
        mask.save(str(masks_dir / f"left_{frame_count:03d}.png"))
        mask.save(str(masks_dir / f"right_{frame_count:03d}.png"))

        # depths: 恒定 dummy_depth 保存为 I;16
        depth = np.full((crop, crop), float(dummy_depth), dtype=np.float32)
        save_depth_as_16png(depth, depths_dir / f"left_{frame_count:03d}.png")
        save_depth_as_16png(depth, depths_dir / f"right_{frame_count:03d}.png")

        frame_count += 1

    cap.release()

    # 生成 frame_annotations_test.jgz
    annotations = []
    focal_length = [800.0, 800.0]
    principal_point = [0.5, 0.5]
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_left = [0, 0, 0]
    T_right = [-0.1, 0, 0]

    for i in range(frame_count):
        for cam in ["left", "right"]:
            img_path = f"images/{cam}_{i:03d}.png"
            depth_path = f"depths/{cam}_{i:03d}.png"
            mask_path = f"masks/{cam}_{i:03d}.png"

            annot = {
                "sequence_name": seq_name,
                "camera_name": cam,
                "frame_number": i,
                "image": {"path": img_path, "size": [crop, crop]},
                "depth": {"path": depth_path},
                "mask": {"path": mask_path},
                "viewpoint": {
                    "R": R,
                    "T": T_left if cam == "left" else T_right,
                    "focal_length": focal_length,
                    "principal_point": principal_point,
                    "intrinsics_format": "ndc_norm_image_bounds",
                },
            }
            annotations.append(annot)

    gz_path = out_root / seq_name / "test" / "frame_annotations_test.jgz"
    with gzip.open(str(gz_path), "wt", encoding="utf8") as f:
        json.dump(annotations, f, indent=2)

    print(f"Created sequence {seq_name} with {frame_count} frames at {out_root/seq_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        help="video_path:sequence_name (多次传入支持多个)",
        required=True,
    )
    parser.add_argument("--out-root", default="./dynamic_replica_data/real")
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--shift", type=int, default=10, help="右目相对左目右移像素数")
    parser.add_argument("--dummy-depth", type=float, default=1.0)

    args = parser.parse_args()

    for p in args.pair:
        if ":" not in p:
            print(f"非法 --pair 参数: {p}, 必须形如 /path/to/vid.mp4:seqname")
            continue
        video_path, seq_name = p.split(":", 1)
        create_sequence_from_video(video_path, seq_name, args.out_root, args.crop, args.shift, args.dummy_depth)


if __name__ == "__main__":
    main()
