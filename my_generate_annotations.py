import json
import gzip
import os
from typing import Optional, List
import numpy as np
from PIL import Image

# 用 dict 而不是 dataclass，避免 JSON 序列化问题

# 示例：生成你的数据注释
# 假设你的数据在 ./dynamic_replica_data/real/your_sequence/test/
# 左右相机，序列长度 100 帧
# 图像：left_000.png, right_000.png 等
# 深度：depth_000.png 等（如果有真实深度）
# mask：mask_000.png 等（如果有）

annotations = []

sequence_name = "sample_000input"  # 默认用这个名
root_dir = "./dynamic_replica_data/real/sample_000input/test"

# 是否生成全白 mask（评测必须有 mask 文件）
create_masks = True
# 是否生成虚拟深度（评测代码要求 depth.path 存在）
create_dummy_depths = True
# 虚拟深度值（单位任意，需为正）
dummy_depth_value = 1.0

# 相机参数（你需要填入实际值）
focal_length = [800.0, 800.0]  # 焦距
principal_point = [0.5, 0.5]   # 主点（NDC 归一化）

# 假设左右相机位姿（简化，实际需要校准）
R_left = [[1,0,0],[0,1,0],[0,0,1]]
T_left = [0, 0, 0]
R_right = [[1,0,0],[0,1,0],[0,0,1]]
T_right = [-0.1, 0, 0]  # 假设基线 0.1

# 裁剪尺寸（自动从图像读取）
first_img_path = os.path.join(root_dir, "images/left_000.png")
if os.path.exists(first_img_path):
    with Image.open(first_img_path) as img:
        crop_width, crop_height = img.size
        print(f"自动检测图像尺寸: {crop_width}x{crop_height}")
else:
    raise FileNotFoundError("找不到第一帧图像，无法确定尺寸")

# 自动检测帧数
images_dir = os.path.join(root_dir, "images")
if os.path.exists(images_dir):
    left_images = [f for f in os.listdir(images_dir) if f.startswith("left_") and f.endswith(".png")]
    num_frames = len(left_images)
    print(f"自动检测帧数: {num_frames}")
else:
    raise FileNotFoundError("找不到images目录")

for frame_num in range(num_frames):
    for cam in ["left", "right"]:
        img_path = f"images/{cam}_{frame_num:03d}.png"
        depth_path = f"depths/{cam}_{frame_num:03d}.png"
        mask_path = f"masks/{cam}_{frame_num:03d}.png"

        abs_img_path = os.path.join(root_dir, img_path)
        if not os.path.isfile(abs_img_path):
            raise FileNotFoundError(f"找不到图像: {abs_img_path}")

        # 读取真实图像尺寸，写入注释（现在固定为裁剪尺寸）
        image_size = [crop_width, crop_height]

        # 生成全白 mask（如果需要）
        if create_masks:
            abs_mask_path = os.path.join(root_dir, mask_path)
            os.makedirs(os.path.dirname(abs_mask_path), exist_ok=True)
            if not os.path.isfile(abs_mask_path):
                mask = Image.new("L", (crop_width, crop_height), 255)
                mask.save(abs_mask_path)

        # 生成虚拟深度（如果需要）
        if create_dummy_depths:
            abs_depth_path = os.path.join(root_dir, depth_path)
            os.makedirs(os.path.dirname(abs_depth_path), exist_ok=True)
            if not os.path.isfile(abs_depth_path):
                depth = np.full(
                    (crop_height, crop_width),
                    float(dummy_depth_value),
                    dtype=np.float16,
                )
                depth_u16 = depth.view(np.uint16)
                depth_img = Image.fromarray(depth_u16, mode="I;16")
                depth_img.save(abs_depth_path)

        if cam == "left":
            viewpoint = {
                "R": R_left,
                "T": T_left,
                "focal_length": focal_length,
                "principal_point": principal_point,
                "intrinsics_format": "ndc_norm_image_bounds"
            }
        else:
            viewpoint = {
                "R": R_right,
                "T": T_right,
                "focal_length": focal_length,
                "principal_point": principal_point,
                "intrinsics_format": "ndc_norm_image_bounds"
            }

        annot = {
            "sequence_name": sequence_name,
            "camera_name": cam,
            "frame_number": frame_num,
            "image": {
                "path": img_path,
                "size": image_size
            },
            "depth": {
                "path": depth_path
            } if depth_path else None,
            "mask": {
                "path": mask_path
            } if mask_path else None,
            "viewpoint": viewpoint
        }
        annotations.append(annot)

# 保存为 JSON，然后压缩成 .jgz
with gzip.open(f"{root_dir}/frame_annotations_test.jgz", "wt", encoding="utf8") as f:
    json.dump(annotations, f, indent=2)

print("注释文件生成完成")