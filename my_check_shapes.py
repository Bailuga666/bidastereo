import os
from PIL import Image
import numpy as np

# 数据目录
root_dir = "./dynamic_replica_data/real/nikita_reading/test"

# 检查前几个文件
num_checks = 5

print("检查图像、深度、mask尺寸一致性：")

expected_size = None
all_consistent = True

for frame_num in range(num_checks):
    print(f"\n帧 {frame_num:03d}:")

    # 图像
    img_path = os.path.join(root_dir, f"images/left_{frame_num:03d}.png")
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            img_size = img.size  # (width, height)
            print(f"  图像尺寸: {img_size} (width x height)")
            if expected_size is None:
                expected_size = img_size
    else:
        print("  图像文件不存在")
        all_consistent = False
        continue

    # 深度
    depth_path = os.path.join(root_dir, f"depths/left_{frame_num:03d}.png")
    if os.path.exists(depth_path):
        with Image.open(depth_path) as depth_img:
            depth_array = np.array(depth_img)
            depth_shape = depth_array.shape  # (height, width)
            print(f"  深度尺寸: {depth_shape} (height x width), dtype: {depth_array.dtype}")
            if (depth_shape[1], depth_shape[0]) != expected_size:
                all_consistent = False
    else:
        print("  深度文件不存在")
        all_consistent = False

    # mask
    mask_path = os.path.join(root_dir, f"masks/left_{frame_num:03d}.png")
    if os.path.exists(mask_path):
        with Image.open(mask_path) as mask_img:
            mask_array = np.array(mask_img)
            mask_shape = mask_array.shape  # (height, width)
            print(f"  mask尺寸: {mask_shape} (height x width), dtype: {mask_array.dtype}")
            if (mask_shape[1], mask_shape[0]) != expected_size:
                all_consistent = False
    else:
        print("  mask文件不存在")
        all_consistent = False

print(f"\n检查完成。期望尺寸: {expected_size} (width x height)，一致性: {'是' if all_consistent else '否'}")