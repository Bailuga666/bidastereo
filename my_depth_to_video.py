import numpy as np
import cv2
import os

# 加载深度数据
depth_file = './outputs/bidastereo_real/depth_nikita_reading_0.npy'
if not os.path.exists(depth_file):
    print("深度文件不存在")
    exit()

depth = np.load(depth_file)  # 假设形状 (frames, 1, height, width)
print(f"加载深度数据，形状: {depth.shape}")
frames, channels, height, width = depth.shape
depth = depth.squeeze(1)  # (frames, height, width)

# 归一化到0-255
depth_min = depth.min()
depth_max = depth.max()
if depth_max > depth_min:
    depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
else:
    depth_norm = np.zeros_like(depth, dtype=np.uint8)

# 保存为视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./outputs/depth_video.mp4', fourcc, 10, (width, height))

for i in range(frames):
    frame = depth_norm[i]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # 灰度转BGR
    out.write(frame_bgr)

out.release()
print(f"深度视频保存到 ./outputs/depth_video.mp4，尺寸: {width}x{height}，帧数: {frames}")