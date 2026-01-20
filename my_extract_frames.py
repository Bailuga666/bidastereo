import cv2
import os
import numpy as np

# 输入单目视频路径
video_path = "your_mono_video.mp4"  # 改成你的单目视频路径

# 输出目录
output_dir = "./dynamic_replica_data/real/nikita_reading/test/images"
os.makedirs(output_dir, exist_ok=True)

# 模拟视差：右图相对于左图向右移动的像素数
disparity_pixels = 10  # 可以改

# 打开视频
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频")
    exit()

# 获取视频尺寸
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频尺寸: {width}x{height}")

# 裁剪尺寸
crop_width = 700
crop_height = 600

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 裁剪左上角 crop_height x crop_width
    frame = frame[:crop_height, :crop_width]

    # 左图：原帧
    left_frame = frame

    # 右图：向右移动 disparity_pixels 像素，右边补黑
    right_frame = np.zeros_like(frame)
    right_frame[:, disparity_pixels:] = frame[:, :-disparity_pixels]

    # 保存左右图
    left_path = os.path.join(output_dir, f"left_{frame_count:03d}.png")
    cv2.imwrite(left_path, left_frame)

    right_path = os.path.join(output_dir, f"right_{frame_count:03d}.png")
    cv2.imwrite(right_path, right_frame)

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"处理了 {frame_count} 帧")

cap.release()
print(f"总共生成 {frame_count} 帧左右图像对")

# 记得修改 generate_annotations.py 里的 image_size 为 [width, height]