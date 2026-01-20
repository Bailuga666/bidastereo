import torch
import pytorch3d
from pytorch3d.structures import Pointclouds

print("=== PyTorch3D 综合测试 ===")
print(f"PyTorch3D版本: {pytorch3d.__version__}")
print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
print(f"PyTorch CUDA版本: {torch.version.cuda}")

# 测试基本导入
try:
    from pytorch3d.renderer.points import rasterize_points
    print("✓ rasterize_points导入成功")
except ImportError as e:
    print(f"✗ rasterize_points导入失败: {e}")

# 测试Pointclouds
try:
    points = torch.randn(1, 2, 3)
    if torch.cuda.is_available():
        points = points.cuda()
    point_cloud = Pointclouds(points=points)
    print("✓ Pointclouds创建成功")
except Exception as e:
    print(f"✗ Pointclouds创建失败: {e}")

# 测试简单tensor操作
try:
    t = torch.randn(2, 3)
    if torch.cuda.is_available():
        t = t.cuda()
    print(f"✓ Tensor GPU操作成功: {t.device}")
except Exception as e:
    print(f"✗ Tensor GPU操作失败: {e}")

# 测试rasterize_points（可能失败）
try:
    points = torch.randn(1, 2, 3)
    if torch.cuda.is_available():
        points = points.cuda()
    point_cloud = Pointclouds(points=points)
    radii = torch.ones(1, 2)
    if torch.cuda.is_available():
        radii = radii.cuda()
    result = rasterize_points(point_cloud, radii, (10, 10))
    print("✓ rasterize_points调用成功")
except Exception as e:
    print(f"✗ rasterize_points调用失败: {e}")

print("测试完成。如果rasterize_points失败，可视化会出错，需禁用。")