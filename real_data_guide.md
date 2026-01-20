# 处理真实双目相机图像的BiDAStereo评估流程

如果你有一组真实的双目相机拍摄的照片（没有depth和mask），可以按照以下步骤处理并生成深度视频。假设图像已校准，左右相机对应。

## 1. 准备环境
- 进入WSL，激活环境：`conda activate bidastereo`
- 设置PYTHONPATH：`export PYTHONPATH=$(cd ../ && pwd):$(pwd):$PYTHONPATH`
- 进入目录：`cd /mnt/d/github_local/bidastereo`

## 2. 放置图像文件
- 创建目录：`mkdir -p dynamic_replica_data/real/your_sequence/test/images`
- 将左相机图像重命名为`left_000.png`、`left_001.png`等，放在`images/`目录。
- 将右相机图像重命名为`right_000.png`、`right_001.png`等，放在同一目录。
- 确保左右图像数量相同，尺寸一致（例如500x400）。

## 3. 生成注释文件、mask和虚拟深度
- 编辑`generate_annotations.py`：
  - 修改`sequence_name = "your_sequence"`（替换nikita_reading）
  - 修改`root_dir = "./dynamic_replica_data/real/your_sequence/test"`
- 运行：`python generate_annotations.py`
  - 自动检测图像尺寸和帧数，生成虚拟depth（全1.0）和mask（全白）。

## 4. 验证数据形状
- 运行：`python check_shapes.py`
  - 检查图像、depth、mask尺寸一致。

## 5. 运行评估
- 编辑`evaluate.py`第99行：`for real_sequence_name in ["your_sequence"]:`（替换nikita_reading）
- 第102行：`seq_len_real = 帧数`（设置为实际帧数）
- 运行：`python ./evaluation/evaluate.py --config-name eval_real_data MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr.pth MODEL.BiDAStereoModel.kernel_size=10 visualize_interval=-1`
  - 生成深度预测，保存为`./outputs/depth_your_sequence_0.npy`

## 6. 生成深度视频
- 运行：`python depth_to_video.py`
  - 自动生成`./outputs/depth_video.mp4`

如果有真实depth图像，替换`depths/`目录的文件；如果有mask，替换`masks/`。相机参数在`generate_annotations.py`中调整。