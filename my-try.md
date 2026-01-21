# BiDAStereo 完整评估流程说明

以下是完整的BiDAStereo评估流程说明，从打开新终端到生成深度视频。假设你已经安装了WSL、Miniconda和相关依赖。

## 1. 进入WSL并激活虚拟环境
- 在Windows搜索栏输入“wsl”或“Ubuntu”，打开WSL终端。
- 进入项目目录：`cd /mnt/d/github_local/bidastereo`
- 激活Conda环境：`conda activate bidastereo`
- export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
- export CUDA_HOME=$CONDA_PREFIX
- 设置PYTHONPATH：`export PYTHONPATH=$(cd ../ && pwd):$(pwd):$PYTHONPATH`

## 2. 准备伪造数据（从单目视频生成立体图像序列）
- 清理旧数据：`rm -rf dynamic_replica_data/real/`
- 将你的单目视频文件重命名为`your_mono_video.mp4`，放置在`/mnt/d/github_local/bidastereo/`目录下。
- 编辑`extract_frames.py`：
  - 修改`video_path = "your_mono_video.mp4"`
  - 设置裁剪尺寸：`crop_width = 500`（宽度）、`crop_height = 400`（高度）
  - 设置视差像素：`disparity_pixels = 10`（右图相对左图的像素偏移）
  - 设置帧数：默认处理视频所有帧，或在循环中加`if frame_count >= 81: break`限制帧数。
- 运行生成图像序列：`python extract_frames.py`
  - 输出：生成`./dynamic_replica_data/real/nikita_reading/test/images/`下的`left_000.png`、`right_000.png`等文件，尺寸500x400。

## 3. 生成注释文件、mask和虚拟深度
- 运行生成注释：`python generate_annotations.py`
  - 自动检测图像尺寸（500x400）和帧数（从images目录）。
  - 输出：生成`./dynamic_replica_data/real/nikita_reading/test/frame_annotations_test.jgz`（注释文件），以及`depths/`和`masks/`目录下的虚拟深度和mask文件。

## 4. 验证数据形状正确
- 运行检查脚本：`python check_shapes.py`
  - 自动检测期望尺寸（500x400），检查前5帧的图像、深度、mask尺寸一致性。
  - 如果不一致，重新生成数据。

## 5. 运行评估测试代码
- 编辑`evaluate.py`第102行：`seq_len_real = 5`改为想要的帧数（例如10或81，匹配实际帧数）。
- 运行评估：`python ./evaluation/evaluate.py --config-name eval_real_data MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr.pth MODEL.BiDAStereoModel.kernel_size=10 visualize_interval=-1`
  - 输出：深度tensor形状[帧数,1,400,500]，保存到`./outputs/depth_nikita_reading_0.npy`。
  - 如果报PyTorch3D错误，确认CUDA可用。

## 6. 根据深度图序列生成深度视频
- 运行转换脚本：`python depth_to_video.py`
  - 自动根据`./outputs/depth_nikita_reading_0.npy`的形状生成视频，保存为`./outputs/depth_video.mp4`（尺寸500x400，帧率10fps）。
  - 用播放器打开查看深度变化视频。

整个流程完成后，你会得到深度预测视频。脚本已优化为自动检测参数，记得激活conda环境和设置PYTHONPATH。如果遇到问题，检查CUDA环境或重新安装依赖。