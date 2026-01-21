# 批量处理真实双目（伪造）数据并评估指南 ✅

下面说明如何把多组真实/单目视频或左右目图像批量转为 BiDAStereo 可评估的**真实数据格式**，并运行评估与生成深度视频。

## 目录结构 & 命名规范 🔧
- 每个序列放在：`dynamic_replica_data/real/<sequence_name>/test/` 下。
- 图像放在：`.../test/images/`，命名规则：
  - 左目：`left_000.png`, `left_001.png`, ...
  - 右目：`right_000.png`, `right_001.png`, ...
- 额外文件（评估需要）：
  - `masks/`：`mask` png（评估加载时会检查存在）
  - `depths/`：占位深度（可选，评估不要求真实深度）
  - `frame_annotations_test.jgz`：序列的 frame 注释（必须存在）

> 注意：左右图像数量与尺寸必须一致（例如 256×256）。

---

## 快速一键生成伪双目序列（推荐） 💡
我们添加了一个脚本 `scripts/create_real_sequences.py`，可以从单目视频批量生成左右图像、masks、占位 depth 与注释文件。

示例用法：

```bash
# 生成两个序列：mydata1, mydata2
python scripts/create_real_sequences.py \
  --pair /path/to/mono1.mp4:mydata1 \
  --pair /path/to/mono2.mp4:mydata2 \
  --out-root ./dynamic_replica_data/real \
  --crop 256 --shift 10
```

脚本默认会：
- 中心裁剪 `crop x crop`（默认 256）
- 右目相对于左目向右平移 `shift` 像素（默认 10）
- 生成 `images/`, `masks/`, `depths/` 和 `frame_annotations_test.jgz`

---

## 手动流程（单序列）📋
如果你更喜欢手动控制，参考以下步骤：

1. 把图像放到 `dynamic_replica_data/real/<seq>/test/images/`，左右命名为 `left_XXX.png` / `right_XXX.png`。
2. 生成全白 mask（若没有）并放到 `.../test/masks/`。
3. （可选）生成占位 depth 到 `.../test/depths/`（脚本会自动生成 float16 的占位深度）。
4. 生成注释文件 `frame_annotations_test.jgz`，包含每一帧的 `image`/`depth`/`mask` 路径与 viewpoint 信息（脚本会自动生成默认 viewpoint）。

我们提供的脚本会替你完成 2-4 步。

---

## 评估运行（生成深度 npy） ▶️
进入项目根目录并确保已激活 conda 环境与设置 PYTHONPATH：

```bash
conda activate bidastereo
export PYTHONPATH=`(cd ../ && pwd):$(pwd):$PYTHONPATH`
```

运行评估（示例）：

```bash
python ./evaluation/evaluate.py --config-name eval_real_data MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr.pth MODEL.BiDAStereoModel.kernel_size=10 visualize_interval=-1
```

说明：`evaluate.py` 的 `real` 分支会自动使用 `dynamic_replica_data/real/<sequence>/test/images/left_*.png` 的帧数作为序列长度（无需手动设置）。

评估流程会把深度保存为： `./outputs/bidastereo_real/depth_<sequence>_<batch>.npy`。

---

## 将 npy 转为深度视频（可视化） 🎞️
使用提供的工具：

```bash
# 将深度 npy 转为视频（inferno colormap）
python my_disp_to_depth_video.py --depth ./outputs/bidastereo_real/depth_mydata1_0.npy --out ./outputs/depth_mydata1.mp4
```

也可用 `--disp` 来输入视差 npy，并设置 `--scale`（若你知道转换比例）。

---

## 批量处理建议与注意事项 ✅
- 推荐先用脚本生成并检查每个序列下 `images/`、`masks/`、`frame_annotations_test.jgz` 是否正确。
- 若 GPU 内存不足，请在评估命令中把 `MODEL.BiDAStereoModel.kernel_size` 设置小一点（例如 10）。
- 若需要并行批量评估多个序列，建议一次评估一个序列（改 `evaluate.py` 中 `for real_sequence_name in [...]`，或循环修改并运行）。

---

如果你愿意，我可以：
- 立刻帮你用 `scripts/create_real_sequences.py` 生成若干序列，或
- 帮你写一个小的 bash 循环来对多个序列自动跑评估并导出视频。

需要我把这份文档保存到仓库并把脚本也提交上去吗？（我已经在仓库创建了对应脚本，告诉我是否要运行示例）
