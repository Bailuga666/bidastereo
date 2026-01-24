import os
import re
import shutil
from glob import glob
# /openbayes/home/sample_045
SRC_DIR = "/openbayes/home/sample_045"
DST_INPUTS = os.path.join(SRC_DIR, "inputs")
DST_PREDS = os.path.join(SRC_DIR, "preds")


def collect_frames(suffix: str):
    pattern = os.path.join(SRC_DIR, f"frame_*_{suffix}.png")
    files = glob(pattern)
    frames = {}
    for f in files:
        m = re.search(r"frame_(\d+)_", os.path.basename(f))
        if not m:
            continue
        frame_id = int(m.group(1))
        frames[frame_id] = f
    return frames


def main():
    os.makedirs(DST_INPUTS, exist_ok=True)
    os.makedirs(DST_PREDS, exist_ok=True)

    left_input = collect_frames("left_input")
    right_input = collect_frames("right_input")
    left_pred = collect_frames("left_pred")
    right_pred = collect_frames("right_pred")

    frame_ids = sorted(set(left_input) & set(right_input) & set(left_pred) & set(right_pred))
    if not frame_ids:
        print("没有找到完整的左右 input/pred 成对帧")
        return

    for idx, frame_id in enumerate(frame_ids):
        shutil.copy(left_input[frame_id], os.path.join(DST_INPUTS, f"left_{idx:03d}.png"))
        shutil.copy(right_input[frame_id], os.path.join(DST_INPUTS, f"right_{idx:03d}.png"))
        shutil.copy(left_pred[frame_id], os.path.join(DST_PREDS, f"left_{idx:03d}.png"))
        shutil.copy(right_pred[frame_id], os.path.join(DST_PREDS, f"right_{idx:03d}.png"))

    print(f"done, frames = {len(frame_ids)}")


if __name__ == "__main__":
    main()
