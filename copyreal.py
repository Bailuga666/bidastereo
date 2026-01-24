import os, glob, shutil

src_left = "/openbayes/home/test/left/group_0000"
src_right = "/openbayes/home/test/right/group_0000"
dst = "dynamic_replica_data/real/my_data/test/images"
os.makedirs(dst, exist_ok=True)

lefts = sorted(glob.glob(os.path.join(src_left, "*.png")))
rights = sorted(glob.glob(os.path.join(src_right, "*.png")))

assert len(lefts) == len(rights), "左右图像数量不一致"

for i, (l, r) in enumerate(zip(lefts, rights)):
    shutil.copy(l, os.path.join(dst, f"left_{i:03d}.png"))
    shutil.copy(r, os.path.join(dst, f"right_{i:03d}.png"))

print("done, frames =", len(lefts))