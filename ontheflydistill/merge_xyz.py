"""Merge multiple extxyz files (with E/F) into one. Skips missing/empty inputs.
Usage: python merge_xyz.py <out.extxyz> <in1.extxyz> [in2 ...]"""
import sys, os
from ase.io import read, write

out = sys.argv[1]
allframes = []
for p in sys.argv[2:]:
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print(f"  skip (missing/empty): {p}")
        continue
    try:
        fr = read(p, index=":")
    except Exception as e:
        print(f"  skip (read error {e}): {p}")
        continue
    allframes.extend(fr)
    print(f"  +{len(fr)} from {p}")
write(out, allframes, format="extxyz")
print(f"merged {len(allframes)} frames -> {out}")
