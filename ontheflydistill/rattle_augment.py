"""
STEP 2 (LOCAL, any env with ASE): rattle augmentation.

Take the saved teacher-MD frames and produce rattled copies (geometry only).
The rattled configs broaden near-equilibrium coverage; they are LABELED by UMA-s
afterwards (relabel_with_uma.py) and merged into the training set.

We rattle ALL atoms (it is fine for the frozen-Pt atoms to move slightly in the
labeled training data; the FixAtoms constraint only matters during MD dynamics,
not for static labeling). FixAtoms info is stripped so ASE write/read is clean.

To keep cost bounded we sub-sample the teacher frames before rattling.

Usage:
  python rattle_augment.py <teacher_md.extxyz> <out_rattled.xyz>
     [--per-frame N] [--stride S] [--stdevs 0.05,0.1] [--seed 42]
"""
import sys, argparse
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher_xyz")
    ap.add_argument("out_xyz")
    ap.add_argument("--per-frame", type=int, default=2,
                    help="rattled copies per (sub-sampled) teacher frame, per stdev")
    ap.add_argument("--stride", type=int, default=3,
                    help="sub-sample teacher frames: keep every Nth")
    ap.add_argument("--stdevs", default="0.05,0.1",
                    help="comma-sep rattle stdevs in Angstrom")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = read(args.teacher_xyz, index="::%d" % args.stride)
    stdevs = [float(x) for x in args.stdevs.split(",")]
    print(f"loaded {len(frames)} sub-sampled teacher frames "
          f"(stride={args.stride}); stdevs={stdevs}, per_frame={args.per_frame}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    out = []
    for fi, base in enumerate(frames):
        # identify fixed (bottom-Pt) indices to keep UNrattled (spec requirement)
        fixed_idx = []
        for c in base.constraints:
            if isinstance(c, FixAtoms):
                fixed_idx = list(c.get_indices())
                break
        for sd in stdevs:
            for k in range(args.per_frame):
                a = base.copy()
                # strip calc results / constraints for a clean geometry-only config
                a.calc = None
                a.set_constraint([])
                if "energy" in a.info:
                    del a.info["energy"]
                if "forces" in a.arrays:
                    del a.arrays["forces"]
                pos0 = a.get_positions().copy()
                seed = int(rng.integers(0, 2**31 - 1))
                a.rattle(stdev=sd, seed=seed)
                # restore fixed-atom positions so bottom-Pt stays put
                if fixed_idx:
                    p = a.get_positions()
                    p[fixed_idx] = pos0[fixed_idx]
                    a.set_positions(p)
                out.append(a)
    write(args.out_xyz, out, format="extxyz")
    print(f"wrote {len(out)} rattled configs -> {args.out_xyz}", flush=True)


if __name__ == "__main__":
    main()
