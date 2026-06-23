"""
Label an extxyz with the configured teacher.

Used for (a) rattle-augmented geometries and (b) AL pre-crash frames from the
local student LAMMPS MD. The teacher (and, for the UMA example, its model+task)
come from the single ``teacher:`` config block, so labels live on the SAME
potential-energy surface as the teacher MD.

If ``system.force_pbc`` is true, each frame's PBC is forced to [T, T, T] before
labeling (LAMMPS 'p p f' dumps come in mixed-PBC and many calculators reject
mixed PBC). This preserves the behavior of the original oc25 labeler.

Usage: python scripts/label.py <in.xyz> <out.extxyz>
Writes <out.extxyz> with info['energy'] + arrays['forces'], and a DONE marker.
"""
import os, sys, time
from ase.io import read, write

from ontheflydistill import config as _cfg
from ontheflydistill.teachers import get_teacher


def main():
    inp = sys.argv[1]
    out = sys.argv[2]
    done = out + ".DONE"
    if os.path.exists(done):
        os.remove(done)
    if not os.path.exists(inp) or os.path.getsize(inp) == 0:
        write(out, [], format="extxyz")
        open(done, "w").write("OK 0\n")
        print(f"input {inp} missing/empty; wrote empty {out}", flush=True)
        return
    frames = read(inp, index=":")
    print(f"loaded {len(frames)} frames from {inp}", flush=True)
    if len(frames) == 0:
        write(out, [], format="extxyz")
        open(done, "w").write("OK 0\n")
        return

    teacher = get_teacher(_cfg.load())
    if not getattr(teacher, "can_relabel", False):
        sys.exit(
            "Configured teacher cannot relabel frames (extxyz/no calculator). "
            "Use teacher.type ase_calculator or uma for labeling."
        )

    t0 = time.time()
    labeled = teacher.label(frames)
    write(out, labeled, format="extxyz")
    open(done, "w").write(f"OK {len(labeled)} in {time.time()-t0:.0f}s\n")
    print(f"wrote {out} ({len(labeled)} frames) {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
