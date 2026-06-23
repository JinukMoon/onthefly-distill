"""
STEP 1 (LOCAL): teacher MD.

Run a teacher MD trajectory (Langevin NVT) from the init structure, producing an
extxyz with energy + forces on every saved frame, from which the NN-MTP student
is distilled. The teacher is whatever ``get_teacher(config)`` returns (any ASE
calculator; the UMA example by default). The teacher's calculator and (for UMA)
the model+task come from the single ``teacher:`` config block, so the teacher
trajectory and the later labels live on the SAME potential-energy surface.

The bottom-slab FixAtoms constraint (``system.fixed_bottom_n`` atoms) is kept
throughout MD when the init structure does not already carry one.

Usage:
  python scripts/teacher_md.py <out.extxyz> [n_steps] [save_every]
"""
import os, sys, time
import numpy as np
from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.constraints import FixAtoms

from ontheflydistill import config as _cfg
from ontheflydistill.teachers import get_teacher

CFG = _cfg.load()
STRUCT = os.path.abspath(_cfg.get("system", "init_structure",
                                  default="examples/ptwater_acid/ptwater_acid_init_clean.vasp",
                                  cfg=CFG))
FIXED_BOTTOM_N = int(_cfg.get("system", "fixed_bottom_n", default=32, cfg=CFG))
SPECORDER = list(_cfg.get("system", "specorder", default=["H", "O", "Pt"], cfg=CFG))

OUT_XYZ = sys.argv[1] if len(sys.argv) > 1 else "teacher_md.extxyz"
DONE = OUT_XYZ + ".DONE"
LOGFILE = OUT_XYZ + ".aselog"

TEMP = float(os.environ.get("TEACHER_TEMP", "300.0"))     # K
DT_FS = float(os.environ.get("TEACHER_DT_FS", "1.0"))     # fs
FRICTION = float(os.environ.get("TEACHER_FRICTION", "0.02"))  # 1/fs (Langevin)
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("TEACHER_STEPS", "3000"))
SAVE_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("TEACHER_SAVE_EVERY", "20"))
SEED = int(os.environ.get("TEACHER_SEED", "42"))


def _bottom_slab_indices(atoms, n):
    z = atoms.get_positions()[:, 2]
    sym = np.array(atoms.get_chemical_symbols())
    slab_sym = SPECORDER[-1]
    slab = np.where(sym == slab_sym)[0]
    if len(slab) == 0:
        slab = np.arange(len(atoms))
    return slab[np.argsort(z[slab])][:n].tolist()


def main():
    t0 = time.time()
    if os.path.exists(DONE):
        os.remove(DONE)
    atoms = read(STRUCT, format="vasp")
    print(f"loaded {len(atoms)} atoms; constraints={atoms.constraints}", flush=True)
    has_fix = any(isinstance(c, FixAtoms) for c in atoms.constraints)
    if not has_fix and FIXED_BOTTOM_N > 0:
        fixed = _bottom_slab_indices(atoms, FIXED_BOTTOM_N)
        atoms.set_constraint(FixAtoms(indices=fixed))
        print(f"re-applied FixAtoms on {len(fixed)} bottom-slab atoms", flush=True)

    print("building teacher calculator ...", flush=True)
    teacher = get_teacher(CFG)
    atoms.calc = teacher.calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMP, rng=np.random.default_rng(SEED))
    Stationary(atoms)

    dyn = Langevin(atoms, DT_FS * units.fs, temperature_K=TEMP,
                   friction=FRICTION, rng=np.random.default_rng(SEED))

    if os.path.exists(OUT_XYZ):
        os.remove(OUT_XYZ)

    saved = {"n": 0}
    log = open(LOGFILE, "w")

    def save_frame():
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces()
        snap = atoms.copy()
        snap.info["energy"] = e
        snap.arrays["forces"] = f
        write(OUT_XYZ, snap, format="extxyz", append=True)
        saved["n"] += 1
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))
        msg = (f"step {dyn.nsteps:6d}  t={dyn.nsteps*DT_FS:7.1f} fs  "
               f"E={e:.4f} eV  T={temp:6.1f} K  saved={saved['n']}  "
               f"wall={time.time()-t0:.1f}s")
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    dyn.attach(save_frame, interval=SAVE_EVERY)

    save_frame()  # frame 0
    print(f"running {N_STEPS} steps @ dt={DT_FS} fs, save_every={SAVE_EVERY} ...", flush=True)
    dyn.run(N_STEPS)

    log.close()
    print(f"teacher MD done: {saved['n']} frames -> {OUT_XYZ}  in {time.time()-t0:.1f}s", flush=True)
    with open(DONE, "w") as fp:
        fp.write(f"OK {saved['n']} frames\n")


if __name__ == "__main__":
    main()
