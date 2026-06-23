"""
STEP 3 (LOCAL): run student NN-MTP MD in LAMMPS from the init structure,
detect failure with a GENERALIZED close-contact criterion (any pair
d_ij < 0.6*(r_i+r_j), covering O-O, O-H, metal-O, etc.), and collect a window of
pre-crash frames for teacher relabeling.

Reuses:
  - LAMMPS pair_nnmtp binary (resolved via config.yaml `lmp_bin` / $LMP_BIN).
  - boundary p p f, FixAtoms (bottom slab atoms) + wall/reflect z, Langevin.

Init structure, specorder, masses and the number of frozen bottom-slab atoms all
come from config.yaml (``system.*``), defaulting to the Pt(111)-water example.

Usage:
  python -m ontheflydistill.student_md_lammps <model.bin> <run_dir> [--total-ps 200] [--seed 42]
Outputs in <run_dir>/:
  md.in, md.log, md.dump (element column), driver finished marker via stdout,
  pre_crash.xyz (frames in the warning zone just before first close contact),
  failure.json (first_close_contact info)
"""
import os, sys, time, json, argparse, subprocess
import numpy as np
from ase.io import read, write
from ase.data import covalent_radii, atomic_numbers
from ase.geometry import get_distances

from ontheflydistill import config as _cfg

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- environment / system config (Pt-water defaults) ----
SPECORDER = list(_cfg.get("system", "specorder", default=["H", "O", "Pt"]))
MASSES = list(_cfg.get("system", "masses", default=[1.008, 15.999, 195.084]))
FIXED_BOTTOM_N = int(_cfg.get("system", "fixed_bottom_n", default=32))
_init = _cfg.get("system", "init_structure",
                 default=os.path.join(HERE, "..", "examples", "ptwater_acid",
                                      "ptwater_acid_init_clean.vasp"))
STRUCT = os.path.abspath(_init)

CONTACT_FRAC = 0.60          # d < frac*(r_i+r_j) => collapsed/close-contact
# Keep a WIDE window before collapse so we capture the PHYSICAL drift region (not just
# the near-collapse force explosion). Teacher relabels all; build_cache's FMAX_FILTER=50
# then drops only the collapse-adjacent high-force frames, keeping the physical configs
# the student actually mishandled (AL), NOT gendistill-style near-TS/collapse configs.
PRE_CRASH_WINDOW = int(os.environ.get("STUDENT_PRECRASH_WINDOW", "200"))  # frames before first contact


def _lmp_bin():
    return _cfg.lmp_bin()


def get_fixed_indices(atoms):
    from ase.constraints import FixAtoms
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            return list(c.get_indices())
    if FIXED_BOTTOM_N <= 0:
        return []
    # fallback: N lowest-z atoms of the heaviest (slab) species in specorder
    z = atoms.get_positions()[:, 2]
    sym = np.array(atoms.get_chemical_symbols())
    slab_sym = SPECORDER[-1]
    slab = np.where(sym == slab_sym)[0]
    if len(slab) == 0:
        slab = np.arange(len(atoms))
    return slab[np.argsort(z[slab])][:FIXED_BOTTOM_N].tolist()


def min_scaled_contact(atoms):
    """Return (min ratio d_ij/(0.6*(r_i+r_j)), pair_info).
    ratio < 1 means a collapsed/close contact for that pair."""
    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    rcov = np.array([covalent_radii[z] for z in nums])
    D = get_distances(pos, pos, cell=np.array(atoms.get_cell()), pbc=atoms.get_pbc())[1]
    thresh = CONTACT_FRAC * (rcov[:, None] + rcov[None, :])
    # avoid 0/0 on the diagonal: compute ratio first, then mask the diagonal to inf
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = D / thresh
    np.fill_diagonal(ratio, np.inf)
    ratio = np.nan_to_num(ratio, nan=np.inf, posinf=np.inf)
    i, j = np.unravel_index(np.argmin(ratio), ratio.shape)
    return float(ratio[i, j]), (int(i), int(j), atoms.get_chemical_symbols()[i],
                                atoms.get_chemical_symbols()[j], float(D[i, j]))


def run(model_bin, run_dir, total_ps, seed, dt=0.0005, temp=300.0, damp=0.1, omp=4):
    os.makedirs(run_dir, exist_ok=True)
    atoms = read(STRUCT, format="vasp")
    n_atoms = len(atoms)
    fixed = get_fixed_indices(atoms)
    fixed_lmp = [i + 1 for i in fixed]
    struct_data = os.path.join(run_dir, "structure_init.data")
    write(struct_data, atoms, format="lammps-data", specorder=SPECORDER)

    mass_lines = "\n".join(f"mass            {i+1} {m}" for i, m in enumerate(MASSES))
    ids_str = " ".join(str(x) for x in fixed_lmp)
    # 0.1 ps for short runs, but cap total frames (~50k) for long targets so a
    # 100 ns run does not explode to ~1e6 frames (also cuts dump I/O => faster MD).
    _max_frames = int(os.environ.get("STUDENT_MAX_FRAMES", "50000"))
    dump_every = max(int(round(0.1 / dt)), int(round((total_ps / dt) / _max_frames)))
    thermo_every = int(round(1.0 / dt))
    steps = int(round(total_ps / dt))
    dump_file = os.path.join(run_dir, "md.dump")
    in_file = os.path.join(run_dir, "md.in")
    log_file = os.path.join(run_dir, "md.log")

    # frozen-group / wall lines only when there ARE frozen atoms (fixed_bottom_n>0)
    if fixed_lmp:
        group_block = f"""group           frozen id {ids_str}
group           mobile subtract all frozen
velocity        frozen set 0.0 0.0 0.0
velocity        mobile create {temp} {seed} dist gaussian
fix             1 frozen setforce 0.0 0.0 0.0
fix             2 mobile langevin {temp} {temp} {damp} {seed}
fix             3 mobile nve
fix             w_top  mobile wall/reflect zhi EDGE
fix             w_bot  mobile wall/reflect zlo EDGE
compute         t_mobile mobile temp
thermo_modify   temp t_mobile"""
        thermo_temp = "c_t_mobile"
    else:
        group_block = f"""velocity        all create {temp} {seed} dist gaussian
fix             2 all langevin {temp} {temp} {damp} {seed}
fix             3 all nve"""
        thermo_temp = "temp"

    lammps_input = f"""units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p f
read_data       {struct_data}
{mass_lines}
pair_style      nnmtp
pair_coeff      * * {os.path.abspath(model_bin)} {' '.join(SPECORDER)}

{group_block}
timestep        {dt}

dump            d_traj all custom {dump_every} {dump_file} id type element x y z
dump_modify     d_traj sort id element {' '.join(SPECORDER)}

thermo          {thermo_every}
thermo_style    custom step time {thermo_temp} pe ke etotal press
run             {steps}
write_data      {os.path.join(run_dir, 'md.data')}
"""
    with open(in_file, "w") as f:
        f.write(lammps_input)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    print(f"[student MD] n_atoms={n_atoms} fixed={len(fixed)} "
          f"steps={steps} ({total_ps} ps) dt={dt*1000} fs seed={seed}", flush=True)
    print(f"[student MD] LAMMPS log -> {log_file}", flush=True)
    t0 = time.time()
    with open(log_file, "w") as logf:
        r = subprocess.run([_lmp_bin(), "-in", in_file], stdout=logf,
                           stderr=subprocess.STDOUT, cwd=run_dir, env=env)
    elapsed = time.time() - t0
    print(f"[student MD] finished in {elapsed/60:.1f} min (returncode={r.returncode})", flush=True)

    # ---- failure detection on the dump (close-contact AND blow-up/crash) ----
    extract_pre_crash(dump_file, run_dir, total_ps_run=total_ps,
                      returncode=r.returncode, frame_spacing_ps=dump_every * dt,
                      log_file=log_file)
    print("[student MD] DRIVER_FINISHED", flush=True)


def extract_pre_crash(dump_file, run_dir, total_ps_run, returncode=0,
                      frame_spacing_ps=0.1, log_file=None):
    out_xyz = os.path.join(run_dir, "pre_crash.xyz")
    fail_json = os.path.join(run_dir, "failure.json")
    if not os.path.exists(dump_file):
        json.dump({"status": "no_dump"}, open(fail_json, "w"), indent=2)
        write(out_xyz, [], format="extxyz")
        print("[extract] no dump file", flush=True)
        return
    try:
        frames = read(dump_file, index=":", format="lammps-dump-text", specorder=SPECORDER)
    except Exception as e:
        json.dump({"status": f"read_failed: {e}"}, open(fail_json, "w"), indent=2)
        write(out_xyz, [], format="extxyz")
        print(f"[extract] dump read failed: {e}", flush=True)
        return

    first_contact = None
    worst = None
    for i, atoms in enumerate(frames):
        ratio, pair = min_scaled_contact(atoms)
        if worst is None or ratio < worst[0]:
            worst = (ratio, i, pair)
        if ratio < 1.0:    # collapsed
            first_contact = i
            break

    # Did LAMMPS actually complete the full target run? A blow-up / lost-atoms / any crash
    # makes returncode != 0 (or stops the dump early, or logs ERROR) WITHOUT a close-contact,
    # because atoms fly APART rather than collide -> must still count as a FAILURE.
    last_ps = len(frames) * frame_spacing_ps
    reached_target = last_ps >= 0.95 * total_ps_run
    log_err = False
    if log_file and os.path.exists(log_file):
        try:
            _tail = open(log_file, errors="ignore").read()[-4000:]
            log_err = ("ERROR" in _tail) or ("Lost atoms" in _tail)
        except Exception:
            pass
    lammps_crashed = (returncode != 0) or (not reached_target) or log_err

    if first_contact is not None:
        status, crash_mode = "collapsed", "close_contact"
    elif lammps_crashed:
        status, crash_mode = "collapsed", "blowup_lost_atoms"
    else:
        status, crash_mode = "stable_in_dump", None

    info = {
        "status": status,
        "crash_mode": crash_mode,
        "n_frames_in_dump": len(frames),
        "total_ps_requested": total_ps_run,
        "last_ps": last_ps,
        "reached_target": reached_target,
        "returncode": returncode,
        "log_error": log_err,
        "first_contact_frame": first_contact,
        "worst_ratio": worst[0] if worst else None,
        "worst_frame": worst[1] if worst else None,
        "worst_pair": worst[2] if worst else None,
        "contact_frac": CONTACT_FRAC,
    }

    if first_contact is not None:
        start = max(0, first_contact - PRE_CRASH_WINDOW)
        kept = frames[start:first_contact]   # pre-collapse warning zone (before close-contact)
        info["kept_range"] = [start, first_contact - 1]
        info["kept_reason"] = "pre_collapse_window"
    elif lammps_crashed:
        # blow-up: keep the window before the dump ended (drift leading to lost atoms)
        start = max(0, len(frames) - PRE_CRASH_WINDOW)
        kept = frames[start:]
        info["kept_range"] = [start, len(frames) - 1]
        info["kept_reason"] = "pre_blowup_tail"
    else:
        start = max(0, len(frames) - PRE_CRASH_WINDOW)
        kept = frames[start:]
        info["kept_range"] = [start, len(frames) - 1]
        info["kept_reason"] = "stable_tail"
    # strip any lingering calc; keep just geometry (teacher will relabel)
    clean = []
    for at in kept:
        a2 = at.copy()
        clean.append(a2)
    write(out_xyz, clean, format="extxyz")
    info["n_kept"] = len(clean)
    json.dump(info, open(fail_json, "w"), indent=2)
    print(f"[extract] status={info['status']} worst_ratio={info['worst_ratio']:.3f} "
          f"first_contact={first_contact} kept={len(clean)} -> {out_xyz}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_bin")
    ap.add_argument("run_dir")
    ap.add_argument("--total-ps", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--omp-threads", type=int, default=4)
    args = ap.parse_args()
    # control-file override of the stability target (change target WITHOUT relaunching
    # the orchestrator): file student_target_ps (in ps) next to the run dir.
    total_ps = args.total_ps
    _ctl = os.path.join(os.path.dirname(os.path.abspath(args.run_dir)), "student_target_ps")
    if os.path.exists(_ctl):
        try:
            total_ps = float(open(_ctl).read().strip())
            print(f"[student MD] target overridden by control file -> {total_ps} ps", flush=True)
        except Exception as _e:
            print(f"[student MD] control file unreadable ({_e}); using --total-ps {total_ps}", flush=True)
    run(args.model_bin, args.run_dir, total_ps, args.seed, omp=args.omp_threads)
