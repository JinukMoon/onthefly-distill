# Pt(111)-water acid HER example

A Pt(111) slab under an acidic water film (64 Pt / 60 O / 136 H, with excess
protons), the system the framework was developed on. The init structure
`ptwater_acid_init_clean.vasp` carries `Selective dynamics` with the bottom 32 Pt
atoms frozen.

This example distills a UMA teacher (`uma-s-1p2`, task `oc25`) into a CPU-deployable
NN-MTP student via the active-learning loop. The SAME `uma-s-1p2`/`oc25` surface is
used for both the teacher MD and the pre-crash labeling, so labels and teacher
trajectory live on one potential-energy surface.

## Prerequisites

- `pip install -e .` from the repo root (REQUIRED — the loop runs the package as
  subprocesses).
- A LAMMPS binary with `pair_style nnmtp` (see `lammps/BUILD.md`). Set its path in
  `config.yaml` (`lmp_bin:`) or export `LMP_BIN`.
- `fairchem-core` for the UMA teacher (`pip install fairchem-core`). To use a
  different teacher, set `teacher.type: ase_calculator` and point
  `teacher.calculator` at your own `module:make_calc`.

## Run

```bash
cd examples/ptwater_acid
export ONTHEFLY_CONFIG=$PWD/config.yaml
# 1) teacher MD -> teacher trajectory with E/F
python ../../scripts/teacher_md.py run/teacher_md.extxyz
# 2) seed the AL pool dataset (here: just the teacher MD; add rattle/merge as desired)
cp run/teacher_md.extxyz run/dataset.extxyz
# 3) run the active-learning loop (train -> student MD -> relabel -> repeat)
bash ../../scripts/al_loop_local.sh
```

The loop trains the student, runs it in LAMMPS, and on a crash relabels the
pre-crash window with the teacher, enriching the pool until the student is stable
to `al_loop.target_ps` (set small here for a quick demo) or AL stalls.

Notes:
- No dataset, caches, dumps, or trained `.bin` are shipped — they are regenerated.
- `target_ps` is small in this example config; raise it for a production run.
