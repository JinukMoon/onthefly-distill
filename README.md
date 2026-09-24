# onthefly-distill

On-the-fly distillation of any teacher machine-learning interatomic potential
(MLIP) into a **CPU-deployable LAMMPS NN-MTP student**, driven by an
active-learning (AL) loop.

## 1. Overview / concept

A large, accurate teacher MLIP (e.g. a foundation model such as UMA/FAIRChem) is
expensive to run in long molecular dynamics (MD). This framework distills it into
a tiny **NN-MTP** (neural-network moment-tensor potential) student that runs in
LAMMPS on CPUs with a pure-C++ pair style (no LibTorch, no Python at MD time).

The student is trained on teacher-labeled configurations, then iteratively
hardened: wherever the student's own MD drifts into unphysical territory and
crashes, those pre-crash configurations are relabeled by the teacher and folded
back into training. The result is a fast student potential validated against the
teacher's surface.

Pipeline: **teacher MLIP -> NN-MTP student -> CPU LAMMPS deployment.**

## 2. The active-learning loop

```
        +------------------------------------------------------------+
        |                                                            |
        v                                                            |
  teacher MD  --->  train NN-MTP student  --->  student LAMMPS MD     |
 (E/F frames)            (.bin export)         (seed, target_ps)      |
                                                     |               |
                                          stable to target? --yes--> SUCCESS
                                                     | no                  (stop)
                                                     v                     |
                                       detect failure (close-contact       |
                                       / blow-up) + keep pre-crash window   |
                                                     |                     |
                                       relabel pre-crash frames with        |
                                       the teacher  --->  enrich pool ------+
                                                     |
                                       no crash-time progress for N rounds -> STALLED (stop)
```

If the teacher is a pre-labeled extxyz dataset (no live calculator), it cannot
relabel new frames, so the loop runs **train -> student MD -> report once** and
stops: static one-shot distillation, AL disabled.

## 3. Repo layout

```
onthefly-distill/
├── README.md
├── LICENSE                        # GPL-2.0 (LAMMPS-derived pair style)
├── CITATION.cff
├── pyproject.toml                 # `pip install -e .`
├── requirements.txt
├── environment.yml
├── config.example.yaml            # copy -> config.yaml
│
├── ontheflydistill/               # importable package
│   ├── __init__.py
│   ├── common.py                  # NN-MTP model + training + .bin export
│   ├── config.py                  # config.yaml + env resolution
│   ├── train_student.py           # build cache + train + export model.bin
│   ├── student_md_lammps.py       # student LAMMPS MD + pre-crash detection
│   ├── rattle_augment.py          # geometry augmentation
│   ├── merge_xyz.py               # merge extxyz files
│   ├── integrity_check.py         # optional stability metric
│   └── teachers/                  # teacher abstraction
│       ├── __init__.py            # get_teacher(config) factory
│       ├── base.py                # Teacher protocol + can_relabel
│       ├── ase_calculator.py      # any ASE Calculator (DEFAULT)
│       ├── uma_teacher.py         # UMA/FAIRChem example (optional dep)
│       └── extxyz_teacher.py      # pre-labeled extxyz (one-shot)
│
├── scripts/
│   ├── teacher_md.py              # teacher MD via get_teacher
│   ├── label.py                   # label extxyz via the configured teacher
│   ├── can_relabel.py             # capability probe used by the loops
│   ├── al_loop_local.sh           # AL loop, local teacher (DEFAULT)
│   ├── al_loop_remote.sh          # AL loop, remote teacher (OPTIONAL)
│   └── remote/
│       └── label_slurm.sh.example # remote SLURM labeling template
│
├── lammps/
│   ├── src/pair_nnmtp*.{cpp,h}    # the NN-MTP pair style (v1 + v2)
│   └── BUILD.md
│
└── examples/
    └── ptwater_acid/              # Pt(111)-water example (init .vasp + config + README)
```

## 4. Install

```bash
git clone <this repo> && cd onthefly-distill
conda env create -f environment.yml      # or: pip install -r requirements.txt
pip install -e .                          # REQUIRED
```

`pip install -e .` is **required**: the loop scripts run the package as
subprocesses and do `from ontheflydistill import ...`. Without the editable
install, the first training step fails with `ModuleNotFoundError`.

## 5. Plug in a teacher

Two paths, selected by `teacher.type` in `config.yaml`:

**(a) ASE-Calculator adapter (DEFAULT) -> full AL loop.** Write a zero-arg callable
that returns any `ase.calculators.calculator.Calculator` and point config at it:

```yaml
teacher:
  type: ase_calculator
  calculator: "mypkg.mycalc:make_calc"   # "module:function"
```

The UMA/FAIRChem example (`teacher.type: uma`) is a built-in instance of this path
(`fairchem-core` is an optional dependency, imported lazily).

**(b) Pre-labeled extxyz -> static one-shot distillation (AL disabled).** If you
already have a labeled trajectory:

```yaml
teacher:
  type: extxyz
  dataset: data/labeled.extxyz
```

There is no calculator to relabel student-visited frames, so the loop trains the
student, runs it once, reports, and exits.

> **PES-consistency warning.** The labeler's `model`+`task` MUST match the
> teacher-MD `model`+`task` — both come from the single `teacher:` block. Running
> the teacher MD on one surface (e.g. `uma-s-1p1`/`omat`) but labeling on another
> (e.g. `uma-s-1p2`/`oc25`) puts labels and trajectory on different
> potential-energy surfaces and corrupts training.

## 6. Build the LAMMPS pair style

The student runs in LAMMPS via `pair_style nnmtp` (pure C++, no LibTorch). See
[`lammps/BUILD.md`](lammps/BUILD.md). Set `lmp_bin:` in `config.yaml` (or export
`LMP_BIN`) to the resulting binary.

## 7. Run the Pt-water example

See [`examples/ptwater_acid/README.md`](examples/ptwater_acid/README.md): teacher
MD with UMA `uma-s-1p2`/`oc25`, then `scripts/al_loop_local.sh` to distill and
harden the student.

## 8. Configuration reference

All keys live in `config.yaml` (see `config.example.yaml`). The loop reads
`config.yaml` in the working directory, or the file named by `$ONTHEFLY_CONFIG`.

| Key | Meaning |
|---|---|
| `lmp_bin` | LAMMPS binary with `pair_style nnmtp` (REQUIRED; or `$LMP_BIN`). |
| `python_bin` | Interpreter for sub-step subprocesses. |
| `work_dir` | AL working directory (datasets, models, runs, logs). |
| `system.init_structure` | Initial structure (VASP POSCAR). |
| `system.species` | Atomic numbers of the species (max 4). |
| `system.specorder` | Element order; MUST match the `pair_coeff` line. |
| `system.masses` | Masses (g/mol) aligned to `specorder`. |
| `system.fixed_bottom_n` | Frozen bottom-slab atoms (0 to disable). |
| `system.force_pbc` | Force `atoms.pbc=True` before labeling (mixed-PBC slab dumps). |
| `teacher.type` | `ase_calculator` \| `uma` \| `extxyz`. |
| `teacher.calculator` | `ase_calculator`: `"module:function"` returning a Calculator. |
| `teacher.model`, `teacher.task` | `uma`: model + task (identical for MD and labeling). |
| `teacher.device` | `uma`: torch device. |
| `teacher.dataset` | `extxyz`: pre-labeled dataset path. |
| `student.*` | NN-MTP hyperparameters (radial basis, r_max, hidden dims, ZBL, epochs). |
| `student.select_energy_scale`, `student.select_force_scale` | The kept model minimizes validation E_MAE/energy_scale + F_MAE/force_scale (defaults 1.0 meV/atom, 10.0 meV/A). |
| `al_loop.target_ps` | Student MD stability target (ps). |
| `al_loop.no_progress_limit` | Stop after this many rounds without crash-time gain. |
| `al_loop.max_iter` | Hard backstop on AL rounds. |
| `al_loop.seed`, `al_loop.omp_threads` | MD seed; OpenMP threads. |
| `remote.*` | OPTIONAL remote-teacher path (`enabled`, `host`, `base`, SLURM partition/gres). |

## 9. Limitations

- **Max 4 species.** `pair_nnmtp.h` declares `int species_Z[4]`; the pair style
  supports at most 4 elements (a fixed C++ array bound).
- **CPU pair style only.** `pair_nnmtp` is a CPU C++ implementation; no GPU pair
  style is provided.
- **extxyz teacher = one-shot.** A pre-labeled dataset has no calculator, so the
  AL loop is disabled (static distillation only).
- **No CI / runtime verification.** This package is assembled statically; it ships
  no test suite or end-to-end verification run.

## 10. License

**GPL-2.0.** The `pair_nnmtp` pair style is derived from LAMMPS (Sandia National
Laboratories, https://www.lammps.org), which is distributed under GPL-2.0;
bundling the framework with it makes the repository as a whole GPL-2.0. See
[`LICENSE`](LICENSE) and the attribution note in [`lammps/BUILD.md`](lammps/BUILD.md).

## 11. Citation

If you use this software, please cite it via [`CITATION.cff`](CITATION.cff), and
cite **LAMMPS** as the origin of the pair style.
