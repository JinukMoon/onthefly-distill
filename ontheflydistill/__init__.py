"""onthefly-distill: on-the-fly MLIP distillation to a CPU-deployable NN-MTP student.

Importing this package has no heavy side effects (no torch CUDA init, no fairchem).
Submodules:
  common             NN-MTP v1 model + training + .bin export (Python<->C++ contract)
  config             config.yaml loader + env resolution (lmp_bin, species, teacher, ...)
  train_student      build cache + train student + export model.bin
  student_md_lammps  run student NN-MTP MD in LAMMPS + pre-crash detection
  teachers           teacher abstraction (ASE-calculator / UMA / pre-labeled extxyz)
"""

__version__ = "0.1.0"
