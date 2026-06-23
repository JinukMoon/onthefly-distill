# Building LAMMPS with the NN-MTP pair style

The `pair_nnmtp` pair style is **pure C++ — no LibTorch, no Python**. It compiles
in a stock LAMMPS with no extra packages. At runtime it reads the student model
from the `.bin` file that `ontheflydistill.common.export_v1` writes (magic
`NNMTP1`); that binary format is the contract between the Python exporter and this
C++ pair style.

## Recommended path (no vendored LAMMPS tree)

```bash
# 1. Get a LAMMPS source tree
git clone -b stable https://github.com/lammps/lammps

# 2. Drop in the pair style sources from this repo
cp onthefly-distill/lammps/src/pair_nnmtp.*    lammps/src/
cp onthefly-distill/lammps/src/pair_nnmtp_v2.* lammps/src/   # optional experimental variant

# 3. Configure + build (no special packages needed for nnmtp; MOLECULE/KSPACE optional)
cd lammps && mkdir build && cd build
cmake ../cmake -DCMAKE_BUILD_TYPE=Release
make -j

# 4. Point config.yaml at the binary
#    lmp_bin: <path>/lammps/build/lmp
#    (or export LMP_BIN=<path>/lammps/build/lmp)
```

The `pair_nnmtp` source always compiles; the CMake logic only excludes Torch-based
pair styles (`pair_e3gnn*`, `pair_nequip_allegro*`) when Torch is off — `nnmtp` is
unaffected.

## Pair style usage

```
pair_style      nnmtp
pair_coeff      * * model.bin <El1> <El2> ...
```

- `nnmtp`  (v1) — the pair style the distillation pipeline uses (default).
- `nnmtp2` (v2) — a separate experimental variant (`pair_nnmtp_v2`); ships for
  completeness but is not used by the loop.

The element symbols after the `.bin` path map LAMMPS atom types 1..N to elements;
their order MUST match `system.specorder` in `config.yaml`.

## Limitation

`pair_nnmtp.h` declares `int species_Z[4]`, so the pair style supports at most
**4 species**. This is a fixed array bound in the C++; raising it requires editing
the source.

## Attribution / License

This pair style is derived from LAMMPS (Sandia National Laboratories,
https://www.lammps.org), distributed under the **GNU General Public License,
version 2 (GPL-2.0)**. Because this code is bundled with the GPL-2.0 pair style,
the repository as a whole is effectively GPL-2.0. See the top-level `LICENSE`.
