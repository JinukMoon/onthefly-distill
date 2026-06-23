"""
Shared utilities for OPM MLIP distillation.

Reuses autoablation's NN-MTP v1 model architecture + LAMMPS test harness.
Teacher = UMA-m-OMAT (trajectory stored as extended xyz with E/F).
Student = NN-MTP "balanced" config (f8b12_64x32_ZBL_r60 nu2, F_MAE=68.2 on Li6PS5Cl).

Two systems trained separately:
  - Co2MnO4: species [1, 8, 25, 27] = H, O, Mn, Co
  - RuMnO  : species [1, 8, 25, 44] = H, O, Mn, Ru
"""

import os
import sys
import time
import copy
import json
import struct
import subprocess
import tempfile
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from ase.io import read, write
from ase.neighborlist import neighbor_list as ase_nl

# ============================================================
# Fixed paths / constants (shared across systems)
# ============================================================
# LAMMPS binary is resolved lazily (config.yaml `lmp_bin` -> $LMP_BIN -> error) so
# that importing this module never requires a configured/built LAMMPS.
def _resolve_lmp_bin():
    from ontheflydistill.config import lmp_bin as _lmp_bin
    return _lmp_bin()

FMAX_FILTER = 50.0      # teacher data expected clean; keep loose safety filter
VAL_SPLIT = 0.1
SEED = 42
BATCH_SIZE = 32

STAGE1_EPOCHS = 150
STAGE2_EPOCHS = 150
STAGE1_LR = 0.005
STAGE2_LR = 0.001
PATIENCE = 40
LOG_EVERY = 5

# LAMMPS production-aligned test: 300 K Langevin, 50 ps, 0.5 fs dt.
# Smaller dt than teacher (1 fs) because student doesn't capture H-O high-freq
# modes (teacher trajectory was sampled every 100 fs, aliasing the 10 fs O-H
# period). Shorter run (50 ps) is sufficient to screen catastrophic failure.
LAMMPS_TEMP = 300.0
LAMMPS_STEPS = 100000          # 50 ps at 0.5 fs
LAMMPS_DT = 0.0005             # ps
LAMMPS_SEED = 42
TEMP_PASS_LOW = 270.0
TEMP_PASS_HIGH = 330.0
TEMP_CHECK_LAST_STEPS = 20000  # last 10 ps


# ============================================================
# Cache builder: extended xyz -> list of torch dicts with neighbor lists
# ============================================================
def build_cache(xyz_path, cache_path, structure_data_path, r_max=6.0, specorder=None):
    """Read UMA extxyz trajectory, filter Fmax<50, build neighbor lists, cache.

    Also writes first frame as LAMMPS data file (structure.data) for MD test.
    `specorder` (list of element symbols) must match the element_order used in
    the LAMMPS pair_coeff line, so atom type 1..N are assigned consistently.
    """
    print(f"Reading {xyz_path} ...", flush=True)
    t0 = time.time()
    frames = read(xyz_path, index=":")
    print(f"  {len(frames)} frames loaded in {time.time() - t0:.1f}s", flush=True)

    if specorder is None:
        specorder = sorted(set(frames[0].get_chemical_symbols()))
    # Write first frame as LAMMPS data (triclinic-aware)
    write(structure_data_path, frames[0], format="lammps-data", specorder=specorder)
    print(f"  first frame -> {structure_data_path} (specorder={specorder})")

    cache = []
    n_skip, n_err = 0, 0
    fmax_seen = []
    for i, atoms in enumerate(frames):
        try:
            f = atoms.get_forces()
            fmax = float(np.abs(f).max())
            fmax_seen.append(fmax)
            if fmax > FMAX_FILTER:
                n_skip += 1
                continue
            e = float(atoms.get_potential_energy())
            idx_i, idx_j, shifts = ase_nl("ijS", atoms, r_max, self_interaction=False)
            cache.append({
                "positions":        torch.tensor(atoms.get_positions(), dtype=torch.float32),
                "cell":             torch.tensor(np.array(atoms.get_cell()), dtype=torch.float32),
                "species":          torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
                "energy":           torch.tensor(e, dtype=torch.float32),
                "forces":           torch.tensor(f, dtype=torch.float32),
                "neighbor_indices": torch.tensor(np.stack([idx_i, idx_j], axis=1), dtype=torch.long),
                "neighbor_shifts":  torch.tensor(shifts, dtype=torch.float32),
                "n_atoms":          torch.tensor(len(atoms), dtype=torch.long),
            })
        except Exception as ex:
            n_err += 1

    torch.save(cache, cache_path)
    stats = {
        "n_total": len(frames),
        "n_kept": len(cache),
        "n_dropped_fmax": n_skip,
        "n_dropped_err": n_err,
        "fmax_median": float(np.median(fmax_seen)) if fmax_seen else 0.0,
        "fmax_p95": float(np.percentile(fmax_seen, 95)) if fmax_seen else 0.0,
        "fmax_max": float(np.max(fmax_seen)) if fmax_seen else 0.0,
    }
    with open(cache_path + ".stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"  cached {stats['n_kept']}/{stats['n_total']} frames -> {cache_path} ({size_mb:.1f} MB)")
    print(f"  Fmax median={stats['fmax_median']:.2f}, p95={stats['fmax_p95']:.2f}, max={stats['fmax_max']:.2f}")
    return stats


class CachedDataset(Dataset):
    def __init__(self, cache_path):
        self.data_list = torch.load(cache_path, weights_only=False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.data_list[i]


def collate_fn(batch):
    out = {k: [] for k in ["positions", "species", "energy", "forces",
                           "neighbor_indices", "neighbor_shifts", "n_atoms", "batch_idx"]}
    offset = 0
    for i, d in enumerate(batch):
        n = d["n_atoms"].item()
        out["positions"].append(d["positions"])
        out["species"].append(d["species"])
        out["energy"].append(d["energy"])
        out["forces"].append(d["forces"])
        out["neighbor_indices"].append(d["neighbor_indices"].clone() + offset)
        cart_shifts = d["neighbor_shifts"] @ d["cell"]
        out["neighbor_shifts"].append(cart_shifts)
        out["n_atoms"].append(d["n_atoms"])
        out["batch_idx"].append(torch.full((n,), i, dtype=torch.long))
        offset += n
    result = {}
    for k, v in out.items():
        if k in ("energy", "n_atoms"):
            result[k] = torch.stack(v)
        else:
            result[k] = torch.cat(v)
    return result


def get_dataloaders(cache_path):
    ds = CachedDataset(cache_path)
    nv = max(1, int(len(ds) * VAL_SPLIT))
    tr, va = random_split(ds, [len(ds) - nv, nv],
                          generator=torch.Generator().manual_seed(SEED))
    tl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"  Train={len(tr)}, Val={len(va)}")
    return tl, vl


# ============================================================
# NN-MTP v1 model (identical to autoablation)
# ============================================================
class ChebyshevRadialBasis(nn.Module):
    def __init__(self, n_basis=8, r_max=6.0):
        super().__init__()
        self.n_basis = n_basis
        self.r_max = r_max

    def cutoff_fn(self, r):
        x = r / self.r_max
        return torch.where(r < self.r_max, (1 - x ** 2) ** 2, torch.zeros_like(r))

    def forward(self, r):
        x = 2.0 * r / self.r_max - 1.0
        x = torch.clamp(x, -1.0, 1.0)
        basis = [torch.ones_like(x), x]
        for n in range(2, self.n_basis):
            basis.append(2 * x * basis[-1] - basis[-2])
        return torch.stack(basis, dim=-1) * self.cutoff_fn(r).unsqueeze(-1)


class ZBLRepulsion(nn.Module):
    def __init__(self, r_inner=0.5, r_outer=2.0):
        super().__init__()
        self.r_inner = r_inner
        self.r_outer = r_outer

    def forward(self, r, Z_i, Z_j):
        Z_i_f, Z_j_f = Z_i.float(), Z_j.float()
        a = 0.4543 * 0.529 / (Z_i_f ** 0.23 + Z_j_f ** 0.23)
        x_zbl = r / a
        phi = sum(c * torch.exp(d * x_zbl) for c, d in
                  zip([0.1818, 0.5099, 0.2802, 0.02817], [-3.2, -0.9423, -0.4028, -0.2016]))
        e_zbl = 14.3996 * Z_i_f * Z_j_f / r.clamp(min=1e-6) * phi
        x_sw = ((r - self.r_inner) / (self.r_outer - self.r_inner)).clamp(0, 1)
        switch = (1 - x_sw) ** 3 * (1 + 3 * x_sw + 6 * x_sw ** 2)
        return e_zbl * switch * (r < self.r_outer).float()


class MTPDescriptor(nn.Module):
    def __init__(self, species_list, n_radial_basis=12, n_radial_funcs=8, r_max=6.0, nu_max=2):
        super().__init__()
        self.species_list = species_list
        self.n_species = len(species_list)
        self.n_radial_basis = n_radial_basis
        self.n_radial_funcs = n_radial_funcs
        self.r_max = r_max
        self.nu_max = nu_max
        self._z_to_idx = {z: i for i, z in enumerate(species_list)}
        self.radial_basis = ChebyshevRadialBasis(n_radial_basis, r_max)
        n_pairs = self.n_species * self.n_species
        self.radial_coeffs = nn.Parameter(
            torch.randn(n_radial_funcs, n_pairs, n_radial_basis) * 0.01
        )
        n_mu = n_radial_funcs
        self.descriptor_dim = n_mu
        if nu_max >= 1:
            self.descriptor_dim += n_mu + n_mu * (n_mu - 1) // 2
        if nu_max >= 2:
            self.descriptor_dim += n_mu

    def _species_to_idx(self, species):
        idx = torch.zeros_like(species)
        for z, i in self._z_to_idx.items():
            idx = torch.where(species == z, torch.tensor(i, device=species.device), idx)
        return idx

    def forward(self, positions, cell, species, neighbor_indices, neighbor_shifts):
        idx_i = neighbor_indices[:, 0]
        idx_j = neighbor_indices[:, 1]
        rij = positions[idx_j] - positions[idx_i] + neighbor_shifts @ cell
        r = torch.norm(rij, dim=-1)
        mask = ((r < self.r_max) & (r > 1e-8)).float()
        sp_idx_i = self._species_to_idx(species[idx_i])
        sp_idx_j = self._species_to_idx(species[idx_j])
        r_safe = r.clamp(min=1e-8)
        rij_hat = rij / r_safe.unsqueeze(-1)
        Q = self.radial_basis(r)
        pair_idx = sp_idx_i * self.n_species + sp_idx_j
        coeffs = self.radial_coeffs[:, pair_idx, :].permute(1, 0, 2)
        f = torch.einsum('pmb,pb->pm', coeffs, Q) * mask.unsqueeze(-1)
        n_atoms = positions.shape[0]
        n_mu = self.n_radial_funcs
        idx_i_exp = idx_i.unsqueeze(-1).expand_as(f)
        M0 = torch.zeros(n_atoms, n_mu, device=positions.device, dtype=positions.dtype)
        M0 = M0.scatter_add(0, idx_i_exp, f)
        descriptors = [M0]
        if self.nu_max >= 1:
            f_rhat = f.unsqueeze(-1) * rij_hat.unsqueeze(1)
            idx_i_exp_3 = idx_i.unsqueeze(-1).unsqueeze(-1).expand_as(f_rhat)
            M1 = torch.zeros(n_atoms, n_mu, 3, device=positions.device, dtype=positions.dtype)
            M1 = M1.scatter_add(0, idx_i_exp_3, f_rhat)
            descriptors.append((M1 ** 2).sum(dim=-1))
            cross_dots = []
            for mu1 in range(n_mu):
                for mu2 in range(mu1 + 1, n_mu):
                    cross_dots.append((M1[:, mu1] * M1[:, mu2]).sum(dim=-1))
            if cross_dots:
                descriptors.append(torch.stack(cross_dots, dim=-1))
        if self.nu_max >= 2:
            outer = rij_hat.unsqueeze(-1) * rij_hat.unsqueeze(-2)
            f_outer = f.unsqueeze(-1).unsqueeze(-1) * outer.unsqueeze(1)
            idx_i_exp_33 = idx_i.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(f_outer)
            M2 = torch.zeros(n_atoms, n_mu, 3, 3, device=positions.device, dtype=positions.dtype)
            M2 = M2.scatter_add(0, idx_i_exp_33, f_outer)
            descriptors.append(M2[:, :, 0, 0] + M2[:, :, 1, 1] + M2[:, :, 2, 2])
        return torch.cat(descriptors, dim=-1)


class NNMTP(nn.Module):
    def __init__(self, species_list, n_radial_basis, n_radial_funcs, r_max,
                 hidden_dims, embed_dim=16, activation="silu", zbl=True, nu_max=2,
                 zbl_r_inner=0.5, zbl_r_outer=2.0):
        super().__init__()
        self.species_list = species_list
        self.n_species = len(species_list)
        self.r_max = r_max
        self.use_zbl = zbl
        self.descriptor = MTPDescriptor(species_list, n_radial_basis, n_radial_funcs,
                                        r_max, nu_max=nu_max)
        self.species_embedding = nn.Embedding(self.n_species, embed_dim)
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        input_dim = self.descriptor.descriptor_dim + embed_dim
        layers = []
        prev = input_dim
        for d in hidden_dims:
            layers += [nn.Linear(prev, d), act_fn()]
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self.energy_shift = nn.Parameter(torch.zeros(self.n_species))
        if self.use_zbl:
            self.zbl = ZBLRepulsion(r_inner=zbl_r_inner, r_outer=zbl_r_outer)

    def _species_to_idx(self, species):
        idx = torch.zeros_like(species)
        for i, z in enumerate(self.species_list):
            idx = torch.where(species == z, torch.tensor(i, device=species.device), idx)
        return idx

    def forward(self, positions, cell, species, neighbor_indices, neighbor_shifts,
                compute_forces=True):
        if compute_forces:
            positions.requires_grad_(True)
        sp_idx = self._species_to_idx(species)
        desc = self.descriptor(positions, cell, species, neighbor_indices, neighbor_shifts)
        features = torch.cat([desc, self.species_embedding(sp_idx)], dim=-1)
        atomic_energies = self.mlp(features).squeeze(-1) + self.energy_shift[sp_idx]
        energy = atomic_energies.sum()
        if self.use_zbl:
            idx_i, idx_j = neighbor_indices[:, 0], neighbor_indices[:, 1]
            rij = positions[idx_j] - positions[idx_i] + neighbor_shifts @ cell
            r = torch.norm(rij, dim=-1)
            energy = energy + 0.5 * self.zbl(r, species[idx_i], species[idx_j]).sum()
        result = {"energy": energy, "atomic_energies": atomic_energies}
        if compute_forces:
            result["forces"] = -torch.autograd.grad(
                energy, positions, create_graph=self.training, retain_graph=True)[0]
        return result


# ============================================================
# Training helpers
# ============================================================
def compute_loss(model, batch, ew, fw, device):
    pos = batch["positions"].to(device)
    sp = batch["species"].to(device)
    ni = batch["neighbor_indices"].to(device)
    ns = batch["neighbor_shifts"].to(device)
    bidx = batch["batch_idx"].to(device)
    tgt_e = batch["energy"].to(device)
    tgt_f = batch["forces"].to(device)
    na = batch["n_atoms"].to(device)
    cell_eye = torch.eye(3, device=device, dtype=pos.dtype)
    out = model(pos, cell_eye, sp, ni, ns, compute_forces=True)
    pred_e = torch.zeros(len(na), device=device)
    pred_e.scatter_add_(0, bidx, out["atomic_energies"])
    nf = na.float()
    return ew * ((pred_e / nf - tgt_e / nf) ** 2).mean() + fw * ((out["forces"] - tgt_f) ** 2).mean()


def validate(model, loader, device):
    model.eval()
    total_e_ae, total_f_ae = 0.0, 0.0
    n_frames, n_fcomp = 0, 0
    with torch.no_grad():
        for batch in loader:
            pos = batch["positions"].to(device).clone().detach().requires_grad_(True)
            sp = batch["species"].to(device)
            ni = batch["neighbor_indices"].to(device)
            ns = batch["neighbor_shifts"].to(device)
            na = batch["n_atoms"].to(device)
            bidx = batch["batch_idx"].to(device)
            cell_eye = torch.eye(3, device=device, dtype=pos.dtype)
            with torch.enable_grad():
                out = model(pos, cell_eye, sp, ni, ns, compute_forces=True)
            pred_e = torch.zeros(len(na), device=device)
            pred_e.scatter_add_(0, bidx, out["atomic_energies"])
            nf = na.float()
            total_e_ae += (pred_e / nf - batch["energy"].to(device) / nf).abs().sum().item()
            total_f_ae += (out["forces"] - batch["forces"].to(device)).abs().sum().item()
            n_frames += len(na)
            n_fcomp += batch["forces"].numel()
    model.train()
    return total_e_ae / max(n_frames, 1) * 1000, total_f_ae / max(n_fcomp, 1) * 1000


# ============================================================
# v1 binary export (NNMTP1)
# ============================================================
def export_v1(model, config, out_path):
    model.eval()
    n_species = len(config['species_list'])
    n_radial_basis = config['n_radial_basis']
    n_radial_funcs = config['n_radial_funcs']
    r_min = config.get('r_min', 0.0)
    r_max = config['r_max']
    embed_dim = config.get('embed_dim', 16)
    hidden_dims = config['hidden_dims']
    nu_max = config.get('nu_max', 2)
    n_mu = n_radial_funcs
    desc_dim = n_mu
    if nu_max >= 1:
        desc_dim += n_mu + n_mu * (n_mu - 1) // 2
    if nu_max >= 2:
        desc_dim += n_mu
    with open(out_path, 'wb') as f:
        f.write(b'NNMTP1')
        f.write(struct.pack('i', n_species))
        # Write species_list atomic numbers (new in this fork) so LAMMPS
        # pair_nnmtp can look up element names generically.
        for z in config['species_list']:
            f.write(struct.pack('i', int(z)))
        f.write(struct.pack('i', n_radial_basis))
        f.write(struct.pack('i', n_radial_funcs))
        f.write(struct.pack('d', r_min))
        f.write(struct.pack('d', r_max))
        f.write(struct.pack('i', desc_dim))
        f.write(struct.pack('i', embed_dim))
        # ZBL section: flag + (r_inner, r_outer) if enabled. Keeps per-model
        # ZBL settings bundled with weights so the C++ pair_nnmtp uses the
        # exact values the student was trained with.
        use_zbl = bool(getattr(model, "use_zbl", config.get("zbl", False)))
        f.write(struct.pack('i', 1 if use_zbl else 0))
        if use_zbl:
            r_inner = float(model.zbl.r_inner)
            r_outer = float(model.zbl.r_outer)
            f.write(struct.pack('d', r_inner))
            f.write(struct.pack('d', r_outer))
        n_mlp_layers = len(hidden_dims) + 1
        f.write(struct.pack('i', n_mlp_layers))
        all_dims = [desc_dim + embed_dim] + hidden_dims + [1]
        for d in all_dims:
            f.write(struct.pack('i', d))

        def write_vec(data):
            flat = data.detach().cpu().numpy().flatten()
            f.write(struct.pack('i', len(flat)))
            for v in flat:
                f.write(struct.pack('d', float(v)))

        write_vec(model.descriptor.radial_coeffs)
        write_vec(model.species_embedding.weight)
        write_vec(model.energy_shift)
        mlp_params = list(model.mlp.parameters())
        f.write(struct.pack('i', len(mlp_params)))
        for p in mlp_params:
            write_vec(p)
    print(f"  Exported v1: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB), desc_dim={desc_dim}")


# ============================================================
# LAMMPS temp stability test (300 K, NVT, 250 ps)
# ============================================================
def run_lammps_test(bin_path, structure_data, element_order, masses):
    """Minimize then run 250 ps NVT @ 300 K. Return temp_stable/steps_s dict.

    masses : list aligned to element_order, in g/mol.
    """
    mass_lines = "\n".join(f"mass            {i+1} {m}" for i, m in enumerate(masses))
    with tempfile.TemporaryDirectory() as tmpdir:
        relaxed = os.path.join(tmpdir, "relaxed.data")
        # Minimize
        min_input = f"""units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p
read_data       {structure_data}
{mass_lines}
pair_style      nnmtp
pair_coeff      * * {os.path.abspath(bin_path)} {element_order}
min_style       cg
minimize        1.0e-6 1.0e-8 1000 10000
write_data      {relaxed}
"""
        with open(os.path.join(tmpdir, "min.lmp"), "w") as f:
            f.write(min_input)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "8"
        r = subprocess.run([_resolve_lmp_bin(), "-in", os.path.join(tmpdir, "min.lmp")],
                           capture_output=True, text=True, cwd=tmpdir, env=env, timeout=300)
        if r.returncode != 0:
            return {"temp_stable": "CRASH", "avg_temp": 0.0,
                    "lammps_steps_s": 0.0, "lammps_time_s": 0.0,
                    "error": "minimize: " + r.stderr[-300:]}
        # MD
        md_input = f"""units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p
read_data       {relaxed}
{mass_lines}
pair_style      nnmtp
pair_coeff      * * {os.path.abspath(bin_path)} {element_order}
velocity        all create {LAMMPS_TEMP} {LAMMPS_SEED} dist gaussian
# Langevin thermostat (matches teacher UMA MD setup: friction = 1 ps^-1)
fix             1 all langevin {LAMMPS_TEMP} {LAMMPS_TEMP} 1.0 {LAMMPS_SEED}
fix             2 all nve
timestep        {LAMMPS_DT}
thermo          1000
thermo_style    custom step temp pe ke etotal press
run             {LAMMPS_STEPS}
"""
        with open(os.path.join(tmpdir, "md.lmp"), "w") as f:
            f.write(md_input)
        t0 = time.time()
        r = subprocess.run([_resolve_lmp_bin(), "-in", os.path.join(tmpdir, "md.lmp")],
                           capture_output=True, text=True, cwd=tmpdir, env=env, timeout=7200)
        wall = time.time() - t0
        if r.returncode != 0:
            return {"temp_stable": "CRASH", "avg_temp": 0.0,
                    "lammps_steps_s": 0.0, "lammps_time_s": wall,
                    "error": (r.stderr[-300:] + r.stdout[-300:])}
        temps = []
        steps_s = 0.0
        for line in r.stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    step = int(parts[0]); temp = float(parts[1])
                    if step > 0:
                        temps.append((step, temp))
                except (ValueError, IndexError):
                    pass
            if "timesteps/s" in line:
                try:
                    for i, p in enumerate(parts):
                        if p.startswith("timesteps/s"):
                            steps_s = float(parts[i - 1].rstrip(","))
                except (ValueError, IndexError):
                    pass
        if not temps:
            return {"temp_stable": "CRASH", "avg_temp": 0.0,
                    "lammps_steps_s": steps_s, "lammps_time_s": wall,
                    "error": "no thermo output"}
        last = [t for s, t in temps if s >= LAMMPS_STEPS - TEMP_CHECK_LAST_STEPS] or [t for _, t in temps[-10:]]
        avg = float(np.mean(last))
        return {
            "temp_stable": "PASS" if TEMP_PASS_LOW <= avg <= TEMP_PASS_HIGH else "FAIL",
            "avg_temp": round(avg, 1),
            "lammps_steps_s": round(steps_s, 1),
            "lammps_time_s": round(wall, 1),
        }
