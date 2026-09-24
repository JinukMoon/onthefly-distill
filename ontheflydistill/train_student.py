"""
STEP 2/4 (LOCAL): build cache from a teacher extxyz (possibly merged with
relabeled pre-crash frames) and train a tiny NN-MTP student with ZBL, then
export the LAMMPS .bin.

Reuses the vendored NN-MTP infra in ``ontheflydistill.common``.

Species / specorder / model hyperparameters come from config.yaml
(``system.species``, ``system.specorder``, ``student.*``), falling back to the
Pt(111)-water acid HER defaults: SPECIES_LIST = [1, 8, 78] (H, O, Pt), ZBL
widened (r_outer=3.0) so H-O / O-O / Pt-O pairs get a repulsive floor before the
NN can send them into a <1 A catastrophe.

Usage:
  python -m ontheflydistill.train_student <merged_teacher.extxyz> <out_prefix>
  -> writes <out_prefix>.pt, <out_prefix>.bin, <out_prefix>.cache.pt,
     <out_prefix>.structure.data  (all in this directory)
"""
import os, sys, time, copy
import torch
import numpy as np

from ontheflydistill import common
from ontheflydistill import config as _cfg

HERE = os.path.dirname(os.path.abspath(__file__))

# (c) Two-tier Fmax filtering (applied per-build inside train()): keep the normal cutoff
# on the BASE data (teacher+rattle) to drop over-rattled junk, but EXEMPT the AL pre-crash
# frames (real student-visited failure configs, teacher-labeled) so they are always kept.
FMAX_BASE = float(os.environ.get("STUDENT_FMAX_BASE", "50.0"))   # base teacher+rattle cutoff
FMAX_AL   = float(os.environ.get("STUDENT_FMAX_AL", "inf"))      # AL frames: no cutoff

# ---- species / specorder / model config from config.yaml (Pt-water defaults) ----
SPECIES_LIST = list(_cfg.get("system", "species", default=[1, 8, 78]))      # H, O, Pt
SPECORDER = list(_cfg.get("system", "specorder", default=["H", "O", "Pt"])) # match pair_coeff order

_student = _cfg.get("student", default={}) or {}
CONFIG = {
    "species_list":   SPECIES_LIST,
    "n_radial_basis": _student.get("n_radial_basis", 12),
    "n_radial_funcs": _student.get("n_radial_funcs", 8),
    "r_max":          _student.get("r_max", 6.0),
    "hidden_dims":    list(_student.get("hidden_dims", [64, 32])),
    "embed_dim":      _student.get("embed_dim", 16),
    "activation":     _student.get("activation", "silu"),
    "zbl":            _student.get("zbl", True),
    "zbl_r_inner":    _student.get("zbl_r_inner", 0.5),
    "zbl_r_outer":    _student.get("zbl_r_outer", 3.0),
    "nu_max":         _student.get("nu_max", 2),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EP_TOTAL = int(os.environ.get("STUDENT_EPOCHS", str(_student.get("epochs", 300))))
MIN_DELTA = float(os.environ.get("STUDENT_MIN_DELTA", "0.5"))   # meV/A; min F_MAE gain to reset early-stop patience

# The kept model is the one with the lowest E_MAE/SELECT_E_SCALE + F_MAE/SELECT_F_SCALE on the
# validation split, not the lowest F_MAE alone: training weights forces far above energies, so
# the energy error swings by several x between force-equivalent checkpoints and a force-only
# pick can keep one with a much worse energy. Early-stop patience still follows F_MAE.
# SELECT_E_SCALE=inf restores the force-only pick.
SELECT_E_SCALE = float(os.environ.get("STUDENT_SELECT_E_SCALE", str(_student.get("select_energy_scale", 1.0))))   # meV/atom
SELECT_F_SCALE = float(os.environ.get("STUDENT_SELECT_F_SCALE", str(_student.get("select_force_scale", 10.0))))   # meV/A


def train(xyz_path, out_prefix):
    cache_path = out_prefix + ".cache.pt"
    struct_data = out_prefix + ".structure.data"
    model_pt = out_prefix + ".pt"
    model_bin = out_prefix + ".bin"

    print(f"Device: {DEVICE}", flush=True)

    # ---- (c) cache = BASE (teacher+rattle) @ FMAX_BASE  +  AL pre-crash frames @ FMAX_AL ----
    # Base keeps the normal cutoff (drops over-rattled junk). The AL frames bypass the cutoff
    # (FMAX_AL=inf) = the AL signal we must keep (student's own physical failure trajectory).
    common.FMAX_FILTER = FMAX_BASE
    import gc
    import glob as _glob
    from ase.io import read as _read, write as _write
    _wdir = os.path.dirname(os.path.abspath(xyz_path))

    # The BASE cache is IDENTICAL across AL iterations (same dataset.extxyz, same FMAX_BASE,
    # same r_max), so building it from scratch every iteration recomputes ~16k neighbor lists
    # for ~25 min with zero benefit. Build it ONCE into a SHARED path (base.cache.pt next to
    # the dataset) and reuse; rebuild only when dataset.extxyz is newer than the cache.
    _base_cache = os.path.join(_wdir, "base.cache.pt")
    if os.path.exists(_base_cache) and os.path.getmtime(_base_cache) >= os.path.getmtime(xyz_path):
        print(f"base cache REUSE: {_base_cache} (>= mtime of {os.path.basename(xyz_path)})", flush=True)
    else:
        stats = common.build_cache(xyz_path, _base_cache, struct_data,
                                   r_max=CONFIG["r_max"], specorder=SPECORDER)
        print(f"base cache BUILT: {stats}", flush=True)
    base_list = torch.load(_base_cache, weights_only=False)
    n_base = len(base_list)

    # AL set GROWS each iter (glob picks up every al_iter*_labeled.extxyz) but it is tiny
    # (~hundreds of frames, seconds) so rebuilding it every iteration is cheap.
    _al_files = sorted(_glob.glob(os.path.join(_wdir, "al_iter*_labeled.extxyz")))
    n_al = 0
    al_list = []
    if _al_files:
        _al = []
        for _f in _al_files:
            try:
                _al += _read(_f, index=":")
            except Exception:
                pass
        if _al:
            _al_tmp = out_prefix + ".al_tmp.extxyz"
            _write(_al_tmp, _al, format="extxyz")
            common.FMAX_FILTER = FMAX_AL
            _al_cache = out_prefix + ".al.cache.pt"
            al_stats = common.build_cache(_al_tmp, _al_cache, out_prefix + ".al.data",
                                          r_max=CONFIG["r_max"], specorder=SPECORDER)
            al_list = torch.load(_al_cache, weights_only=False)
            n_al = len(al_list)

    # Always write the merged cache get_dataloaders consumes (base [+ AL]). List concat is
    # by reference (no tensor copy), so this does NOT double RAM; the earlier OOM came from
    # get_dataloaders re-loading a 2nd copy, which the del + gc.collect() below prevents.
    torch.save(base_list + al_list, cache_path)
    print(f"(c) cache: base {n_base} (cut {FMAX_BASE}) + AL {n_al} (uncut) "
          f"= {n_base + n_al} frames -> {os.path.basename(cache_path)}", flush=True)
    del al_list, base_list
    gc.collect()
    n_total_cached = n_base + n_al
    if n_total_cached < 2:
        sys.exit(f"too few frames cached ({n_total_cached}); aborting train")

    torch.manual_seed(42); np.random.seed(42)
    train_loader, val_loader = common.get_dataloaders(cache_path)
    model = common.NNMTP(**CONFIG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}, desc_dim={model.descriptor.descriptor_dim}, "
          f"zbl=(r_in={model.zbl.r_inner}, r_out={model.zbl.r_outer})", flush=True)

    # ---- WARM-START from previous AL iteration's model (model_iter{N-1}.pt) ----
    # GATED OFF BY DEFAULT (STUDENT_WARMSTART=0): warm-start + aggressive early-stop was
    # found to DEGRADE MD stability across AL iters (crash time 2076->528->258 ps, failure
    # mode shifting close-contact -> energy blowup = under-trained potential). Scratch +
    # full epochs each iteration is the default; set STUDENT_WARMSTART=1 to re-enable warm.
    import re as _re
    warm = False
    if os.environ.get("STUDENT_WARMSTART", "0") == "1":
        _m = _re.search(r"model_iter(\d+)$", os.path.basename(out_prefix))
        if _m:
            _prev = os.path.join(os.path.dirname(os.path.abspath(out_prefix)),
                                 f"model_iter{int(_m.group(1)) - 1}.pt")
            if os.path.exists(_prev):
                try:
                    model.load_state_dict(torch.load(_prev, map_location=DEVICE)["model_state_dict"])
                    warm = True
                    print(f"WARM-START from {_prev}", flush=True)
                except Exception as _e:
                    print(f"warm-start load failed ({_e}); scratch init", flush=True)
    if not warm:
        print(f"SCRATCH init (full {EP_TOTAL} ep, no warm-start)", flush=True)
    ep_total = int(os.environ.get("STUDENT_EPOCHS_WARM", "120")) if warm else EP_TOTAL
    patience_limit = (common.PATIENCE // 2) if warm else common.PATIENCE
    _lr = common.STAGE1_LR * (0.5 if warm else 1.0)

    best_score, best_e, best_f, best_f_patience, best_state = float("inf"), None, None, float("inf"), None
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=_lr)
    wu = int(ep_total * 0.2)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / max(wu, 1) if e < wu
        else max(0.01, 1 - (e - wu) / max(ep_total - wu, 1)))

    print(f"\nForce-focused stage ({ep_total} ep, patience={patience_limit}, warm={warm}); "
          f"kept model = min E/{SELECT_E_SCALE:g} + F/{SELECT_F_SCALE:g}", flush=True)
    patience_counter = 0
    for ep in range(ep_total):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = common.compute_loss(model, batch, 1e-5, 10.0, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sch.step()
        if (ep + 1) % common.LOG_EVERY == 0 or ep == ep_total - 1:
            e_mae, f_mae = common.validate(model, val_loader, DEVICE)
            tag = ""
            score = e_mae / SELECT_E_SCALE + f_mae / SELECT_F_SCALE
            if score < best_score:                      # keep best model on ANY improvement
                best_score, best_e, best_f = score, e_mae, f_mae
                best_state = copy.deepcopy(model.state_dict())
                tag = " (kept)"
            if f_mae < best_f_patience - MIN_DELTA:      # reset patience only on SIGNIFICANT gain
                best_f_patience = f_mae
                patience_counter = 0
                tag += " *best*"
            else:
                patience_counter += common.LOG_EVERY
            print(f"  Ep {ep+1:4d}/{ep_total}: E={e_mae:.1f} F={f_mae:.2f} p={patience_counter}{tag}",
                  flush=True)
            if patience_counter >= patience_limit:
                print(f"  Early stop at ep {ep+1}", flush=True)
                break

    training_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "model_config": CONFIG,
                "best_force_mae": best_f, "best_energy_mae": best_e,
                "selection": {"score": best_score, "energy_scale": SELECT_E_SCALE,
                              "force_scale": SELECT_F_SCALE},
                "training_time_s": training_time,
                "n_params": n_params, "system": "student",
                "source_xyz": xyz_path}, model_pt)
    common.export_v1(model, CONFIG, model_bin)
    if best_state:
        print(f"\nkept model: E_MAE {best_e:.2f} meV/atom, F_MAE {best_f:.2f} meV/A  in {training_time:.0f}s", flush=True)
    print(f"saved: {model_pt}", flush=True)
    print(f"saved: {model_bin}", flush=True)


if __name__ == "__main__":
    xyz = sys.argv[1]
    prefix = sys.argv[2]
    train(xyz, prefix)
