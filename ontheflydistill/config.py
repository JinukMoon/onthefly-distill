"""Configuration loader for onthefly-distill.

All environment coupling (paths, binaries, species, teacher, AL knobs, remote
host) flows through a single ``config.yaml`` plus a few environment-variable
overrides. Nothing here imports torch, fairchem, or any heavy dependency, so the
package stays import-light.

Resolution rules:
  * ``lmp_bin``    : config ``lmp_bin`` -> ``$LMP_BIN`` env -> clear error.
  * ``python_bin`` : config ``python_bin`` -> ``$PYTHON_BIN`` env -> ``sys.executable``.
  * config file    : ``$ONTHEFLY_CONFIG`` env -> ``./config.yaml`` in CWD.

Pt-water defaults are baked in so that, with no config present, the package
reproduces the original Pt(111)-water acid HER behaviour.
"""

import os
import sys
import functools

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
        "pyyaml is required to load config.yaml. Install it with "
        "`pip install pyyaml` (or `pip install -e .`)."
    ) from exc


# ------------------------------------------------------------------
# Pt-water defaults (used when a key is absent from config.yaml).
# ------------------------------------------------------------------
DEFAULTS = {
    "python_bin": None,            # falls back to sys.executable
    "work_dir": "./run",
    "system": {
        "init_structure": "examples/ptwater_acid/ptwater_acid_init_clean.vasp",
        "species": [1, 8, 78],          # H, O, Pt (atomic numbers)
        "specorder": ["H", "O", "Pt"],  # MUST match pair_coeff element order
        "masses": [1.008, 15.999, 195.084],
        "fixed_bottom_n": 32,           # frozen bottom-slab atoms (0 to disable)
        "force_pbc": True,              # force atoms.pbc=True before labeling
    },
    "teacher": {
        "type": "ase_calculator",       # ase_calculator | uma | extxyz
        "calculator": None,             # "mod:func" import-path callable (ase_calculator)
        "model": "uma-s-1p2",           # uma  (consumed by teacher_md.py AND label.py)
        "task": "oc25",                 # uma  (MUST match between MD and labeling)
        "device": "cuda",               # uma
        "dataset": None,                # extxyz teacher dataset path
    },
    "student": {
        "n_radial_basis": 12,
        "n_radial_funcs": 8,
        "r_max": 6.0,
        "hidden_dims": [64, 32],
        "embed_dim": 16,
        "activation": "silu",
        "zbl": True,
        "zbl_r_inner": 0.5,
        "zbl_r_outer": 3.0,
        "nu_max": 2,
        "epochs": 300,
    },
    "al_loop": {
        "target_ps": 100000,
        "no_progress_limit": 4,
        "max_iter": 30,
        "seed": 42,
        "omp_threads": 8,
    },
    "remote": {
        "enabled": False,
        "host": "myserver",
        "base": "/remote/path",
        "slurm_partition": "gpu",
        "slurm_gres": "gpu:1",
    },
}


def _deep_merge(base, override):
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_path():
    """Return the resolved config.yaml path (may not exist)."""
    return os.environ.get("ONTHEFLY_CONFIG", os.path.join(os.getcwd(), "config.yaml"))


@functools.lru_cache(maxsize=None)
def load(path=None):
    """Load config.yaml merged over Pt-water defaults. Missing file => defaults."""
    if path is None:
        path = config_path()
    user = {}
    if path and os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user)
    # carry through any top-level keys we did not template (e.g. lmp_bin)
    for k, v in (user or {}).items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def get(*keys, default=None, cfg=None):
    """Nested lookup: get('system', 'species'). Returns ``default`` if absent."""
    node = cfg if cfg is not None else load()
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def lmp_bin(cfg=None):
    """Resolve the LAMMPS binary: config ``lmp_bin`` -> ``$LMP_BIN`` -> error."""
    cfg = cfg if cfg is not None else load()
    candidate = cfg.get("lmp_bin") or os.environ.get("LMP_BIN")
    if not candidate:
        raise RuntimeError(
            "LAMMPS binary not configured. Build lmp per lammps/BUILD.md, then set "
            "`lmp_bin: /path/to/lmp` in config.yaml or export LMP_BIN=/path/to/lmp."
        )
    return candidate


def python_bin(cfg=None):
    """Interpreter used for sub-step subprocesses."""
    cfg = cfg if cfg is not None else load()
    return cfg.get("python_bin") or os.environ.get("PYTHON_BIN") or sys.executable
