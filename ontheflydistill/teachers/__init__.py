"""Teacher factory.

``get_teacher(config)`` reads the ``teacher:`` block (and ``system.force_pbc``)
and returns the right teacher. No heavy dependency (fairchem/torch) is imported
until the corresponding teacher is actually constructed.
"""

from .base import Teacher, BaseTeacher

__all__ = ["Teacher", "BaseTeacher", "get_teacher"]


def get_teacher(config=None):
    """Build a teacher from config.

    ``config`` may be a full config dict; if None, the package config is loaded.
    """
    from ontheflydistill import config as _cfg
    cfg = config if config is not None else _cfg.load()

    ttype = _cfg.get("teacher", "type", default="ase_calculator", cfg=cfg)
    force_pbc = bool(_cfg.get("system", "force_pbc", default=True, cfg=cfg))

    if ttype == "ase_calculator":
        from .ase_calculator import ASECalculatorTeacher
        return ASECalculatorTeacher(
            _cfg.get("teacher", "calculator", default=None, cfg=cfg),
            force_pbc=force_pbc,
        )
    if ttype == "uma":
        from .uma_teacher import UMATeacher
        return UMATeacher(
            model=_cfg.get("teacher", "model", default="uma-s-1p2", cfg=cfg),
            task=_cfg.get("teacher", "task", default="oc25", cfg=cfg),
            device=_cfg.get("teacher", "device", default="cuda", cfg=cfg),
            force_pbc=force_pbc,
        )
    if ttype == "extxyz":
        from .extxyz_teacher import ExtxyzTeacher
        return ExtxyzTeacher(_cfg.get("teacher", "dataset", default=None, cfg=cfg))
    raise ValueError(
        f"Unknown teacher.type {ttype!r}; expected ase_calculator | uma | extxyz."
    )
