"""ASE-Calculator teacher — the DEFAULT, portable path.

Wraps ANY ``ase.calculators.calculator.Calculator``: ``label()`` attaches the
calculator to each frame, reads energy + forces, detaches. This generalizes the
labeling loop the original Pt-water relabel scripts used.

Plug in your own teacher MLIP by pointing ``teacher.calculator`` in config.yaml
at an import-path callable ``"module:function"`` that returns a Calculator, e.g.::

    teacher:
      type: ase_calculator
      calculator: "mypkg.mycalc:make_calc"
"""

import importlib

from .base import BaseTeacher


def _resolve_calculator(spec):
    """Resolve a ``"module:function"`` string to a live ASE Calculator.

    The callable is imported and invoked with no arguments; it must return an
    ``ase.calculators.calculator.Calculator``. A Calculator instance or a
    zero-arg callable may also be passed directly (programmatic use).
    """
    if spec is None:
        raise ValueError(
            "teacher.calculator is not set. Point it at an import-path callable "
            '"module:function" that returns an ASE Calculator.'
        )
    if callable(spec):
        return spec()
    if hasattr(spec, "get_potential_energy"):
        return spec
    if isinstance(spec, str):
        if ":" not in spec:
            raise ValueError(
                f"teacher.calculator must be 'module:function', got {spec!r}."
            )
        mod_name, func_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, func_name)
        return factory()
    raise TypeError(f"Cannot resolve calculator from {spec!r}.")


class ASECalculatorTeacher(BaseTeacher):
    """Teacher backed by a live ASE Calculator. Can drive MD and relabel frames."""

    can_relabel = True

    def __init__(self, calculator, force_pbc=True):
        self.calc = _resolve_calculator(calculator)
        self.force_pbc = force_pbc

    def label(self, frames):
        labeled = []
        for atoms in frames:
            if self.force_pbc:
                # LAMMPS 'p p f' dumps come in mixed-PBC; many calculators reject
                # mixed PBC, so force [T, T, T] before labeling.
                atoms.pbc = True
            atoms.calc = self.calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            atoms.calc = None
            atoms.info["energy"] = float(e)
            atoms.arrays["forces"] = f
            labeled.append(atoms)
        return labeled
