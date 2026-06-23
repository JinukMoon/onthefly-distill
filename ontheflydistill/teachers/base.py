"""Teacher abstraction for onthefly-distill.

A *teacher* is whatever provides reference energies and forces for the student to
learn from. Two capabilities:

  * ``label(frames)`` — attach ``info['energy']`` + ``arrays['forces']`` to each
    frame. Every teacher implements this.
  * ``run_md(...)``   — drive a teacher MD trajectory. Only teachers backed by a
    live calculator (ASE-calculator / UMA) implement this; the pre-labeled-extxyz
    teacher cannot.

``can_relabel`` is the capability flag the AL loop branches on: ``True`` when the
teacher owns a calculator and can label student-visited pre-crash frames each
iteration, ``False`` for a static pre-labeled dataset (one-shot distillation).
"""

from typing import List, Protocol, runtime_checkable

try:
    from ase import Atoms
except Exception:  # pragma: no cover - ASE always present in practice
    Atoms = object  # type: ignore


@runtime_checkable
class Teacher(Protocol):
    """Minimal teacher protocol."""

    #: True if the teacher can relabel arbitrary new frames (owns a calculator).
    can_relabel: bool

    def label(self, frames: "List[Atoms]") -> "List[Atoms]":
        """Return frames with info['energy'] + arrays['forces'] attached."""
        ...


class BaseTeacher:
    """Convenience base class with the capability flag defaulted off."""

    can_relabel: bool = False

    def label(self, frames):  # pragma: no cover - abstract
        raise NotImplementedError

    def run_md(self, *args, **kwargs):  # pragma: no cover - optional
        raise NotImplementedError(
            f"{type(self).__name__} cannot drive teacher MD (no live calculator)."
        )
