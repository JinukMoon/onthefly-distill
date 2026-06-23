"""Pre-labeled-extxyz teacher — static one-shot distillation path.

If you already have a labeled trajectory (E/F attached), set::

    teacher:
      type: extxyz
      dataset: data/labeled.extxyz

There is no live calculator, so this teacher CANNOT relabel student-visited
pre-crash frames. ``can_relabel = False`` tells the AL loop to run a single
train -> student MD -> report pass and stop (no active learning). ``label()``
asserts that energies/forces are already present and is otherwise a no-op.
"""

from .base import BaseTeacher


class ExtxyzTeacher(BaseTeacher):
    """Teacher backed by a static pre-labeled extxyz dataset (no calculator)."""

    can_relabel = False

    def __init__(self, dataset):
        if not dataset:
            raise ValueError(
                "teacher.dataset must point at a pre-labeled extxyz file when "
                "teacher.type is 'extxyz'."
            )
        self.dataset = dataset

    def label(self, frames):
        for atoms in frames:
            has_e = ("energy" in atoms.info) or (atoms.calc is not None)
            has_f = ("forces" in atoms.arrays)
            assert has_e and has_f, (
                "ExtxyzTeacher cannot compute labels: a frame is missing energy "
                "or forces. The extxyz teacher only ingests already-labeled data."
            )
        return frames
