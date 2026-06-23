"""UMA / FAIRChem teacher — EXAMPLE adapter (optional dependency).

This is the teacher the Pt(111)-water example uses. ``fairchem-core`` is an
OPTIONAL dependency: it is imported INSIDE the class, so importing this module
(or the whole ``ontheflydistill`` package) never triggers fairchem / CUDA init.

config.yaml::

    teacher:
      type: uma
      model: uma-s-1p2      # consumed by BOTH teacher_md.py and label.py
      task: oc25            # MUST match between teacher MD and labeling
      device: cuda

The labeler's model+task MUST equal the teacher-MD model+task so that labels and
the teacher trajectory live on the same potential-energy surface.
"""

from .ase_calculator import ASECalculatorTeacher


def _build_fairchem_calc(model, task, device):
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The UMA teacher requires fairchem-core. Install it with "
            "`pip install fairchem-core`, or use teacher.type: ase_calculator "
            "with your own calculator instead."
        ) from exc
    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name=task)


class UMATeacher(ASECalculatorTeacher):
    """FAIRChem/UMA teacher. Builds a FAIRChemCalculator lazily (fairchem optional)."""

    can_relabel = True

    def __init__(self, model="uma-s-1p2", task="oc25", device="cuda", force_pbc=True):
        self.model = model
        self.task = task
        self.device = device
        self.force_pbc = force_pbc
        self.calc = _build_fairchem_calc(model, task, device)
