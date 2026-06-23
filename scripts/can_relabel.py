"""Print '1' if the configured teacher can relabel new frames, else '0'.

Used by the loop scripts to branch: a teacher with no live calculator (extxyz)
cannot relabel student-visited pre-crash frames, so active learning is disabled
and the loop runs a single static train -> MD -> report pass.
"""
from ontheflydistill import config as _cfg
from ontheflydistill.teachers import get_teacher

if __name__ == "__main__":
    teacher = get_teacher(_cfg.load())
    print("1" if getattr(teacher, "can_relabel", False) else "0")
