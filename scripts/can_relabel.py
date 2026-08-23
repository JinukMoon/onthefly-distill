"""Print '1' if the configured teacher can relabel new frames, else '0'.

Used by the loop scripts to branch: a teacher with no live calculator (extxyz)
cannot relabel student-visited pre-crash frames, so active learning is disabled
and the loop runs a single static train -> MD -> report pass.
"""
import contextlib
import os
import sys

from ontheflydistill import config as _cfg
from ontheflydistill.teachers import get_teacher

if __name__ == "__main__":
    # The caller captures our WHOLE stdout and compares it literally to "1",
    # but building the teacher may import/construct a chatty calculator (e.g.
    # mace_mp prints banners). Route everything the construction emits to
    # stderr and keep the real stdout for the one-character answer.
    real_stdout = os.dup(1)
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            teacher = get_teacher(_cfg.load())
    finally:
        os.dup2(real_stdout, 1)
        os.close(real_stdout)
    print("1" if getattr(teacher, "can_relabel", False) else "0")
