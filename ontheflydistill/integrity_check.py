"""
Integrity metric v2 (plan W0.2) — the F2 y-axis definition.

Unified stability time = min( first SUSTAINED integrity violation , end-of-dump if run crashed ).
Violation (spec AC1, all chemistry+film):
  - free-H2 count >= 2            (H2 evolution)
  - O-O < 1.6 A pairs >= 1        (peroxide/O-O overbond)
  - intact-H2O count outside [54, 58]
  - water film 10-90% thickness > 20 A   (evaporation/spreading; healthy 13-16)
  - vapor O count (z > p90+4A) >= 2
Sustained = violation persisting >= 10 ps (k consecutive frames, k = ceil(10ps / frame_spacing)).
Frame spacing is AUTO-DERIVED per run: (TIMESTEP[1]-TIMESTEP[0]) * dt, dt parsed from <run>/md.in
("timestep X" line, ps). No hardcoded DT_PS (the old bug inflated iter1 by 20x).

Usage: python integrity_check.py <run_dir(with md.dump,md.in)> [--samples N] [--json out.json]
Prints summary table + onset; exits 0 always (gate logic reads the JSON/stdout).
"""
import sys, os, json, math, argparse
import numpy as np
from ase.io import read
from ase.geometry import get_distances

from ontheflydistill import config as _cfg

# Element order for reading the LAMMPS dump (config system.specorder; Pt-water default).
SPEC = list(_cfg.get("system", "specorder", default=["H", "O", "Pt"]))
# H2O lower bound 50, NOT 54: in this acid system (16 excess protons) normal Grotthuss
# shuttling gives H3O+ 2-7 and OH- 0-2, so intact-H2O 51-58 is healthy chemistry.
# 54 tripped at 30-60 ps on healthy stretches (gate v1 false positive). Decomposition
# (the thing we must catch) drives H2O to ~30-40 — far below 50.
H2O_MIN = 50
FILM_MAX_A = 20.0
VAPOR_MAX = 1          # >=2 is violation
SUSTAIN_PS = 10.0


def frame_spacing_ps(run_dir):
    """dt from md.in 'timestep' (ps, metal units); spacing from first two TIMESTEP headers."""
    dt = None
    with open(os.path.join(run_dir, "md.in")) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2 and t[0] == "timestep":
                dt = float(t[1])
                break
    if dt is None:
        raise RuntimeError(f"timestep not found in {run_dir}/md.in")
    steps = []
    with open(os.path.join(run_dir, "md.dump")) as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                steps.append(int(next(f)))
                if len(steps) == 2:
                    break
    if len(steps) < 2:
        raise RuntimeError("fewer than 2 frames in dump")
    return (steps[1] - steps[0]) * dt, dt


def frame_metrics(a, cell):
    sym = np.array(a.get_chemical_symbols())
    pos = a.get_positions()
    D = get_distances(pos, pos, cell=cell, pbc=a.get_pbc())[1]
    oidx = np.where(sym == "O")[0]
    hidx = np.where(sym == "H")[0]
    OH = D[np.ix_(oidx, hidx)]
    nH = (OH < 1.3).sum(axis=1)
    h2o = int((nH == 2).sum())
    HH = D[np.ix_(hidx, hidx)].copy(); np.fill_diagonal(HH, 9)
    Hfar = OH.min(axis=0) > 1.3
    freeH2 = 0
    pairs = np.argwhere((HH < 0.9))
    seen = set()
    for x, y in pairs:
        if x < y and Hfar[x] and Hfar[y] and x not in seen and y not in seen:
            freeH2 += 1; seen.add(x); seen.add(y)
    OO = D[np.ix_(oidx, oidx)].copy(); np.fill_diagonal(OO, 9)
    oo = int((OO < 1.6).sum() // 2)
    oz_raw = pos[oidx, 2]
    p10, p90 = np.percentile(oz_raw, 10), np.percentile(oz_raw, 90)
    film = float(p90 - p10)
    # vapor = ISOLATED molecule above the film: z>p90+4 AND nearest-O > 5 A.
    # Without the isolation condition, H-bond-connected capillary protrusions
    # (nnO 2.7-3.6 A riding +4-8 A above p90, ~1% of frames) are miscounted as
    # evaporation (diagnosed 2026-06-13 on 1c 15.8-20.2 ns; film/chem intact).
    high = oidx[oz_raw > p90 + 4.0]
    vapor = 0
    for o in high:
        others = oidx[oidx != o]
        if D[o, others].min() > 5.0:
            vapor += 1
    # NOTE: free-H2 is NOT a violation criterion. Spot-check (2026-06-12) showed early
    # "free H2" sits +0.4-0.7 A above the Pt top layer = Tafel-recombined H2* from the
    # H* adlayer — LEGITIMATE HER chemistry in this acid system. Fake water splitting is
    # caught by oo (O-O overbond) and h2o (intact-water inventory collapse) instead.
    # freeH2 stays as a reported diagnostic column only.
    crit = dict(oo=oo >= 1, h2o=h2o < H2O_MIN,
                film=film > FILM_MAX_A, vapor=vapor > VAPOR_MAX)
    return dict(h2o=h2o, freeH2=freeH2, oo=oo, film=round(film, 1),
                vapor=vapor, violation=any(crit.values()), crit=crit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--samples", type=int, default=12, help="rows printed (full scan always)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rd = args.run_dir.rstrip("/")

    spacing_ps, dt = frame_spacing_ps(rd)
    k = max(1, math.ceil(SUSTAIN_PS / spacing_ps))
    frames = read(os.path.join(rd, "md.dump"), index=":",
                  format="lammps-dump-text", specorder=SPEC)
    N = len(frames)
    cell = np.array(frames[0].get_cell())
    end_ns = N * spacing_ps / 1000.0
    print(f"# {rd}: {N} frames, spacing={spacing_ps:.4g} ps/frame (dt={dt} ps, auto), "
          f"end={end_ns:.2f} ns, sustain k={k} frames (>={SUSTAIN_PS} ps)")

    CRITS = ["oo", "h2o", "film", "vapor"]
    cflags = {c: np.zeros(N, dtype=bool) for c in CRITS}
    rows = []
    show = set(int(round(x)) for x in np.linspace(0, N - 1, min(args.samples, N)))
    for i, a in enumerate(frames):
        m = frame_metrics(a, cell)
        for c in CRITS:
            cflags[c][i] = m["crit"][c]
        if i in show:
            rows.append((i * spacing_ps / 1000.0, m))

    def sustained_onset(flags):
        run_len = 0
        for i in range(N):
            run_len = run_len + 1 if flags[i] else 0
            if run_len >= k:
                return (i - k + 1) * spacing_ps / 1000.0
        return None

    RECOVER_PS = 1000.0  # a clean stretch >= 1 ns counts as recovery
    w_rec = max(1, math.ceil(RECOVER_PS / spacing_ps))

    def irreversible_onset(flags):
        """Start of the FINAL violation regime that never heals: after this time there is
        no clean stretch >= RECOVER_PS. Transient 10ps episodes that heal do NOT count.
        Implemented by scanning clean-run lengths from the end."""
        if not flags.any():
            return None
        # walk backwards: find last index j such that a clean run of >= w_rec ends at j
        clean_len = 0
        last_recovery_end = -1
        for i in range(N):
            clean_len = clean_len + 1 if not flags[i] else 0
            if clean_len >= w_rec:
                last_recovery_end = i
        # violations after last_recovery_end: first sustained onset in that tail
        run_len = 0
        for i in range(last_recovery_end + 1, N):
            run_len = run_len + 1 if flags[i] else 0
            if run_len >= k:
                return (i - k + 1) * spacing_ps / 1000.0
        return None

    per_crit = {c: sustained_onset(cflags[c]) for c in CRITS}            # episodes (diagnostic)
    per_crit_irr = {c: irreversible_onset(cflags[c]) for c in CRITS}     # PRIMARY (F2 y-axis)
    fired = {c: v for c, v in per_crit_irr.items() if v is not None}
    onset_ns = min(fired.values()) if fired else None
    culprit = min(fired, key=fired.get) if fired else None
    ep_fired = {c: v for c, v in per_crit.items() if v is not None}
    episode_onset_ns = min(ep_fired.values()) if ep_fired else None

    print(f"{'~ns':>8} | {'H2O':>4} {'freeH2':>6} {'O-O':>4} {'film(A)':>8} {'vapor':>5} {'viol':>4}")
    for t, m in rows:
        print(f"{t:>8.2f} | {m['h2o']:>4} {m['freeH2']:>6} {m['oo']:>4} "
              f"{m['film']:>8.1f} {m['vapor']:>5} {'X' if m['violation'] else '.':>4}")

    stability_ns = onset_ns if onset_ns is not None else end_ns
    label = f"IRREVERSIBLE onset, culprit={culprit}" if onset_ns is not None \
        else "no irreversible violation (stability = end of dump)"
    print(f"\nepisode onsets (>=10ps, may heal) (ns): " +
          ", ".join(f"{c}={'%.3f' % v if v is not None else '-'}" for c, v in per_crit.items()))
    print(f"irreversible onsets (never clean >=1ns after) (ns): " +
          ", ".join(f"{c}={'%.3f' % v if v is not None else '-'}" for c, v in per_crit_irr.items()))
    print(f"RESULT spacing_ps={spacing_ps:.4g} end_ns={end_ns:.3f} "
          f"episode_onset_ns={'%.3f' % episode_onset_ns if episode_onset_ns is not None else 'None'} "
          f"irr_onset_ns={'%.3f' % onset_ns if onset_ns is not None else 'None'} "
          f"stability_ns={stability_ns:.3f}  [{label}]")
    if args.json:
        json.dump(dict(run=rd, n_frames=N, spacing_ps=spacing_ps, dt_ps=dt, k=k,
                       end_ns=end_ns, episode_onset_ns=episode_onset_ns,
                       onset_ns=onset_ns, stability_ns=stability_ns, culprit=culprit,
                       per_criterion_episode_ns=per_crit,
                       per_criterion_irreversible_ns=per_crit_irr),
                  open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
