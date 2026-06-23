#!/bin/bash
###############################################################################
# Autonomous active-learning loop (LOCAL, portable DEFAULT).
#
#  Each AL iteration trains the NN-MTP student FROM SCRATCH (full epochs) on
#  base(dataset.extxyz) + ALL al_iter*_labeled.extxyz (the (c) glob, uncut),
#  runs the student in LAMMPS, and on failure relabels the pre-crash window with
#  the configured teacher and adds it to the pool.
#
#  Each round:
#    1. train scratch -> model_scratchR.bin
#    2. student LAMMPS MD seed SEED, target = al_loop.target_ps
#    3. stable to target -> SUCCESS, stop
#       crash@T          -> relabel pre_crash with the teacher -> al_iter{K}_labeled
#    4. no crash-time improvement for NOPROG_LIMIT rounds -> STALLED, stop
#
#  Teacher capability: if the teacher cannot relabel (extxyz / no calculator),
#  the loop runs train -> MD -> report ONCE and exits 0 (static one-shot
#  distillation; no active learning).
#
#  Everything is config-driven (config.yaml). Labeling is LOCAL by default;
#  remote SLURM labeling is only used when remote.enabled is true.
###############################################################################
set -u

# ---- resolve repo root + config-driven values ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY="$(python -c 'from ontheflydistill import config as c; print(c.python_bin())')"
WORK="$(python -c 'import os;from ontheflydistill import config as c; print(os.path.abspath(c.get("work_dir", default="./run")))')"
TARGET_PS="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","target_ps", default=100000))')"
NOPROG_LIMIT="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","no_progress_limit", default=4))')"
MAXK="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","max_iter", default=30))')"
SEED="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","seed", default=42))')"
export OMP_NUM_THREADS="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","omp_threads", default=8))')"
export STUDENT_WARMSTART=0          # SCRATCH each iter (the validated recipe)

LABEL_SCRIPT="$SCRIPT_DIR/label.py"
TEACHER_MD_SCRIPT="$SCRIPT_DIR/teacher_md.py"
TRAIN_MOD="ontheflydistill.train_student"
STUDENT_MD_MOD="ontheflydistill.student_md_lammps"

mkdir -p "$WORK"
LOG="$WORK/al_loop_local.log"
STATUS="$WORK/.al_status"
DATASET="$WORK/dataset.extxyz"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
setstatus(){ echo "$*" > "$STATUS"; }

# --- label pre_crash.xyz -> labeled.extxyz with the configured teacher ---
label_frames(){
    local lin="$1" lout="$2" tag="$3"
    [ -s "$lout" ] && { log "  label[$tag]: $lout exists; skip."; return 0; }
    [ -s "$lin" ] || { log "  label[$tag]: input $lin missing/empty."; return 1; }
    log "  label[$tag]: relabeling with configured teacher ..."
    "$PY" -u "$LABEL_SCRIPT" "$lin" "$lout" >> "$WORK/label.log" 2>&1
    [ -s "$lout" ] && { log "  label[$tag]: -> $lout OK."; return 0; }
    log "  label[$tag]: labeling FAILED (see label.log)."; return 1
}

# first missing al_iter{k}_labeled index
nextK(){ local k=1; while [ -f "$WORK/al_iter${k}_labeled.extxyz" ]; do k=$((k+1)); done; echo "$k"; }
crashps(){ "$PY" -c "import json;d=json.load(open('$1/failure.json'));print('%.1f'%(float(d.get('last_ps',0) or 0)))" 2>/dev/null || echo 0; }
statusof(){ "$PY" -c "import json;print(json.load(open('$1/failure.json'))['status'])" 2>/dev/null; }

# ---- teacher capability gate -----------------------------------------------
CAN_RELABEL="$("$PY" "$SCRIPT_DIR/can_relabel.py" 2>/dev/null || echo 1)"

if [ ! -s "$DATASET" ]; then
    log "ERROR: dataset $DATASET not found. Build it first (teacher_md.py -> merge), "
    log "       or for an extxyz teacher copy your labeled set to $DATASET."
    setstatus "ERROR no_dataset $(date)"; exit 1
fi

if [ "$CAN_RELABEL" != "1" ]; then
    log "==== STATIC ONE-SHOT DISTILLATION (teacher cannot relabel) ===="
    MP="$WORK/model_scratch1"; MB="${MP}.bin"; RD="$WORK/run_scratch1"
    setstatus "oneshot training $(date)"
    "$PY" -u -m "$TRAIN_MOD" "$DATASET" "$MP" > "$WORK/train_oneshot.log" 2>&1
    [ -f "$MB" ] || { log "  train FAILED (see train_oneshot.log)."; setstatus "FAILED train oneshot $(date)"; exit 1; }
    setstatus "oneshot student-MD $(date)"
    "$PY" -u -m "$STUDENT_MD_MOD" "$MB" "$RD" --total-ps "$TARGET_PS" --seed "$SEED" --omp-threads "$OMP_NUM_THREADS" \
        > "$WORK/md_oneshot.log" 2>&1
    [ -f "$RD/failure.json" ] || { log "  MD FAILED (see md_oneshot.log)."; setstatus "FAILED md oneshot $(date)"; exit 1; }
    ST=$(statusof "$RD"); PS=$(crashps "$RD")
    log "  one-shot result: status=$ST crash@${PS}ps"
    log "AL disabled (extxyz teacher, static one-shot distillation)"
    setstatus "DONE oneshot status=$ST crash=${PS}ps $(date)"
    exit 0
fi

log "==== AUTONOMOUS AL LOOP START (scratch each iter, target=${TARGET_PS} ps) ===="
log "AL pool now: $(ls "$WORK"/al_iter*_labeled.extxyz 2>/dev/null | tr '\n' ' ')"

BEST_PS=0; NOPROG=0

# ---------------- ROUND 0: initial student from the base dataset --------------
log "ROUND 0: train initial student (base dataset + any existing AL pool) ..."
MP0="$WORK/model_scratch0"; MB0="${MP0}.bin"; RD0="$WORK/run_scratch0"
if [ ! -f "$MB0" ]; then
    setstatus "round0 training $(date)"
    "$PY" -u -m "$TRAIN_MOD" "$DATASET" "$MP0" > "$WORK/train_scratch0.log" 2>&1
    [ -f "$MB0" ] || { log "ROUND 0: train FAILED (see train_scratch0.log)."; setstatus "FAILED train round0 $(date)"; exit 1; }
fi
if [ ! -f "$RD0/failure.json" ]; then
    setstatus "round0 student-MD $(date)"
    "$PY" -u -m "$STUDENT_MD_MOD" "$MB0" "$RD0" --total-ps "$TARGET_PS" --seed "$SEED" --omp-threads "$OMP_NUM_THREADS" \
        > "$WORK/md_scratch0.log" 2>&1
    [ -f "$RD0/failure.json" ] || { log "ROUND 0: MD FAILED (see md_scratch0.log)."; setstatus "FAILED md round0 $(date)"; exit 1; }
fi
S0=$(statusof "$RD0"); P0=$(crashps "$RD0")
log "ROUND 0: status=$S0 crash@${P0}ps"
if [ "$S0" = "stable_in_dump" ]; then
    log "*** SUCCESS at ROUND 0: student stable to target. STOPPING. ***"
    setstatus "SUCCESS round0 stable $(date)"; exit 0
fi
BEST_PS="$P0"
K=$(nextK)
if label_frames "$RD0/pre_crash.xyz" "$WORK/al_iter${K}_labeled.extxyz" "scr0"; then
    log "ROUND 0: failures -> al_iter${K}_labeled (pool grows)."
else
    log "ROUND 0: labeling failed; STOPPING."; setstatus "STOPPED label_fail round0 $(date)"; exit 1
fi

# ---------------- FORWARD ROUNDS ---------------------------------------------
R=0
while [ "$R" -lt "$MAXK" ]; do
    R=$((R+1))
    MP="$WORK/model_scratch${R}"; MB="${MP}.bin"
    RD="$WORK/run_scratch${R}"
    log "================== SCRATCH ROUND $R (best ${BEST_PS} ps; no-progress ${NOPROG}/${NOPROG_LIMIT}) =================="

    if [ -f "$MB" ]; then log "  train: $MB exists; skip."; else
        log "  train scratch on base + AL pool -> $MB"
        setstatus "round$R training $(date)"
        "$PY" -u -m "$TRAIN_MOD" "$DATASET" "$MP" > "$WORK/train_scratch${R}.log" 2>&1
        [ -f "$MB" ] || { log "  train FAILED (see train_scratch${R}.log). STOPPING."; setstatus "FAILED train round$R $(date)"; exit 1; }
    fi

    if [ -f "$RD/failure.json" ]; then log "  MD: $RD/failure.json exists; skip."; else
        log "  student MD seed $SEED, target ${TARGET_PS} ps"
        setstatus "round$R student-MD $(date)"
        "$PY" -u -m "$STUDENT_MD_MOD" "$MB" "$RD" --total-ps "$TARGET_PS" --seed "$SEED" --omp-threads "$OMP_NUM_THREADS" \
            > "$WORK/md_scratch${R}.log" 2>&1
        [ -f "$RD/failure.json" ] || { log "  MD FAILED (no failure.json, see md_scratch${R}.log). STOPPING."; setstatus "FAILED md round$R $(date)"; exit 1; }
    fi

    ST=$(statusof "$RD"); PS=$(crashps "$RD")
    log "  ROUND $R: status=$ST crash@${PS}ps (best so far ${BEST_PS})"

    if [ "$ST" = "stable_in_dump" ]; then
        log "*** SUCCESS at ROUND $R: student stable to target. STOPPING. ***"
        setstatus "SUCCESS round$R stable $(date)"; exit 0
    fi

    IMP=$("$PY" -c "print(1 if float('$PS')>float('$BEST_PS')*1.05 else 0)" 2>/dev/null)
    if [ "$IMP" = "1" ]; then
        log "  progress: ${PS} ps > 1.05 x best ${BEST_PS} ps."
        BEST_PS="$PS"; NOPROG=0
    else
        NOPROG=$((NOPROG+1))
        log "  no-progress: ${PS} ps <= 1.05 x best ${BEST_PS} (${NOPROG}/${NOPROG_LIMIT})."
        if [ "$NOPROG" -ge "$NOPROG_LIMIT" ]; then
            log "*** STALLED: no improvement for ${NOPROG} rounds (best ${BEST_PS} ps). STOPPING. ***"
            setstatus "STOPPED no_progress round$R best=${BEST_PS}ps $(date)"; exit 0
        fi
    fi

    K=$(nextK)
    if label_frames "$RD/pre_crash.xyz" "$WORK/al_iter${K}_labeled.extxyz" "scr${R}"; then
        log "  ROUND $R: failures -> al_iter${K}_labeled; retraining next round."
    else
        log "  ROUND $R: labeling failed; STOPPING."; setstatus "STOPPED label_fail round$R $(date)"; exit 1
    fi
done

log "==== reached MAXK=$MAXK backstop; STOPPING. ===="
setstatus "STOPPED backstop $(date)"
exit 0
