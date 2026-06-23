#!/bin/bash
###############################################################################
# CROSS-SERVER on-the-fly distillation active-learning loop (OPTIONAL).
#
#   Teacher MD : runs on a remote GPU host (config remote.host)
#   Labeling   : remote SLURM job (scripts/remote/label_slurm.sh.example)
#   Student    : NN-MTP trained + run in LAMMPS LOCALLY
#
#   This is the OPTIONAL remote-teacher path, used when the teacher MLIP is too
#   heavy for the local machine. It is enabled only when remote.enabled is true
#   in config.yaml; otherwise use scripts/al_loop_local.sh.
#
#   PHASE A : assemble the initial dataset (wait for remote teacher MD, rsync
#             trajectories down, merge, rattle-augment + remote-label).
#   PHASE B : AL iterations (train local -> student MD local -> remote-label
#             pre-crash window -> enrich -> retrain) until stable or stalled.
#
#   Teacher capability: an extxyz teacher has no calculator, so AL is disabled
#   and the loop runs a single train -> MD -> report pass (exit 0).
###############################################################################
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY="$(python -c 'from ontheflydistill import config as c; print(c.python_bin())')"
WORK="$(python -c 'import os;from ontheflydistill import config as c; print(os.path.abspath(c.get("work_dir", default="./run")))')"
REMOTE_ENABLED="$(python -c 'from ontheflydistill import config as c; print(1 if c.get("remote","enabled", default=False) else 0)')"
REMOTE_HOST="$(python -c 'from ontheflydistill import config as c; print(c.get("remote","host", default="myserver"))')"
REMOTE_BASE="$(python -c 'from ontheflydistill import config as c; print(c.get("remote","base", default="/remote/path"))')"
TARGET_PS="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","target_ps", default=100000))')"
MAX_ITER="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","max_iter", default=30))')"
NO_PROGRESS_LIMIT="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","no_progress_limit", default=4))')"
export STUDENT_EPOCHS="$(python -c 'from ontheflydistill import config as c; print(c.get("student","epochs", default=300))')"
export OMP_NUM_THREADS="$(python -c 'from ontheflydistill import config as c; print(c.get("al_loop","omp_threads", default=4))')"

LABEL_SLURM="$SCRIPT_DIR/remote/label_slurm.sh.example"
TRAIN_MOD="ontheflydistill.train_student"
STUDENT_MD_MOD="ontheflydistill.student_md_lammps"

mkdir -p "$WORK"
LOG="$WORK/al_loop_remote.log"
STATUS="$WORK/.al_status"
POLL_TEACHER=120
POLL_LABEL=60
RATTLE_STRIDE="${RATTLE_STRIDE:-5}"
RATTLE_PER_FRAME="${RATTLE_PER_FRAME:-2}"
RATTLE_STDEVS="${RATTLE_STDEVS:-0.05,0.1}"

DATASET="$WORK/dataset.extxyz"
TEACHER_DL="$WORK/teacher_dl"
mkdir -p "$TEACHER_DL"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
setstatus() { echo "$*" > "$STATUS"; }
SSH() { ssh "$REMOTE_HOST" "$@" 2>/dev/null; }

if [ "$REMOTE_ENABLED" != "1" ]; then
    log "remote.enabled is false in config.yaml. Use scripts/al_loop_local.sh, "
    log "or set remote.enabled: true (with remote.host/remote.base) to use this path."
    exit 1
fi

# ---- teacher capability gate -----------------------------------------------
CAN_RELABEL="$("$PY" "$SCRIPT_DIR/can_relabel.py" 2>/dev/null || echo 1)"
if [ "$CAN_RELABEL" != "1" ]; then
    log "==== STATIC ONE-SHOT DISTILLATION (teacher cannot relabel) ===="
    [ -s "$DATASET" ] || { log "ERROR: dataset $DATASET not found for extxyz teacher."; setstatus "ERROR no_dataset $(date)"; exit 1; }
    MP="$WORK/model_iter1"; MB="${MP}.bin"; RD="$WORK/al_iter1"
    "$PY" -u -m "$TRAIN_MOD" "$DATASET" "$MP" > "$WORK/train_oneshot.log" 2>&1
    [ -f "$MB" ] || { log "  train FAILED."; setstatus "FAILED train oneshot $(date)"; exit 1; }
    "$PY" -u -m "$STUDENT_MD_MOD" "$MB" "$RD" --total-ps "$TARGET_PS" --seed 42 --omp-threads "$OMP_NUM_THREADS" \
        > "$WORK/md_oneshot.log" 2>&1
    [ -f "$RD/failure.json" ] || { log "  MD FAILED."; setstatus "FAILED md oneshot $(date)"; exit 1; }
    log "AL disabled (extxyz teacher, static one-shot distillation)"
    setstatus "DONE oneshot $(date)"
    exit 0
fi

###############################################################################
# helper: label an xyz on the remote host via SLURM. Submits, polls .DONE, rsyncs.
#   label_remote <local_in.xyz> <local_out.extxyz> <iter_tag>
###############################################################################
label_remote() {
    local lin="$1" lout="$2" tag="$3"
    local rdir="$REMOTE_BASE/label_${tag}"
    local rin="in.xyz" rout="out.extxyz"

    if [ -s "$lout" ]; then
        log "    label[$tag]: $lout already present; skipping remote job."
        return 0
    fi
    if [ ! -s "$lin" ]; then
        log "    label[$tag]: input $lin missing/empty; nothing to label."
        return 1
    fi

    local attempt
    for attempt in 1 2; do
        log "    label[$tag] attempt $attempt: rsync tools+input -> $REMOTE_HOST:$rdir"
        SSH "mkdir -p '$rdir'" || true
        rsync -az "$LABEL_SLURM" "$SCRIPT_DIR/label.py" "$REPO_ROOT/config.yaml" "$lin" \
              "$REMOTE_HOST":"$rdir/" >/dev/null 2>&1
        SSH "cp '$rdir/$(basename "$lin")' '$rdir/$rin'" || true
        SSH "rm -f '$rdir/$rout' '$rdir/$rout.DONE'" || true

        log "    label[$tag] submitting SLURM label job ..."
        SSH "cd '$rdir' && LABEL_IN='$rin' LABEL_OUT='$rout' sbatch $(basename "$LABEL_SLURM")" \
            | tee -a "$LOG"
        sleep 3

        local waited=0 maxwait=$((6*3600))
        while true; do
            if SSH "test -f '$rdir/$rout.DONE'"; then
                log "    label[$tag] DONE detected on remote."
                break
            fi
            local nq
            nq=$(SSH "squeue -u \$USER -h -o '%j' | grep -c label" || echo 0)
            if [ "${nq:-0}" -eq 0 ] && ! SSH "test -f '$rdir/$rout.DONE'"; then
                log "    label[$tag] WARNING: no label job queued and no DONE (attempt $attempt)."
                break
            fi
            sleep "$POLL_LABEL"
            waited=$((waited+POLL_LABEL))
            [ "$waited" -ge "$maxwait" ] && { log "    label[$tag] timeout (attempt $attempt)."; break; }
        done

        if SSH "test -f '$rdir/$rout.DONE'"; then
            rsync -az "$REMOTE_HOST":"$rdir/$rout" "$lout" >/dev/null 2>&1
            [ -s "$lout" ] && { log "    label[$tag] rsynced -> $lout."; return 0; }
        fi
        log "    label[$tag] attempt $attempt failed; retrying once."
    done
    log "    label[$tag] FAILED after retries."
    return 1
}

log "==== CROSS-SERVER on-the-fly AL loop START (MAX_ITER=$MAX_ITER) ===="
log "teacher MD + labeling on $REMOTE_HOST | student=NN-MTP LAMMPS local"
log "student stability target = ${TARGET_PS} ps"

###############################################################################
# PHASE A1 - wait for remote teacher DONE markers, rsync teacher trajectories
###############################################################################
setstatus "PHASE_A1 waiting-for-teacher $(date)"
log "PHASE A1: waiting for teacher DONE markers on $REMOTE_HOST (run0..3) ..."
while true; do
    NDONE=$(SSH "ls $REMOTE_BASE/run0/DONE_TEACHER $REMOTE_BASE/run1/DONE_TEACHER $REMOTE_BASE/run2/DONE_TEACHER $REMOTE_BASE/run3/DONE_TEACHER 2>/dev/null | wc -l" | tr -d ' ')
    NDONE=${NDONE:-0}
    log "  waiting for teacher DONE_TEACHER... ($NDONE/4 present)"
    [ "$NDONE" -ge 4 ] && { log "PHASE A1: all 4 DONE_TEACHER present."; break; }
    sleep "$POLL_TEACHER"
done

for i in 0 1 2 3; do
    dst="$TEACHER_DL/run${i}.extxyz"
    if [ -s "$dst" ]; then
        log "  teacher run$i already downloaded; skipping."
    else
        rsync -az "$REMOTE_HOST":"$REMOTE_BASE/run${i}/teacher_md.extxyz" "$dst" >/dev/null 2>&1
        [ -s "$dst" ] || log "  WARNING: run$i teacher_md.extxyz download empty!"
    fi
done

###############################################################################
# PHASE A2 - merge the teacher trajectories into dataset.extxyz
###############################################################################
if [ -s "$DATASET" ]; then
    log "PHASE A2: $DATASET already exists; skipping initial merge."
else
    setstatus "PHASE_A2 merge-teacher $(date)"
    log "PHASE A2: merging teacher trajectories -> $DATASET"
    "$PY" -u -m ontheflydistill.merge_xyz "$DATASET" \
        "$TEACHER_DL/run0.extxyz" "$TEACHER_DL/run1.extxyz" \
        "$TEACHER_DL/run2.extxyz" "$TEACHER_DL/run3.extxyz" >> "$LOG" 2>&1
fi

###############################################################################
# PHASE A3 - rattle augmentation (geometry only) -> remote-label -> append
###############################################################################
RATTLE_GEOM="$WORK/rattled_geoms.xyz"
RATTLE_LABELED="$WORK/rattled_labeled.extxyz"
RATTLE_MERGED_FLAG="$WORK/.rattle_merged"

if [ -f "$RATTLE_MERGED_FLAG" ]; then
    log "PHASE A3: rattle frames already merged; skipping."
else
    setstatus "PHASE_A3 rattle+label $(date)"
    if [ ! -s "$RATTLE_GEOM" ]; then
        "$PY" -u -m ontheflydistill.rattle_augment "$DATASET" "$RATTLE_GEOM" \
            --per-frame "$RATTLE_PER_FRAME" --stride "$RATTLE_STRIDE" \
            --stdevs "$RATTLE_STDEVS" --seed 42 >> "$LOG" 2>&1
    fi
    if [ -s "$RATTLE_GEOM" ]; then
        if label_remote "$RATTLE_GEOM" "$RATTLE_LABELED" "rattle"; then
            "$PY" -u -m ontheflydistill.merge_xyz "$DATASET" "$DATASET" "$RATTLE_LABELED" >> "$LOG" 2>&1
            touch "$RATTLE_MERGED_FLAG"
        else
            log "PHASE A3: WARNING labeling failed; proceeding with teacher-only dataset."
            touch "$RATTLE_MERGED_FLAG"
        fi
    else
        log "PHASE A3: no rattled geometries generated; skipping."
        touch "$RATTLE_MERGED_FLAG"
    fi
fi

###############################################################################
# PHASE B - AL iterations
###############################################################################
ITER=0
NO_PROGRESS=0
BEST_LASTPS=0
while true; do
    ITER=$((ITER+1))
    log "================== AL ITERATION $ITER (target ${TARGET_PS} ps; backstop ${MAX_ITER}; no-progress ${NO_PROGRESS}/${NO_PROGRESS_LIMIT}) =================="
    MODEL_PREFIX="$WORK/model_iter${ITER}"
    MODEL_BIN="${MODEL_PREFIX}.bin"
    RUN_DIR="$WORK/al_iter${ITER}"
    PRE_CRASH="$RUN_DIR/pre_crash.xyz"
    FAIL_JSON="$RUN_DIR/failure.json"
    LABELED="$WORK/al_iter${ITER}_labeled.extxyz"

    if [ -f "$MODEL_BIN" ]; then
        log "  b.a (iter $ITER): $MODEL_BIN exists; skipping training."
    else
        setstatus "iter$ITER b.a training $(date)"
        "$PY" -u -m "$TRAIN_MOD" "$DATASET" "$MODEL_PREFIX" >> "$WORK/train_iter${ITER}.log" 2>&1
        [ -f "$MODEL_BIN" ] || { log "  b.a FAILED. See train_iter${ITER}.log. ABORT."; setstatus "FAILED b.a iter$ITER $(date)"; exit 1; }
        log "  b.a done -> $MODEL_BIN"
    fi

    if [ -f "$FAIL_JSON" ]; then
        log "  b.b/c (iter $ITER): $FAIL_JSON exists; skipping student MD."
    else
        setstatus "iter$ITER b.b student-MD $(date)"
        "$PY" -u -m "$STUDENT_MD_MOD" "$MODEL_BIN" "$RUN_DIR" \
            --total-ps "$TARGET_PS" --seed 42 --omp-threads "$OMP_NUM_THREADS" \
            >> "$WORK/student_md_iter${ITER}.log" 2>&1
        [ -f "$FAIL_JSON" ] || { log "  b.b/c FAILED. See student_md_iter${ITER}.log. ABORT."; setstatus "FAILED b.b iter$ITER $(date)"; exit 1; }
    fi

    STAT=$("$PY" -c "import json;print(json.load(open('$FAIL_JSON'))['status'])" 2>/dev/null)
    LASTPS=$("$PY" -c "import json;print('%.3f'%float(json.load(open('$FAIL_JSON')).get('last_ps',0) or 0))" 2>/dev/null)
    log "  b.c (iter $ITER): status=$STAT crash@${LASTPS}ps"

    if [ "$STAT" = "stable_in_dump" ]; then
        log "  *** SUCCESS: student MD stable to ${TARGET_PS} ps (iter $ITER). STOPPING. ***"
        setstatus "SUCCESS iter$ITER stable $(date)"; exit 0
    fi

    IMPROVED=$("$PY" -c "print(1 if float('$LASTPS') > float('$BEST_LASTPS')*1.05 else 0)" 2>/dev/null)
    if [ "$IMPROVED" = "1" ]; then
        log "  progress: crash time ${LASTPS} ps > 1.05 x prev best ${BEST_LASTPS} ps."
        BEST_LASTPS="$LASTPS"; NO_PROGRESS=0
    else
        NO_PROGRESS=$((NO_PROGRESS+1))
        log "  no-progress: crash ${LASTPS} ps <= 1.05 x best ${BEST_LASTPS} ps (${NO_PROGRESS}/${NO_PROGRESS_LIMIT})."
        if [ "$NO_PROGRESS" -ge "$NO_PROGRESS_LIMIT" ]; then
            log "  *** AL STALLED: no improvement for ${NO_PROGRESS} iters. STOPPING. ***"
            setstatus "STOPPED no_progress iter$ITER best=${BEST_LASTPS}ps $(date)"; exit 0
        fi
    fi

    if [ "$ITER" -ge "$MAX_ITER" ]; then
        log "  reached backstop MAX_ITER=$MAX_ITER (last status=$STAT). STOPPING."
        setstatus "STOPPED backstop_max_iter=$MAX_ITER $(date)"; exit 0
    fi

    if [ ! -s "$PRE_CRASH" ]; then
        log "  b.d (iter $ITER): no pre-crash frames captured. STOPPING."
        setstatus "STOPPED no_precrash iter$ITER $(date)"; exit 0
    fi

    setstatus "iter$ITER b.d label-precrash $(date)"
    if label_remote "$PRE_CRASH" "$LABELED" "iter${ITER}"; then
        log "  iter $ITER labeled pre-crash frames -> $LABELED (picked up by (c) glob); retraining iter $((ITER+1))."
    else
        log "  b.d WARNING: labeling failed; STOPPING."
        setstatus "STOPPED label_failed iter$ITER $(date)"; exit 0
    fi
done
