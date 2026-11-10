#!/usr/bin/env bash
# RQ0: shared reference (RQ0-01, ResNet34 + ImageNet + balanced softmax, aug base) and the architecture check
# (RQ0-02, ConvNeXt-Tiny on the same recipe). Rules: rules/rq0_prespecification.md (committed before the first run).
# 2 arms x 6 designs (100% labels x seeds 42/1337/2026, 10% labels x draws 0/1/2 at seed 42) = 12 runs.
# The order alternates arms, so a partial queue still compares like with like.
# Finished runs (metrics.json present) are skipped and an interrupted run restarts, so after a shutdown rerun the call.
# The queue writes its session id to <OUT>/queue.pid; stop it with kill -TERM -- -$(cat runs/rq0/queue.pid).
# Usage (from the repo root or anywhere):
#   bash scripts/run_rq0.sh              status table (runs done, rough hours left)
#   setsid nohup bash scripts/run_rq0.sh all >> runs/rq0.log 2>&1 < /dev/null &
# Smoke test: OUT=/tmp/rq0 bash scripts/run_rq0.sh all --limit 640 --epochs 1 --min-steps 0
# CONVNEXT selects the timm weights of RQ0-02 (see the rules); MINUTES are rough (ResNet34 from E3, ConvNeXt from 1 epoch).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/code"
STAGE="${1:-}"
shift $(( $# > 0 ))
EXTRA=("$@")
export TQDM_DISABLE=1

OUT="${OUT:-${ROOT}/runs/rq0}"
CONVNEXT="${CONVNEXT:-convnext_tiny.fb_in1k}"
ARMS=(rq0-01 rq0-02)
declare -A MODEL=([rq0-01]=resnet34 [rq0-02]="${CONVNEXT}")
declare -A TAGNAME=([rq0-01]=resnet34 [rq0-02]=convnext_tiny)
declare -A MINUTES=([rq0-01_0.1]=7 [rq0-01_1.0]=31 [rq0-02_0.1]=13 [rq0-02_1.0]=60)   # ConvNeXt: 1-epoch timing 2026-11-11, ~1.9x ResNet34

PLAN=""
train() {   # arm, fraction, draw, seed; a failure is reported and the queue goes on
    local arm=$1 frac=$2 draw=$3 seed=$4
    local name="${arm}_${TAGNAME[${arm}]}_b2_f${frac}_d${draw}_s${seed}"
    if [[ -n "${PLAN}" ]]; then
        echo "${arm} $([[ -f "${OUT}/${name}/metrics.json" ]] && echo 1 || echo 0) ${MINUTES[${arm}_${frac}]}"
        return 0
    fi
    [[ -f "${OUT}/${name}/metrics.json" ]] && return 0
    rm -rf "${OUT:?}/${name}"
    echo "$(date '+%F %T') [$(compgen -G "${OUT}/rq0-*/metrics.json" | wc -l) done] ${name}"
    local t0=${SECONDS}
    if ! uv run python -m ramscore.train --name "${name}" --out "${OUT}" --amp --n-boot 0 --model "${MODEL[${arm}]}" \
        --pretrained --head softmax --logit-adjust 1 --label-fraction "${frac}" --fraction-seed "${draw}" --seed "${seed}" \
        --epochs 50 --min-steps 2000 --aug base --seed-aug "${EXTRA[@]}" > "${OUT}/logs/${name}.log" 2>&1; then
        echo "FAILED ${name} (see ${OUT}/logs/${name}.log); rerun the script to retry" >&2
        return 1
    fi
    echo "$(date '+%F %T') finished ${name} in $(( (SECONDS - t0) / 60 )) min"
}

queue() {
    local draw seed arm
    for draw in 0 1 2; do for arm in "${ARMS[@]}"; do train "${arm}" 0.1 "${draw}" 42; done; done
    for seed in 42 1337 2026; do for arm in "${ARMS[@]}"; do train "${arm}" 1.0 0 "${seed}"; done; done
}

if [[ -z "${STAGE}" ]]; then
    PLAN=1
    echo "OUT=${OUT}  RQ0-02 weights=${CONVNEXT}  (hours are rough)"
    queue | awk '{ n[$1]++; d[$1] += $2; if (!$2) m[$1] += $3 } END { printf "%-10s %10s %10s\n", "arm", "runs done", "hours left"; for (a in n) printf "%-10s %5d/%-4d %10.1f\n", a, d[a], n[a], m[a] / 60 }'
    exit 0
fi
[[ "${STAGE}" == all ]] || { echo "unknown stage ${STAGE}; use: all" >&2; exit 1; }
mkdir -p "${OUT}/logs"
ps -o sid= -p $$ | tr -d " " > "${OUT}/queue.pid"
echo "$(date '+%F %T') RQ0 started (pid $$, RQ0-02 weights ${CONVNEXT})"
queue
echo "$(date '+%F %T') RQ0 finished: $(compgen -G "${OUT}/rq0-*/metrics.json" | wc -l) runs with metrics.json"
