#!/usr/bin/env bash
# RQ1 stage 1: the arms that need only existing train.py flags (rules/rq1_prespecification.md, committed 4fccaab).
#   rq1-01 ResNet34 scratch            rq1-04 ViT-S ImageNet-1k (augreg_in1k) + LLRD
#   rq1-02 ResNet34 DINO (E1)          rq1-05 ViT-S MAE (E1) + LLRD
#   rq1-03 ResNet34 DINO+ChronoCon (E1) rq1-06 ViT-S DINO (E1) + LLRD
# Head B2 (softmax + logit adjustment 1), aug base seeded, val QWK monitor; ResNet34 = RQ0-01 recipe, ViT-S adds
# layer decay 0.85, lr 1e-4, warm-up 10% (E6 stage-1 choice on validation). 6 arms x 6 designs = 36 runs.
# The order alternates arms, so a partial queue still compares like with like.
# Finished runs (metrics.json present) are skipped and an interrupted run restarts, so after a shutdown rerun the call.
# The queue checks the encoder sha256 sums first and writes its session id to <OUT>/queue.pid;
# stop it with kill -TERM -- -$(cat runs/rq1/queue.pid).
# Usage: bash scripts/run_rq1.sh            status table (runs done, rough hours left)
#        setsid nohup bash scripts/run_rq1.sh all >> runs/rq1.log 2>&1 < /dev/null &
# Smoke test: OUT=/tmp/rq1 bash scripts/run_rq1.sh all --limit 640 --epochs 1 --min-steps 0
# MINUTES are rough: ResNet34 from E3, ViT-S from E2 medians.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/code"
STAGE="${1:-}"
shift $(( $# > 0 ))
EXTRA=("$@")
export TQDM_DISABLE=1

OUT="${OUT:-${ROOT}/runs/rq1}"
ENC="${ROOT}/artifacts/encoders"
VIT="--layer-decay 0.85 --lr 1e-4 --warmup 0.1"
ARMS=(rq1-01 rq1-02 rq1-03 rq1-04 rq1-05 rq1-06)
declare -A TAGNAME=([rq1-01]=resnet34_scratch [rq1-02]=resnet34_dino [rq1-03]=resnet34_dinochrono
                    [rq1-04]=vits_imagenet [rq1-05]=vits_mae [rq1-06]=vits_dino)
declare -A FLAGS=(
    [rq1-01]="--model resnet34"
    [rq1-02]="--model resnet34 --init-from ${ENC}/e1_dino_resnet34/encoder.pt"
    [rq1-03]="--model resnet34 --init-from ${ENC}/e1_dinochrono_resnet34/encoder.pt"
    [rq1-04]="--model vit_small_patch16_224.augreg_in1k --pretrained ${VIT}"
    [rq1-05]="--model vit_small_patch16_224 --init-from ${ENC}/e1_mae_vits/encoder.pt ${VIT}"
    [rq1-06]="--model vit_small_patch16_224 --init-from ${ENC}/e1_dino_vits/encoder.pt ${VIT}"
)
declare -A MINUTES=([resnet34_0.1]=7 [resnet34_1.0]=31 [vits_0.1]=10 [vits_1.0]=40)

PLAN=""
train() {   # arm, fraction, draw, seed; a failure is reported and the queue goes on
    local arm=$1 frac=$2 draw=$3 seed=$4
    local name="${arm}_${TAGNAME[${arm}]}_b2_f${frac}_d${draw}_s${seed}"
    if [[ -n "${PLAN}" ]]; then
        echo "${arm} $([[ -f "${OUT}/${name}/metrics.json" ]] && echo 1 || echo 0) ${MINUTES[${TAGNAME[${arm}]%%_*}_${frac}]}"
        return 0
    fi
    [[ -f "${OUT}/${name}/metrics.json" ]] && return 0
    rm -rf "${OUT:?}/${name}"
    echo "$(date '+%F %T') [$(compgen -G "${OUT}/rq1-*/metrics.json" | wc -l) done] ${name}"
    local t0=${SECONDS}
    # shellcheck disable=SC2086  # FLAGS holds several words on purpose
    if ! uv run python -m ramscore.train --name "${name}" --out "${OUT}" --amp --n-boot 0 ${FLAGS[${arm}]} \
        --head softmax --logit-adjust 1 --label-fraction "${frac}" --fraction-seed "${draw}" --seed "${seed}" \
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
    echo "OUT=${OUT}  (hours are rough)"
    queue | awk '{ n[$1]++; d[$1] += $2; if (!$2) m[$1] += $3 } END { printf "%-10s %10s %10s\n", "arm", "runs done", "hours left"; for (a in n) printf "%-10s %5d/%-4d %10.1f\n", a, d[a], n[a], m[a] / 60 }' | sort
    exit 0
fi
[[ "${STAGE}" == all ]] || { echo "unknown stage ${STAGE}; use: all" >&2; exit 1; }
(cd "${ROOT}/artifacts" && sha256sum --quiet -c SHA256SUMS) || { echo "encoder checksums failed; queue not started" >&2; exit 1; }
mkdir -p "${OUT}/logs"
ps -o sid= -p $$ | tr -d " " > "${OUT}/queue.pid"
echo "$(date '+%F %T') RQ1 stage 1 started (pid $$)"
queue
echo "$(date '+%F %T') RQ1 stage 1 finished: $(compgen -G "${OUT}/rq1-*/metrics.json" | wc -l) runs with metrics.json"
