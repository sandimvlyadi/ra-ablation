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
#        setsid nohup bash scripts/run_rq1.sh all >> runs/rq1.log 2>&1 < /dev/null &      stage 1 (RQ1-01..06)
#        ... run_rq1.sh tune17                 addendum 1: 3 linear-probe lr runs (pick on validation)
#        ... LP_LR=<lr> run_rq1.sh lp17        RQ1-17a (LP) + RQ1-17b (LP-FT), ImageNet ResNet34
#        ... run_rq1.sh tune12                 addendum 3: 3 SupCon-weight runs (pick on validation)
#        ... SUPCON_W=<w> run_rq1.sh supcon12  RQ1-12 (SupCon + CE) + RQ1-13 (grade-distance negatives, M219)
#        ... run_rq1.sh ssl0810                addendum 4: SimCLR (RQ1-08) and BYOL (RQ1-10) pre-training on the E1 pool,
#                                              then 6 fine-tunes each; pre-training resumes from its checkpoint
#        ... run_rq1.sh boneage07              addendum 6: RSNA bone-age regression pre-training (M005), then 6 fine-tunes
#        ... run_rq1.sh skelex18               addendum 7: SKELEX ViT-L, linear probe (RQ1-18a) + fine-tune LLRD (RQ1-18b)
#        ... run_rq1.sh norm1415               addendum 5: RQ1-14 normality model (M197 stage 1) + RQ1-15 gate on RQ0-01
# Smoke test of ssl0810: also set PRE=<dir> PRETRAIN_ARGS="--epochs 1 --limit 300" so the pre-training stays small.
# Smoke test: OUT=/tmp/rq1 bash scripts/run_rq1.sh all --limit 640 --epochs 1 --min-steps 0
# MINUTES are rough: ResNet34 from E3, ViT-S from E2 medians.
set -uo pipefail
ROOT="${QUEUE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# bash reads a script while it runs, so a queue runs a frozen copy: editing this file cannot reach a running queue
if [[ -n "${1:-}" && -z "${QUEUE_ROOT:-}" ]]; then
    mkdir -p "${ROOT}/runs/.queue_snapshots"
    snap="${ROOT}/runs/.queue_snapshots/run_rq1_$1_$(date +%Y%m%d_%H%M%S).sh"
    cp "$0" "${snap}"
    QUEUE_ROOT="${ROOT}" exec bash "${snap}" "$@"
fi
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
declare -A MINUTES=([resnet34_0.1]=7 [resnet34_1.0]=31 [vits_0.1]=10 [vits_1.0]=40 [supcon_0.1]=10 [supcon_1.0]=44
                    [pretrain]=270 [norm_0.1]=5 [norm_1.0]=15 [gate]=1 [boneage]=80
                    [vitl_lp_0.1]=15 [vitl_lp_1.0]=60 [vitl_0.1]=45 [vitl_1.0]=200)   # ViT-L: not measured yet
PRE="${PRE:-${ROOT}/runs/rq1_pretrain}"
read -r -a PRETRAIN_EXTRA <<< "${PRETRAIN_ARGS:-}"

PLAN=""
run() {   # name, minutes, then the arm's train.py flags; a failure is reported and the queue goes on
    local name=$1 minutes=$2
    shift 2
    if [[ -n "${PLAN}" ]]; then
        echo "${name%%_*} $([[ -f "${OUT}/${name}/metrics.json" ]] && echo 1 || echo 0) ${minutes}"
        return 0
    fi
    [[ -f "${OUT}/${name}/metrics.json" ]] && return 0
    rm -rf "${OUT:?}/${name}"
    echo "$(date '+%F %T') [$(compgen -G "${OUT}/rq1-*/metrics.json" | wc -l) done] ${name}"
    local t0=${SECONDS}
    if ! uv run python -m ramscore.train --name "${name}" --out "${OUT}" --amp --n-boot 0 "$@" \
        --head softmax --logit-adjust 1 --epochs 50 --min-steps 2000 --aug base --seed-aug "${EXTRA[@]}" \
        > "${OUT}/logs/${name}.log" 2>&1; then
        echo "FAILED ${name} (see ${OUT}/logs/${name}.log); rerun the script to retry" >&2
        return 1
    fi
    echo "$(date '+%F %T') finished ${name} in $(( (SECONDS - t0) / 60 )) min"
}

design() { echo --label-fraction "$1" --fraction-seed "$2" --seed "$3"; }

train() {   # stage-1 arm, fraction, draw, seed
    local arm=$1 frac=$2 draw=$3 seed=$4
    # shellcheck disable=SC2046,SC2086  # FLAGS and design() hold several words on purpose
    run "${arm}_${TAGNAME[${arm}]}_b2_f${frac}_d${draw}_s${seed}" "${MINUTES[${TAGNAME[${arm}]%%_*}_${frac}]}" \
        ${FLAGS[${arm}]} $(design "${frac}" "${draw}" "${seed}")
}

queue() {
    local draw seed arm
    for draw in 0 1 2; do for arm in "${ARMS[@]}"; do train "${arm}" 0.1 "${draw}" 42; done; done
    for seed in 42 1337 2026; do for arm in "${ARMS[@]}"; do train "${arm}" 1.0 0 "${seed}"; done; done
}

# Addendum 1: linear-probe lr on validation (ImageNet ResNet34, 10% d0 s42), then RQ1-17a (LP) and RQ1-17b (LP-FT,
# stage 2 from the 17a run of the same design, full fine-tune recipe). LP has no LLRD and no warm-up.
tune17() {
    local lr
    for lr in 1e-4 1e-3 1e-2; do
        # shellcheck disable=SC2046
        run "rq1-17tune_lr${lr}_resnet34_imagenet_lp_b2_f0.1_d0_s42" 7 --model resnet34 --pretrained --freeze-backbone \
            --lr "${lr}" $(design 0.1 0 42)
    done
}

lp17_one() {   # fraction, draw, seed
    local frac=$1 draw=$2 seed=$3 tag="b2_f$1_d$2_s$3"
    local lp="rq1-17a_resnet34_imagenet_lp_${tag}"
    # shellcheck disable=SC2046
    run "${lp}" "${MINUTES[resnet34_${frac}]}" --model resnet34 --pretrained --freeze-backbone --lr "${LP_LR}" \
        $(design "${frac}" "${draw}" "${seed}") || return 1
    if [[ -z "${PLAN}" && ! -f "${OUT}/${lp}/model_best.pt" ]]; then
        echo "SKIPPED rq1-17b ${tag}: ${lp}/model_best.pt missing" >&2
        return 1
    fi
    # shellcheck disable=SC2046
    run "rq1-17b_resnet34_imagenet_lpft_${tag}" "${MINUTES[resnet34_${frac}]}" --model resnet34 \
        --init-from "${OUT}/${lp}/model_best.pt" --keep-head $(design "${frac}" "${draw}" "${seed}")
}

lp17() {
    local draw seed
    for draw in 0 1 2; do lp17_one 0.1 "${draw}" 42; done
    for seed in 42 1337 2026; do lp17_one 1.0 0 "${seed}"; done
}

# Addendum 3: SupCon weight on validation (RQ1-12, ImageNet ResNet34, 10% d0 s42), then RQ1-12 and RQ1-13 with that
# weight. Two views; the classification loss sees the first only. RQ1-13 differs from RQ1-12 only by --supcon-alpha.
tune12() {
    local w
    for w in 0.1 0.5 1.0; do
        # shellcheck disable=SC2046
        run "rq1-12tune_lam${w}_resnet34_imagenet_supcon_b2_f0.1_d0_s42" "${MINUTES[supcon_0.1]}" --model resnet34 \
            --pretrained --supcon-weight "${w}" $(design 0.1 0 42)
    done
}

supcon12_one() {   # fraction, draw, seed
    local frac=$1 draw=$2 seed=$3 tag="b2_f$1_d$2_s$3"
    # shellcheck disable=SC2046
    run "rq1-12_resnet34_imagenet_supcon_${tag}" "${MINUTES[supcon_${frac}]}" --model resnet34 --pretrained \
        --supcon-weight "${SUPCON_W}" $(design "${frac}" "${draw}" "${seed}")
    # shellcheck disable=SC2046
    run "rq1-13_resnet34_imagenet_supconord_${tag}" "${MINUTES[supcon_${frac}]}" --model resnet34 --pretrained \
        --supcon-weight "${SUPCON_W}" --supcon-alpha 0.25 $(design "${frac}" "${draw}" "${seed}")
}

supcon12() {
    local draw seed
    for draw in 0 1 2; do supcon12_one 0.1 "${draw}" 42; done
    for seed in 42 1337 2026; do supcon12_one 1.0 0 "${seed}"; done
}

# Addendum 4: SimCLR and BYOL with the E1 DINO ResNet34 recipe (pool h1200, from scratch, 100 epochs, AdamW lr 5e-4 per
# 256, wd 0.04 -> 0.4, warm-up 10, seed 2026), batch 128, the two asymmetric DINO global crops; encoder = online network.
pretrain() {   # name, method
    local name=$1 method=$2 dir="${PRE}/$1"
    if [[ -n "${PLAN}" ]]; then
        echo "${name%%_*}-pre $([[ -f "${dir}/done.json" ]] && echo 1 || echo 0) ${MINUTES[pretrain]}"
        return 0
    fi
    [[ -f "${dir}/done.json" ]] && return 0
    mkdir -p "${PRE}/logs"
    echo "$(date '+%F %T') pre-training ${name}$([[ -f "${dir}/checkpoint.pt" ]] && echo ' (resume)')"
    local t0=${SECONDS}
    if ! uv run python -m ramscore.pretrain --name "${name}" --out "${PRE}" --method "${method}" --model resnet34 --amp \
        --resume "${PRETRAIN_EXTRA[@]}" >> "${PRE}/logs/${name}.log" 2>&1; then
        echo "FAILED pre-training ${name} (see ${PRE}/logs/${name}.log); rerun the script to resume" >&2
        return 1
    fi
    (cd "${dir}" && sha256sum encoder.pt > encoder.sha256)
    echo "$(date '+%F %T') pre-trained ${name} in $(( (SECONDS - t0) / 60 )) min"
}

finetunes() {   # arm, tag, encoder: the 6 designs fine-tuned from one encoder
    local arm=$1 tag=$2 enc=$3 draw seed
    for draw in 0 1 2; do
        # shellcheck disable=SC2046
        run "${arm}_${tag}_b2_f0.1_d${draw}_s42" "${MINUTES[resnet34_0.1]}" --model resnet34 --init-from "${enc}" \
            $(design 0.1 "${draw}" 42)
    done
    for seed in 42 1337 2026; do
        # shellcheck disable=SC2046
        run "${arm}_${tag}_b2_f1.0_d0_s${seed}" "${MINUTES[resnet34_1.0]}" --model resnet34 --init-from "${enc}" \
            $(design 1.0 0 "${seed}")
    done
}

ssl_arm() {   # arm, tag, method
    pretrain "${1}_${3}_resnet34" "${3}" || return 1
    finetunes "$1" "$2" "${PRE}/${1}_${3}_resnet34/encoder.pt"
}

ssl0810() {
    ssl_arm rq1-08 resnet34_simclr simclr
    ssl_arm rq1-10 resnet34_byol byol
}

# Addendum 6: bone-age regression on the prepared RSNA images (ImageNet ResNet34, M005 recipe), then RQ1-07 fine-tunes.
BONEAGE="${BONEAGE:-${ROOT}/data/external/rsna-bone-age/png512}"
boneage07() {
    local name=rq1-07_boneage_resnet34 dir="${PRE}/rq1-07_boneage_resnet34"
    if [[ -n "${PLAN}" ]]; then
        echo "rq1-07-pre $([[ -f "${dir}/done.json" ]] && echo 1 || echo 0) ${MINUTES[boneage]}"
    elif [[ ! -f "${dir}/done.json" ]]; then
        mkdir -p "${PRE}/logs"
        echo "$(date '+%F %T') pre-training ${name}$([[ -f "${dir}/checkpoint.pt" ]] && echo ' (resume)')"
        local t0=${SECONDS}
        if ! uv run python -m ramscore.boneage pretrain --data "${BONEAGE}" --name "${name}" --out "${PRE}" --amp \
            "${PRETRAIN_EXTRA[@]}" >> "${PRE}/logs/${name}.log" 2>&1; then
            echo "FAILED pre-training ${name} (see ${PRE}/logs/${name}.log); rerun the script to resume" >&2
            return 1
        fi
        (cd "${dir}" && sha256sum encoder.pt > encoder.sha256)
        echo "$(date '+%F %T') pre-trained ${name} in $(( (SECONDS - t0) / 60 )) min"
    fi
    finetunes rq1-07 resnet34_boneage "${dir}/encoder.pt"
}

# Addendum 7: SKELEX converted to timm (skelex.py convert/verify). 18a = linear probe with the addendum-1 lr (1e-2, no
# LLRD, no warm-up); 18b = the ViT recipe (LLRD 0.85, lr 1e-4, warm-up 10%) with gradient checkpointing (same result).
SKELEX="${SKELEX:-${ROOT}/data/external/skelex/encoder_timm.pt}"
skelex18_one() {   # fraction, draw, seed
    local frac=$1 draw=$2 seed=$3 tag="b2_f$1_d$2_s$3"
    # shellcheck disable=SC2046
    run "rq1-18a_vitl_skelex_lp_${tag}" "${MINUTES[vitl_lp_${frac}]}" --model vit_large_patch16_224 \
        --init-from "${SKELEX}" --freeze-backbone --lr 1e-2 $(design "${frac}" "${draw}" "${seed}")
    # shellcheck disable=SC2046,SC2086
    run "rq1-18b_vitl_skelex_${tag}" "${MINUTES[vitl_${frac}]}" --model vit_large_patch16_224 --init-from "${SKELEX}" \
        ${VIT} --grad-checkpointing $(design "${frac}" "${draw}" "${seed}")
}

skelex18() {
    local draw seed
    for draw in 0 1 2; do skelex18_one 0.1 "${draw}" 42; done
    for seed in 42 1337 2026; do skelex18_one 1.0 0 "${seed}"; done
}

# Addendum 5: RQ1-14 trains on the grade-0 joints of the labelled training patients of each design; RQ1-15 gates the
# RQ0-01 classifier of the same design with that score (no training). Both write the train.py run layout.
norm_run() {   # name, minutes, then ramscore.normality arguments
    local name=$1 minutes=$2
    shift 2
    if [[ -n "${PLAN}" ]]; then
        echo "${name%%_*} $([[ -f "${OUT}/${name}/metrics.json" ]] && echo 1 || echo 0) ${minutes}"
        return 0
    fi
    [[ -f "${OUT}/${name}/metrics.json" ]] && return 0
    rm -rf "${OUT:?}/${name}"
    echo "$(date '+%F %T') [$(compgen -G "${OUT}/rq1-*/metrics.json" | wc -l) done] ${name}"
    local t0=${SECONDS}
    if ! uv run python -m ramscore.normality "$@" --name "${name}" --out "${OUT}" "${EXTRA[@]}" \
        > "${OUT}/logs/${name}.log" 2>&1; then
        echo "FAILED ${name} (see ${OUT}/logs/${name}.log); rerun the script to retry" >&2
        return 1
    fi
    echo "$(date '+%F %T') finished ${name} in $(( (SECONDS - t0) / 60 )) min"
}

norm_one() {   # fraction, draw, seed
    local frac=$1 draw=$2 seed=$3 tag="b2_f$1_d$2_s$3"
    local n14="rq1-14_resnet34_imagenet_fewsome_${tag}" ref="${ROOT}/runs/rq0/rq0-01_resnet34_${tag}"
    # shellcheck disable=SC2046
    norm_run "${n14}" "${MINUTES[norm_${frac}]}" train --amp $(design "${frac}" "${draw}" "${seed}") || return 1
    if [[ -z "${PLAN}" && ! -f "${ref}/model_best.pt" ]]; then
        echo "SKIPPED rq1-15 ${tag}: ${ref}/model_best.pt missing" >&2
        return 1
    fi
    # shellcheck disable=SC2046
    norm_run "rq1-15_resnet34_imagenet_gate_${tag}" "${MINUTES[gate]}" gate --normality "${OUT}/${n14}" \
        --classifier "${ref}" $(design "${frac}" "${draw}" "${seed}")
}

norm1415() {
    local draw seed
    for draw in 0 1 2; do norm_one 0.1 "${draw}" 42; done
    for seed in 42 1337 2026; do norm_one 1.0 0 "${seed}"; done
}

table() { awk '{ n[$1]++; d[$1] += $2; if (!$2) m[$1] += $3 } END { printf "%-12s %10s %10s\n", "arm", "runs done", "hours left"; for (a in n) printf "%-12s %5d/%-4d %10.1f\n", a, d[a], n[a], m[a] / 60 }' | sort; }

if [[ -z "${STAGE}" ]]; then
    PLAN=1
    LP_LR="${LP_LR:-x}"
    SUPCON_W="${SUPCON_W:-x}"
    echo "OUT=${OUT}  (hours are rough)"
    { queue; tune17; lp17; tune12; supcon12; ssl0810; norm1415; boneage07; skelex18; } | table
    exit 0
fi
case "${STAGE}" in
    all|tune17|tune12|ssl0810|norm1415|boneage07|skelex18) ;;
    supcon12) [[ -n "${SUPCON_W:-}" ]] || { echo "supcon12 needs the weight chosen in addendum 3: SUPCON_W=<w> bash scripts/run_rq1.sh supcon12" >&2; exit 1; } ;;
    lp17) [[ -n "${LP_LR:-}" ]] || { echo "lp17 needs the lr chosen in addendum 1: LP_LR=<lr> bash scripts/run_rq1.sh lp17" >&2; exit 1; } ;;
    *) echo "unknown stage ${STAGE}; use: all (stage 1), tune17, lp17, tune12, supcon12, ssl0810, norm1415, boneage07, skelex18" >&2; exit 1 ;;
esac
(cd "${ROOT}/artifacts" && sha256sum --quiet -c SHA256SUMS) || { echo "encoder checksums failed; queue not started" >&2; exit 1; }
mkdir -p "${OUT}/logs"
ps -o sid= -p $$ | tr -d " " > "${OUT}/queue.pid"
echo "$(date '+%F %T') RQ1 ${STAGE} started (pid $$)"
case "${STAGE}" in
    all) queue ;;
    tune17) tune17 ;;
    lp17) lp17 ;;
    tune12) tune12 ;;
    supcon12) supcon12 ;;
    ssl0810) ssl0810 ;;
    norm1415) norm1415 ;;
    boneage07) boneage07 ;;
    skelex18) skelex18 ;;
esac
echo "$(date '+%F %T') RQ1 ${STAGE} finished: $(compgen -G "${OUT}/rq1-*/metrics.json" | wc -l) runs with metrics.json"
