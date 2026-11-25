#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 DL3DV_ROOT DEPTH_CHECKPOINT OUTPUT_DIR BATCH_SIZE [Hydra overrides...]" >&2
    exit 2
fi
data=$(cd "$1" && pwd)
checkpoint=$2
if [[ "$checkpoint" != "null" ]]; then
    checkpoint=$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")
    [[ -r "$checkpoint" ]] || { echo "Cannot read the depth checkpoint: $checkpoint" >&2; exit 2; }
fi
mkdir -p "$3"
output=$(cd "$3" && pwd)
batch=$4
shift 4
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export RESPLAT_PROFILE_JSON="$output/step-profile.json"
export RESPLAT_PROFILE_WARMUP=${RESPLAT_PROFILE_WARMUP:-10}
export RESPLAT_PROFILE_SAMPLES=${RESPLAT_PROFILE_SAMPLES:-20}
steps=$((RESPLAT_PROFILE_WARMUP + RESPLAT_PROFILE_SAMPLES))
exec "${PYTHON:-python}" -m src.main +experiment=dl3dv \
    "dataset.roots=['$data']" \
    "data_loader.train.batch_size=$batch" \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.num_target_views=6 \
    dataset.view_sampler.min_distance_between_context_views=24 \
    dataset.view_sampler.max_distance_between_context_views=45 \
    dataset.view_sampler.initial_min_distance_between_context_views=20 \
    dataset.view_sampler.initial_max_distance_between_context_views=30 \
    'dataset.image_shape=[256,448]' \
    model.encoder.init_use_local_knn=true \
    trainer.num_nodes=1 \
    trainer.num_sanity_val_steps=0 \
    "trainer.max_steps=$steps" \
    "checkpointing.pretrained_depth=$checkpoint" \
    checkpointing.save_top_k=0 \
    train.eval_model_every_n_val=0 \
    wandb.mode=disabled \
    "output_dir=$output" \
    "$@"
