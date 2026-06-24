#!/bin/bash
set -ex

# ─── SFT warmup on tau-retail successful trajectories ────────────────────
# Light SFT (1 epoch, lr=5e-6) to teach Qwen3-4B the JSON tool-call format.
# Save to HF hub for use as RL starting point.

export WANDB_API_KEY=${WANDB_API_KEY:-}
export HF_TOKEN=${HF_TOKEN:-}

# ─── Sync dependencies ──────────────────────────────────────────────────
uv sync --extra fsdp
source .venv/bin/activate

# ─── Prepare SFT dataset ─────────────────────────────────────────────────
python scripts/tau_retail/prepare_sft_data.py --output-dir /root/data/tau_retail_sft
# datasets.load_dataset expects a directory, not a file
mkdir -p /root/data/tau_retail_sft_dir
cp /root/data/tau_retail_sft/tau_retail_sft.parquet /root/data/tau_retail_sft_dir/

# ─── SFT training ────────────────────────────────────────────────────────
MODEL_PATH="Qwen/Qwen3-4B"
RUN_NAME="sft_tau_retail_qwen3_4b"
SFT_DATA="/root/data/tau_retail_sft_dir"
HF_REPO="rehaanahmad2013/sdpo-tau-retail-sft-qwen3-4b"

python -m skyrl.train.main_sft \
  strategy=fsdp \
  model.path=$MODEL_PATH \
  dataset_name=$SFT_DATA \
  dataset_split=train \
  messages_key=messages \
  max_length=16384 \
  num_steps=200 \
  batch_size=2 \
  micro_train_batch_size_per_gpu=1 \
  remove_microbatch_padding=true \
  seed=42 \
  optimizer_config.lr=5e-6 \
  optimizer_config.weight_decay=0.0 \
  optimizer_config.max_grad_norm=1.0 \
  optimizer_config.num_warmup_steps=10 \
  optimizer_config.scheduler=constant_with_warmup \
  placement.num_nodes=1 \
  placement.num_gpus_per_node=8 \
  fsdp_config.cpu_offload=false \
  fsdp_config.reshard_after_forward=true \
  train_on_what=all_assistant_messages \
  force_recache=true \
  logger=wandb \
  project_name=sdpo-tau-retail-glm \
  run_name=$RUN_NAME \
  ckpt_path="/root/ckpts/$RUN_NAME" \
  ckpt_interval=0 \
  hf_save_interval=200 \
  export_path="/root/exports/$RUN_NAME" \
  resume_from="" \
  "$@"

# ─── Upload to HF Hub ────────────────────────────────────────────────────
echo "Uploading SFT model to HF Hub: $HF_REPO"
python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo(repo_id='$HF_REPO', repo_type='model', exist_ok=True)
api.upload_folder(
    folder_path='/root/exports/$RUN_NAME/global_step_200',
    repo_id='$HF_REPO',
    repo_type='model',
)
print(f'Uploaded to https://huggingface.co/$HF_REPO')
"

# ─── Write EVAL.md ───────────────────────────────────────────────────────
mkdir -p .openresearch/artifacts
cat > .openresearch/artifacts/EVAL.md << EOF
# SFT Warmup — Tau-Retail (Qwen3-4B)

## Config
- Model: Qwen/Qwen3-4B
- Data: 115 tau-retail tasks (canonical actions + real tool outputs)
- Steps: 200, lr=5e-6, 1 epoch equivalent
- train_on_what: all_assistant_messages
- HF repo: $HF_REPO

## Result
SFT model saved to HF hub for use as RL starting point.
EOF
