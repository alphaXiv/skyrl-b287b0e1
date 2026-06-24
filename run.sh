#!/bin/bash
set -ex

# ─── SDPO from SFT model: K=0 stable (lr=1e-6) AND K=3 collapse (lr=1e-5) ─
# Runs two configs sequentially from the SFT-warmed model.
# K=0 lr=1e-6: stable baseline
# K=3 lr=1e-5 naive IS: collapse experiment (3 seeds)

export WANDB_API_KEY=${WANDB_API_KEY:-}
export HF_TOKEN=${HF_TOKEN:-}
SEED=${SEED:-0}
MODE=${MODE:-k0_stable}  # k0_stable or k3_collapse

uv sync --extra fsdp
source .venv/bin/activate
python scripts/tau_retail/prepare_data.py --output-dir /root/data/tau_retail

MODEL_PATH="alphaXiv/sdpo-tau-retail-sft-qwen3-4b"
SFT_REPO="alphaXiv/sdpo-tau-retail-sft-qwen3-4b"
DATA_DIR="/root/data/tau_retail"

# Ensure HF repo has model at root + tokenizer
python -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
try:
    api.hf_hub_download(repo_id='$SFT_REPO', filename='model.safetensors')
    print('Model already at root')
except Exception:
    print('Reorganizing: moving policy/* to root...')
    api.snapshot_download(repo_id='$SFT_REPO', local_dir='/tmp/sft_model', allow_patterns=['policy/*'])
    import os
    for f in os.listdir('/tmp/sft_model/policy'):
        api.upload_file(path_or_fileobj=f'/tmp/sft_model/policy/{f}', path_in_repo=f, repo_id='$SFT_REPO', repo_type='model')
    print('Model files moved to root')
try:
    api.hf_hub_download(repo_id='$SFT_REPO', filename='tokenizer_config.json')
    print('Tokenizer present')
except Exception:
    print('Uploading tokenizer...')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
    tok.push_to_hub('$SFT_REPO', token='$HF_TOKEN')
    print('Tokenizer uploaded')
"

if [ "$MODE" = "k0_stable" ]; then
    RUN_NAME="sdpo_sync_k0_sft_lr1e6_seed${SEED}"
    LR=1e-6
    STALENESS=0
    COLOCATE=true
    POLICY_GPUS=4
    NUM_ENGINES=4
    GPU_MEM=0.45
    USE_IS=false
    IS_CLIP=null
    NUM_WORKERS=32
    WEIGHT_SYNC=""
else
    RUN_NAME="sdpo_async_k3_sft_lr1e5_naive_is_seed${SEED}"
    LR=1e-5
    STALENESS=3
    COLOCATE=false
    POLICY_GPUS=4
    NUM_ENGINES=4
    GPU_MEM=0.8
    USE_IS=true
    IS_CLIP=null
    NUM_WORKERS=64
    WEIGHT_SYNC="weight_sync_backend=nccl"
fi

COLOCATE_FLAG=""
if [ "$COLOCATE" = "true" ]; then
    COLOCATE_FLAG="trainer.placement.colocate_all=true"
else
    COLOCATE_FLAG="trainer.placement.colocate_all=false trainer.placement.colocate_policy_ref=true"
fi

python -m skyrl.train.entrypoints.main_sdpo \
  data.train_data="['$DATA_DIR/tau_retail_train.parquet']" \
  data.val_data="['$DATA_DIR/tau_retail_eval.parquet']" \
  trainer.strategy=fsdp \
  $COLOCATE_FLAG \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.ref.model.path=$MODEL_PATH \
  trainer.critic.model.path=null \
  trainer.placement.policy_num_gpus_per_node=$POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$POLICY_GPUS \
  trainer.epochs=6 \
  trainer.max_training_steps=40 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=4096 \
  trainer.algorithm.max_seq_len=11264 \
  trainer.algorithm.policy_loss_type=sdpo \
  trainer.algorithm.advantage_estimator=sdpo_no_op \
  trainer.algorithm.use_kl_in_reward=true \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.temperature=1.0 \
  trainer.algorithm.zero_variance_filter=false \
  trainer.algorithm.sdpo.use_is=$USE_IS \
  trainer.algorithm.sdpo.is_clip=$IS_CLIP \
  trainer.algorithm.sdpo.hint_max_length=2048 \
  trainer.algorithm.sdpo.teacher_mode=policy \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.seed=$SEED \
  trainer.fully_async.max_staleness_steps=$STALENESS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_WORKERS \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.eval_batch_size=20 \
  generator.eval_n_samples_per_prompt=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.logger=wandb \
  trainer.project_name=sdpo-tau-retail-glm \
  trainer.run_name=$RUN_NAME \
  trainer.ckpt_path="/root/ckpts/$RUN_NAME" \
  trainer.export_path="/root/exports/$RUN_NAME" \
  trainer.resume_mode=none \
  environment.env_class=tau_retail \
  generator.n_samples_per_prompt=1 \
  generator.max_turns=10 \
  generator.batched=false \
  generator.sampling_params.max_generate_length=512 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.logprobs=1 \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEM \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.async_engine=true \
  ${WEIGHT_SYNC:+generator.inference_engine.$WEIGHT_SYNC} \
  generator.max_input_length=10240 \
  "$@"

mkdir -p .openresearch/artifacts
cat > .openresearch/artifacts/EVAL.md << EOF
# SDPO from SFT — $MODE — Seed $SEED

## Config
- Model: alphaXiv/sdpo-tau-retail-sft-qwen3-4b (SFT-warmed Qwen3-4B)
- Mode: $MODE (lr=$LR, staleness=$STALENESS, use_is=$USE_IS)
- Trust-region teacher, G=1
- Steps: 40
- WandB: sdpo-tau-retail-glm / $RUN_NAME
EOF
