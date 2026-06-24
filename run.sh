#!/bin/bash
set -ex

# ─── SDPO K=3 naive IS collapse experiment ───────────────────────────────
# Model: Qwen3-4B. WandB: sdpo-tau-retail-glm.
# Fully async K=3 + naive unclipped IS (use_is=true, is_clip=null).
# Trust-region teacher, lr=1e-6, G=1.
# Expected: improvement then collapse within 40 steps.
#
# Usage: SEED=0 bash run.sh

export WANDB_API_KEY=${WANDB_API_KEY:-}
export HF_TOKEN=${HF_TOKEN:-}
SEED=${SEED:-0}

# ─── Sync dependencies ──────────────────────────────────────────────────
uv sync --extra fsdp
source .venv/bin/activate

# ─── Prepare tau-retail dataset ──────────────────────────────────────────
python scripts/tau_retail/prepare_data.py --output-dir /root/data/tau_retail

# ─── Training config ─────────────────────────────────────────────────────
MODEL_PATH="Qwen/Qwen3-4B"
RUN_NAME="sdpo_async_k3_naive_is_seed${SEED}"
DATA_DIR="/root/data/tau_retail"

python -m skyrl.train.entrypoints.main_sdpo \
  data.train_data="['$DATA_DIR/tau_retail_train.parquet']" \
  data.val_data="['$DATA_DIR/tau_retail_eval.parquet']" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.ref.model.path=$MODEL_PATH \
  trainer.critic.model.path=null \
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
  trainer.algorithm.sdpo.use_is=true \
  trainer.algorithm.sdpo.is_clip=null \
  trainer.algorithm.sdpo.hint_max_length=2048 \
  trainer.algorithm.sdpo.teacher_mode=policy \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.seed=$SEED \
  trainer.fully_async.max_staleness_steps=3 \
  trainer.fully_async.num_parallel_generation_workers=64 \
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
  generator.inference_engine.num_engines=4 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.max_input_length=10240 \
  "$@"

# ─── Write EVAL.md ───────────────────────────────────────────────────────
mkdir -p .openresearch/artifacts
cat > .openresearch/artifacts/EVAL.md << EOF
# SDPO K=3 Naive IS Collapse — Tau-Retail (Qwen3-4B) — Seed $SEED

## Config
- Algorithm: SDPO (reverse KL, trust-region teacher, lr=1e-6)
- Staleness: K=3 (fully async)
- IS correction: naive unclipped (use_is=true, is_clip=null)
- G=1, failures retained
- Model: Qwen/Qwen3-4B
- Steps: 40, Batch size: 16
- Seed: $SEED
- WandB: sdpo-tau-retail-glm / $RUN_NAME

## Expected Result
Brief improvement then collapse as IS ratios blow up on rare tokens.
EOF
