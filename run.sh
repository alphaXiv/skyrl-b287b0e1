#!/bin/bash
set -ex

# ─── SDPO sync K=0 on tau-retail (sanity check: clean & stable) ──────────
# Model: Qwen3-4B (same as paper). WandB: sdpo-tau-retail-glm.
# Uses sync RayPPOTrainer with colocated placement (K=0 via fully_async.max_staleness_steps=0).

export WANDB_API_KEY=${WANDB_API_KEY:-}
export HF_TOKEN=${HF_TOKEN:-}

# ─── Install tau-bench + litellm (litellm not in fsdp extra) ─────────────
uv sync --extra fsdp
uv pip install litellm
uv pip install --no-deps git+https://github.com/sierra-research/tau-bench.git

# ─── Prepare tau-retail dataset ──────────────────────────────────────────
uv run --no-sync --extra fsdp python scripts/tau_retail/prepare_data.py --output-dir /root/data/tau_retail

# ─── Training config ─────────────────────────────────────────────────────
MODEL_PATH="Qwen/Qwen3-4B"
RUN_NAME="sdpo_sync_k0_seed0"
DATA_DIR="/root/data/tau_retail"

uv run --no-sync --extra fsdp -m skyrl.train.entrypoints.main_sdpo \
  data.train_data="['$DATA_DIR/tau_retail_train.parquet']" \
  data.val_data="['$DATA_DIR/tau_retail_eval.parquet']" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.ref.model.path=$MODEL_PATH \
  trainer.critic.model.path=null \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
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
  trainer.algorithm.sdpo.use_is=false \
  trainer.algorithm.sdpo.hint_max_length=1024 \
  trainer.policy.optimizer_config.lr=1e-5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.fully_async.max_staleness_steps=0 \
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
  trainer.resume_mode=disabled \
  environment.env_class=tau_retail \
  generator.n_samples_per_prompt=1 \
  generator.max_turns=10 \
  generator.batched=true \
  generator.sampling_params.max_generate_length=256 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.logprobs=1 \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.num_engines=8 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.enforce_eager=false \
  generator.max_input_length=10240 \
  "$@"

# ─── Write EVAL.md ───────────────────────────────────────────────────────
mkdir -p .openresearch/artifacts
cat > .openresearch/artifacts/EVAL.md << EOF
# SDPO Sync K=0 — Tau-Retail (Qwen3-4B)

## Config
- Algorithm: SDPO (reverse KL, sampled-token estimator)
- Staleness: K=0 (sync, colocated)
- IS correction: disabled (use_is=false)
- Model: Qwen/Qwen3-4B
- Steps: 40
- Batch size: 16, G=1
- Hint: canonical action plan (Task.actions)
- Teacher: frozen ref model (= policy at init) on hinted input
- WandB: sdpo-tau-retail-glm / $RUN_NAME

## Results
See WandB for eval/all/avg_score (pass rate) trajectory.
Training should be clean and stable (no collapse at K=0).
EOF
