"""SDPO (Self-Distillation Policy Optimization) entrypoint for SkyRL.

Implements the SDPO loss from Hübotter et al.:
  L = E_τ~π_θ [ Σ_t KL(π_θ(·|s_t) || stopgrad(π_θ(·|s_t, c))) ]

Sampled-token reverse-KL estimator (alpha=1, reverse KL):
  per_token_loss = (student_log_prob - teacher_log_prob).detach() * student_log_prob

Off-policy IS correction (naive, unclipped — causes collapse at K>0):
  ratio = exp(student_log_prob - rollout_log_prob).detach()
  per_token_loss *= ratio

The teacher = the policy model (frozen ref) run on the *hinted* input
(hint prepended to the prompt).  Teacher log-probs are computed via the ref
model forward pass on teacher_sequences, then stashed into the rewards/advantages
pipeline so they reach the loss function.

Trainer selection:
  - ``trainer.fully_async.max_staleness_steps > 0`` → SDPOAsyncTrainer (FullyAsync)
  - ``== 0`` → SDPOSyncTrainer (sync RayPPOTrainer, colocated)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import ray

from skyrl.train.config import (
    SkyRLTrainConfig,
    AlgorithmConfig,
    BaseConfig,
    make_config,
)
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_policy_loss,
    register_advantage_estimator,
    PolicyLossRegistry,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors


# ─── Config ──────────────────────────────────────────────────────────────

@dataclass
class SDPOConfig(BaseConfig):
    """SDPO hyperparameters."""
    use_is: bool = False
    is_clip: Optional[float] = None
    hint_max_length: int = 1024


@dataclass
class SDPOAlgorithmConfig(AlgorithmConfig):
    sdpo: SDPOConfig = field(default_factory=SDPOConfig)


SDPOSkyRLConfig = make_config(algorithm_cls=SDPOAlgorithmConfig)


# ─── Loss function ───────────────────────────────────────────────────────

@register_policy_loss("sdpo")
def sdpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """SDPO loss: per-token reverse KL with optional naive IS correction.

    ``advantages`` carries the teacher log-probs (stashed via
    ``apply_reward_kl_penalty`` + the ``sdpo_no_op`` advantage estimator).
    """
    teacher_log_probs = advantages

    log_ratio = (log_probs - teacher_log_probs).detach()
    per_token_loss = log_ratio * log_probs

    metrics: Dict[str, Any] = {}
    sdpo_cfg = getattr(config, "sdpo", None)

    if sdpo_cfg is not None and sdpo_cfg.use_is and rollout_logprobs is not None:
        is_log_ratio = (log_probs - rollout_logprobs).detach()
        is_log_ratio = torch.clamp(is_log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(is_log_ratio)
        if sdpo_cfg.is_clip is not None:
            ratio = ratio.clamp(max=sdpo_cfg.is_clip)
        per_token_loss = per_token_loss * ratio
        with torch.no_grad():
            metrics["sdpo/is_ratio_mean"] = ratio.float().mean().item()
            metrics["sdpo/is_ratio_max"] = ratio.float().max().item()
            metrics["sdpo/is_ratio_std"] = ratio.float().std().item()

    if loss_mask is not None:
        mask_sum = loss_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * loss_mask).sum() / mask_sum
        with torch.no_grad():
            kl_per_token = log_ratio.float()
            metrics["sdpo/kl_mean"] = (kl_per_token * loss_mask).sum().item() / mask_sum.item()
            metrics["sdpo/kl_max"] = (kl_per_token.abs() * loss_mask).max().item()
            # Debug: log per-token stats for first sample first 10 tokens
            import os as _os
            if _os.environ.get("SDPO_DEBUG"):
                from loguru import logger as _logger
                _logger.info(f"[SDPO LOSS DEBUG] loss={loss.item():.4f} mask_sum={mask_sum.item():.0f}")
                _logger.info(f"[SDPO LOSS DEBUG] student logp: mean={log_probs[loss_mask>0].mean().item():.3f} std={log_probs[loss_mask>0].std().item():.3f}")
                _logger.info(f"[SDPO LOSS DEBUG] teacher logp: mean={teacher_log_probs[loss_mask>0].mean().item():.3f} std={teacher_log_probs[loss_mask>0].std().item():.3f}")
                _logger.info(f"[SDPO LOSS DEBUG] log_ratio: mean={log_ratio[loss_mask>0].mean().item():.3f} max={log_ratio[loss_mask>0].max().item():.3f} min={log_ratio[loss_mask>0].min().item():.3f}")
                _logger.info(f"[SDPO LOSS DEBUG] per_token_loss: mean={per_token_loss[loss_mask>0].mean().item():.3f} max={per_token_loss[loss_mask>0].max().item():.3f}")
    else:
        loss = per_token_loss.mean()

    return loss, metrics


@register_advantage_estimator("sdpo_no_op")
def sdpo_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


# ─── Trainer mixin ───────────────────────────────────────────────────────

class SDPOTrainerMixin:
    """SDPO overrides shared by sync and async trainers."""

    def _sdpo_init(self):
        self._uid_to_hint: Dict[str, str] = {}
        train_ds = getattr(self, "train_dataset", None)
        if train_ds is not None and hasattr(train_ds, "dataframe"):
            for i in range(len(train_ds)):
                try:
                    row = train_ds.dataframe[i]
                    self._uid_to_hint[str(i)] = row.get("hint", "")
                except Exception:
                    pass
        self._hint_cache: Dict[str, List[int]] = {}

    def _get_hint_ids(self, uid: str) -> List[int]:
        if uid in self._hint_cache:
            return self._hint_cache[uid]
        hint_text = self._uid_to_hint.get(uid, "")
        if not hint_text:
            self._hint_cache[uid] = []
            return []
        hint_ids = self.tokenizer.encode(hint_text, add_special_tokens=False)
        sdpo_cfg = self.cfg.trainer.algorithm.sdpo
        max_len = sdpo_cfg.hint_max_length
        if len(hint_ids) > max_len:
            hint_ids = hint_ids[:max_len]
        self._hint_cache[uid] = hint_ids
        return hint_ids

    def convert_to_training_input(self, generator_output, uids):
        training_input = super().convert_to_training_input(generator_output, uids)

        prompt_ids_list = generator_output["prompt_token_ids"]
        response_ids_list = generator_output["response_ids"]

        teacher_prompts: List[List[int]] = []
        for i, uid in enumerate(uids):
            hint_ids = self._get_hint_ids(uid)
            teacher_prompts.append(hint_ids + list(prompt_ids_list[i]))

        dummy_rewards = [[0.0] * len(r) for r in response_ids_list]
        dummy_loss_masks = [[0] * len(r) for r in response_ids_list]
        (
            teacher_sequences,
            teacher_attention_mask,
            _,
            _,
            _,
            _,
            _,
        ) = convert_prompts_responses_to_batch_tensors(
            self.tokenizer,
            teacher_prompts,
            response_ids_list,
            dummy_rewards,
            dummy_loss_masks,
            None,
            None,
            max_seq_len=self.cfg.trainer.algorithm.max_seq_len,
        )

        training_input["teacher_sequences"] = teacher_sequences
        training_input["teacher_attention_mask"] = teacher_attention_mask

        # ─── DEBUG: dump first 3 samples ───────────────────────────────
        if self.global_step < 3:
            import json as _json
            os.makedirs(".openresearch/artifacts", exist_ok=True)
            debug_path = f".openresearch/artifacts/debug_step{self.global_step}.jsonl"
            with open(debug_path, "a") as f:
                for i in range(min(3, len(uids))):
                    hint_text = self._uid_to_hint.get(uids[i], "")[:500]
                    prompt_decoded = self.tokenizer.decode(prompt_ids_list[i])[:1000]
                    response_decoded = self.tokenizer.decode(response_ids_list[i])[:1000]
                    teacher_prompt_decoded = self.tokenizer.decode(teacher_prompts[i])[:1500]
                    teacher_seq_decoded = self.tokenizer.decode(teacher_sequences[i].tolist())[:1500]
                    loss_masks_i = generator_output.get("loss_masks", [[0]])[i] if i < len(generator_output.get("loss_masks", [])) else [0]
                    rewards_i = generator_output.get("rewards", [[0.0]])[i] if i < len(generator_output.get("rewards", [])) else [0.0]
                    entry = {
                        "step": self.global_step,
                        "sample_idx": i,
                        "uid": uids[i],
                        "hint_text": hint_text,
                        "prompt_decoded": prompt_decoded,
                        "response_decoded": response_decoded,
                        "teacher_prompt_decoded": teacher_prompt_decoded,
                        "teacher_seq_decoded": teacher_seq_decoded,
                        "response_len": len(response_ids_list[i]),
                        "prompt_len": len(prompt_ids_list[i]),
                        "loss_mask_sum": sum(loss_masks_i) if isinstance(loss_masks_i, list) else 0,
                        "reward": rewards_i if isinstance(rewards_i, (int, float)) else (sum(rewards_i) if isinstance(rewards_i, list) else 0),
                        "stop_reason": generator_output.get("stop_reasons", [""])[i] if i < len(generator_output.get("stop_reasons", [])) else "",
                    }
                    f.write(_json.dumps(entry) + "\n")
            from loguru import logger as _logger
            _logger.info(f"[SDPO DEBUG] Dumped {min(3, len(uids))} samples to {debug_path}")
            # Also log the first sample to stdout for immediate visibility
            for i in range(min(2, len(uids))):
                _logger.info(f"[SDPO DEBUG] step={self.global_step} sample={i} uid={uids[i]}")
                _logger.info(f"[SDPO DEBUG] HINT: {self._uid_to_hint.get(uids[i], '')[:300]}")
                _logger.info(f"[SDPO DEBUG] PROMPT: {self.tokenizer.decode(prompt_ids_list[i])[:500]}")
                _logger.info(f"[SDPO DEBUG] RESPONSE: {self.tokenizer.decode(response_ids_list[i])[:500]}")
                _logger.info(f"[SDPO DEBUG] TEACHER_PROMPT: {self.tokenizer.decode(teacher_prompts[i])[:500]}")
                _logger.info(f"[SDPO DEBUG] REWARD: {generator_output.get('rewards', [['?']])[i]}")
                _logger.info(f"[SDPO DEBUG] STOP_REASON: {generator_output.get('stop_reasons', [''])[i]}")

        return training_input

    def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch):
        fwd_keys = ["sequences", "attention_mask"]
        data_fwd_pass = training_input.select(keys=fwd_keys, metadata_keys=["response_length"])

        action_log_probs = None
        if not self._skip_policy_forward(training_input):
            action_log_probs = self._execute_forward_pass(
                "policy",
                data_fwd_pass,
                key="logprobs",
                mini_batch_boundaries=training_input.metadata.get("policy_mini_batch_boundaries"),
            )
        self.dispatch.empty_cache()

        base_log_probs = None
        if self.ref_model is not None:
            teacher_data = TrainingInputBatch({
                "sequences": training_input["teacher_sequences"],
                "attention_mask": training_input["teacher_attention_mask"],
            })
            teacher_data.metadata = {"response_length": training_input.metadata["response_length"]}
            base_log_probs = self._execute_forward_pass(
                "ref", teacher_data, key="logprobs", mini_batch_boundaries=None
            )
            self.dispatch.empty_cache("ref")

        sequences_all = training_input["sequences"]
        base_log_probs = base_log_probs[: len(sequences_all)] if base_log_probs is not None else None
        action_log_probs = action_log_probs[: len(sequences_all)] if action_log_probs is not None else None

        training_input["base_action_log_probs"] = base_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = None

        # ─── DEBUG: log teacher vs student log-probs ───────────────────
        if self.global_step < 3 and action_log_probs is not None and base_log_probs is not None:
            from loguru import logger as _logger
            loss_mask = training_input["loss_mask"]
            for i in range(min(2, len(action_log_probs))):
                # Find first 10 tokens where loss_mask=1
                mask_i = loss_mask[i]
                valid_idx = (mask_i > 0).nonzero(as_tuple=True)[0]
                if len(valid_idx) == 0:
                    _logger.info(f"[SDPO DEBUG] step={self.global_step} sample={i}: NO valid tokens in loss_mask!")
                    continue
                first_tokens = valid_idx[:10]
                student_lps = action_log_probs[i, first_tokens].tolist()
                teacher_lps = base_log_probs[i, first_tokens].tolist()
                _logger.info(f"[SDPO DEBUG] step={self.global_step} sample={i}: first 10 token log-probs:")
                _logger.info(f"[SDPO DEBUG]   student: {[round(x,3) for x in student_lps]}")
                _logger.info(f"[SDPO DEBUG]   teacher: {[round(x,3) for x in teacher_lps]}")
                _logger.info(f"[SDPO DEBUG]   diff:    {[round(s-t,3) for s,t in zip(student_lps, teacher_lps)]}")
                # Decode those tokens
                seq_i = training_input["sequences"][i]
                resp_len = training_input.metadata["response_length"]
                token_ids = seq_i[-(resp_len):][first_tokens].tolist()
                _logger.info(f"[SDPO DEBUG]   tokens:  {self.tokenizer.decode(token_ids)}")
                # Stats over all valid tokens
                all_student = action_log_probs[i][mask_i > 0]
                all_teacher = base_log_probs[i][mask_i > 0]
                _logger.info(f"[SDPO DEBUG]   student logp: mean={all_student.mean().item():.3f} std={all_student.std().item():.3f}")
                _logger.info(f"[SDPO DEBUG]   teacher logp: mean={all_teacher.mean().item():.3f} std={all_teacher.std().item():.3f}")
                _logger.info(f"[SDPO DEBUG]   diff mean={((all_student - all_teacher).mean()).item():.3f} max={((all_student - all_teacher).abs().max()).item():.3f}")
                # loss_mask stats
                _logger.info(f"[SDPO DEBUG]   loss_mask sum: {mask_i.sum().item()}, shape: {mask_i.shape}")
                # Also check if teacher_sequences and student sequences have same response tokens
                teacher_seq_i = training_input["teacher_sequences"][i]
                student_seq_i = training_input["sequences"][i]
                resp_tokens_student = student_seq_i[-resp_len:].tolist()
                resp_tokens_teacher = teacher_seq_i[-resp_len:].tolist()
                match = resp_tokens_student == resp_tokens_teacher
                _logger.info(f"[SDPO DEBUG]   response tokens match between student/teacher seq: {match}")
                if not match:
                    n_diff = sum(1 for a,b in zip(resp_tokens_student, resp_tokens_teacher) if a != b)
                    _logger.info(f"[SDPO DEBUG]   {n_diff} tokens differ out of {resp_len}")

        return training_input

    def apply_reward_kl_penalty(self, data: TrainingInputBatch) -> TrainingInputBatch:
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_log_probs: torch.Tensor = data["base_action_log_probs"]
        data["rewards"] = teacher_log_probs * loss_masks_all
        return data


class SDPOSyncTrainer(SDPOTrainerMixin, RayPPOTrainer):
    """Sync SDPO trainer (K=0)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdpo_init()


class SDPOAsyncTrainer(SDPOTrainerMixin, FullyAsyncRayPPOTrainer):
    """Fully-async SDPO trainer (K>0)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdpo_init()


# ─── Entrypoint ──────────────────────────────────────────────────────────

class SDPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        cfg = kwargs.get("cfg", args[0] if args else None)
        if cfg is not None and cfg.trainer.fully_async.max_staleness_steps > 0:
            return SDPOAsyncTrainer(*args, **kwargs)
        return SDPOSyncTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = SDPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SDPOSkyRLConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
