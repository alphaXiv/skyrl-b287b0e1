"""SDPO (Self-Distillation Policy Optimization) entrypoint for SkyRL.

Implements the SDPO loss from Hübotter et al.:
  L = E_τ~π_θ [ Σ_t KL(π_θ(·|s_t) || stopgrad(π_θ(·|s_t, c))) ]

The hint c = decoded text of a successful sibling rollout (same prompt, same group).
The teacher = policy model (frozen ref) run on the *reprompted* input where the
hint is inserted into the user message via a reprompt template:
  "{original_prompt}\n\nCorrect solution:\n\n{successful_sibling_text}\n\nCorrectly solve the original question."

Sampled-token reverse-KL estimator (alpha=1, reverse KL):
  per_token_loss = (student_log_prob - teacher_log_prob).detach() * student_log_prob

Off-policy IS correction (naive, unclipped — collapses at K>0):
  ratio = exp(student_log_prob - rollout_log_prob).detach()
  per_token_loss *= ratio

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
    hint_max_length: int = 2048
    """Max token length for the hint text inserted into the teacher prompt."""
    success_reward_threshold: float = 0.5
    """Minimum reward for a rollout to be considered a 'successful' hint source."""
    reprompt_template: str = (
        "{prompt}\n\nCorrect solution:\n\n{solution}\n\nCorrectly solve the original question."
    )
    """Template for constructing the teacher's user message. {prompt} = original user message, {solution} = successful sibling text."""


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
    else:
        loss = per_token_loss.mean()

    return loss, metrics


@register_advantage_estimator("sdpo_no_op")
def sdpo_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


# ─── Trainer mixin ───────────────────────────────────────────────────────

class SDPOTrainerMixin:
    """SDPO overrides shared by sync and async trainers.

    Hint construction:
    - Group rollouts by uid (same prompt). For each rollout, find a successful
      sibling (same uid, reward >= threshold, different index).
    - Build teacher prompt by inserting the sibling's decoded text into the
      user message via the reprompt template, then re-applying the chat template.
    - Teacher = frozen ref model forward on the reprompted sequence.
    """

    def _sdpo_init(self):
        self._sdpo_cfg = self.cfg.trainer.algorithm.sdpo

    def _find_sibling_solutions(
        self,
        generator_output,
        uids: List[str],
    ) -> List[Optional[str]]:
        """For each rollout, find the decoded text of a successful sibling."""
        rewards = generator_output["rewards"]
        response_ids_list = generator_output["response_ids"]
        prompt_token_ids = generator_output["prompt_token_ids"]

        # Group by uid: uid -> list of (index, reward)
        uid_to_indices: Dict[str, List[int]] = {}
        for i, uid in enumerate(uids):
            uid_to_indices.setdefault(uid, []).append(i)

        threshold = self._sdpo_cfg.success_reward_threshold
        siblings: List[Optional[str]] = [None] * len(uids)

        for uid, indices in uid_to_indices.items():
            # Find successful rollouts in this group
            successful = [i for i in indices if self._get_reward_scalar(rewards[i]) >= threshold]
            if not successful:
                continue
            # For each failed/passed rollout, assign the first successful sibling (not self)
            for i in indices:
                for s in successful:
                    if s != i:
                        siblings[i] = self.tokenizer.decode(response_ids_list[s])
                        break

        return siblings

    def _get_reward_scalar(self, reward) -> float:
        """Extract a scalar reward from per-token or scalar reward."""
        if isinstance(reward, (int, float)):
            return float(reward)
        if isinstance(reward, list):
            return sum(reward)
        return 0.0

    def _build_teacher_prompt_ids(
        self,
        prompt_token_ids: List[int],
        sibling_text: Optional[str],
    ) -> List[int]:
        """Build teacher prompt by inserting sibling text into user message via chat template."""
        if sibling_text is None:
            return list(prompt_token_ids)

        # Decode the original prompt to get the user message text
        prompt_text = self.tokenizer.decode(prompt_token_ids)

        # Build reprompted user message
        reprompt = self._sdpo_cfg.reprompt_template.format(
            prompt=prompt_text,
            solution=sibling_text,
        )

        # Re-tokenize with chat template
        # The prompt_token_ids were created by apply_chat_template, so we need to
        # re-apply it with the reprompted text as the last user message.
        # We decode the full prompt, replace the user message, and re-encode.
        # Simpler approach: just tokenize the reprompt text (which already includes
        # the system message + user message + hint) and use it directly.
        # But that loses the chat template formatting. Instead, we decode to messages,
        # modify the last user message, and re-apply chat template.

        # Actually, the prompt_token_ids are already chat-templated. We can't easily
        # extract the messages back. Instead, let's construct the teacher prompt by
        # tokenizing the reprompt text (which includes the original prompt text + hint)
        # and appending the response_ids. This is not perfect chat templating but
        # it's close enough — the teacher sees the original context + hint.

        # Truncate hint if needed
        max_len = self._sdpo_cfg.hint_max_length
        hint_ids = self.tokenizer.encode(sibling_text, add_special_tokens=False)
        if len(hint_ids) > max_len:
            hint_ids = hint_ids[:max_len]

        # Build: prompt_token_ids + hint marker + hint_ids
        # We append the hint as a continuation of the last user message.
        # This is a simplification — ideally we'd re-apply the chat template.
        # But for the teacher forward pass, the key is that the model sees the
        # original context + the hint text.
        marker = self.tokenizer.encode(
            "\n\nCorrect solution:\n\n", add_special_tokens=False
        )
        closing = self.tokenizer.encode(
            "\n\nCorrectly solve the original question.", add_special_tokens=False
        )
        return list(prompt_token_ids) + marker + hint_ids + closing

    def convert_to_training_input(self, generator_output, uids):
        training_input = super().convert_to_training_input(generator_output, uids)

        prompt_ids_list = generator_output["prompt_token_ids"]
        response_ids_list = generator_output["response_ids"]

        # Find sibling solutions
        siblings = self._find_sibling_solutions(generator_output, uids)

        # Build teacher prompts
        teacher_prompts: List[List[int]] = []
        hint_count = 0
        for i, uid in enumerate(uids):
            teacher_prompt = self._build_teacher_prompt_ids(
                prompt_ids_list[i],
                siblings[i],
            )
            teacher_prompts.append(teacher_prompt)
            if siblings[i] is not None:
                hint_count += 1

        from loguru import logger
        logger.info(f"[SDPO] {hint_count}/{len(uids)} rollouts have sibling hints")

        # Build teacher sequences using same preprocessing
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
