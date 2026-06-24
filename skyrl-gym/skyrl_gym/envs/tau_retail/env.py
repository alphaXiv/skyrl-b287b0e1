"""Tau-retail environment wrapping tau-bench's MockRetailDomainEnv.

Single-agent (no LLM user simulator): the task instruction is the initial user
message, the agent executes tool calls, and calls ``transfer_to_human_agents`` to
end the episode.  Reward is 0/1 from tau-bench's DB-state + output verifier.

The agent communicates actions as JSON objects:
    {"name": "<tool_name>", "arguments": {<kw>: <val>, ...}}
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig
from dataclasses import dataclass

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType


@dataclass
class TauRetailEnvConfig:
    task_split: str = "test"


def _patch_trivial_user():
    """Monkeypatch tau-bench to support a ``trivial`` user strategy (no LLM)."""
    import tau_bench.envs.user as tau_user
    import tau_bench.envs.base as tau_base

    if getattr(tau_user, "_trivial_patched", False):
        return

    class TrivialUserSim(tau_user.BaseUserSimulationEnv):
        def reset(self, instruction: Optional[str] = None) -> str:
            return instruction or ""

        def step(self, content: str) -> str:
            return "###STOP###"

        def get_total_cost(self) -> float:
            return 0.0

    _original_load_user = tau_user.load_user

    def _patched_load_user(user_strategy, model=None, provider=None):
        us = user_strategy
        if isinstance(us, str) and us == "trivial":
            return TrivialUserSim()
        if hasattr(us, "value") and us.value == "trivial":
            return TrivialUserSim()
        return _original_load_user(us, model, provider)

    tau_user.load_user = _patched_load_user
    tau_base.load_user = _patched_load_user
    tau_user._trivial_patched = True


class TauRetailEnv(BaseTextEnv):
    """Wrap tau-bench retail domain as a skyrl_gym text env."""

    def __init__(self, env_config: Union[TauRetailEnvConfig, DictConfig, dict, None] = None, extras: Dict[str, Any] = None):
        super().__init__()
        if extras is None:
            extras = {}
        _patch_trivial_user()

        from tau_bench.envs.retail import MockRetailDomainEnv

        self.task_index = extras.get("task_index", 0)
        self.max_turns = extras.get("max_turns", 15)

        self.tau_env = MockRetailDomainEnv(
            user_strategy="trivial",
            user_model="trivial",
            task_split="test",
            task_index=self.task_index,
        )

        self.tools_info: List[Dict] = self.tau_env.tools_info
        self.wiki: str = self.tau_env.wiki
        self.task = self.tau_env.task
        self.chat_history: ConversationType = []
        self._system_prompt_built = False

    def _build_system_prompt(self) -> str:
        tool_lines = []
        for t in self.tools_info:
            fn = t["function"]
            params = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            param_strs = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                desc = pinfo.get("description", "")[:120]
                req = " (required)" if pname in required else ""
                param_strs.append(f"    {pname}: {ptype}{req} — {desc}")
            tool_lines.append(f"- {fn['name']}: {fn.get('description', '')[:200]}\n  Arguments:\n" + "\n".join(param_strs))

        return (
            f"{self.wiki}\n\n"
            "You are a retail customer service agent.\n\n"
            "At each turn, output ONLY a single JSON object representing your next action:\n"
            '{"name": "<tool_name>", "arguments": {<arg_name>: <value>, ...}}\n\n'
            "Do not output anything other than the JSON object.\n\n"
            "Available tools:\n" + "\n".join(tool_lines) + "\n\n"
            "When you have completed the task, call the \"transfer_to_human_agents\" tool to end."
        )

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        self.tau_env.reset(task_index=self.task_index)
        self.chat_history = []
        system_content = self._build_system_prompt()
        if len(prompt) > 0 and prompt[0]["role"] == "system":
            prompt = prompt[1:]
        prompt = [{"role": "system", "content": system_content}] + prompt
        return prompt, {}

    def _parse_action(self, action_str: str):
        action_str = action_str.strip()
        # Strip markdown code fences
        action_str = re.sub(r"^```(?:json)?\s*", "", action_str)
        action_str = re.sub(r"\s*```$", "", action_str)
        # Find first { ... } JSON object
        match = re.search(r"\{.*\}", action_str, re.DOTALL)
        if match:
            action_str = match.group(0)
        try:
            d = json.loads(action_str)
        except Exception:
            return None
        name = d.get("name")
        kwargs = d.get("arguments", d.get("args", {}))
        if name is None:
            return None
        from tau_bench.types import Action
        return Action(name=name, kwargs=kwargs)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Error: Could not parse action. Output a JSON object: {\"name\": \"...\", \"arguments\": {...}}"
            new_obs = {"role": "user", "content": obs}
            self.chat_history.append(new_obs)
            done = self.turns >= self.max_turns
            return BaseTextEnvStepOutput(
                observations=[new_obs], reward=0.0, done=done, metadata={"parse_error": True}
            )

        try:
            resp = self.tau_env.step(parsed)
        except (KeyError, Exception) as e:
            obs = f"Error: {e}"
            new_obs = {"role": "user", "content": obs}
            self.chat_history.append(new_obs)
            done = self.turns >= self.max_turns
            return BaseTextEnvStepOutput(
                observations=[new_obs], reward=0.0, done=done, metadata={"env_error": str(e)}
            )
        done = resp.done or self.turns >= self.max_turns
        reward = float(resp.reward) if done else 0.0
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=True,
                metadata={"reward_info": resp.info.reward_info.model_dump() if resp.info.reward_info else {}},
            )
        else:
            new_obs = {"role": "user", "content": str(resp.observation)}
            self.chat_history.append(new_obs)
            return BaseTextEnvStepOutput(
                observations=[new_obs], reward=0.0, done=False, metadata={}
            )

    def get_metrics(self) -> Dict[str, Any]:
        return {}

    def close(self):
        pass
