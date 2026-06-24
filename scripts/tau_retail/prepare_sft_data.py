"""Prepare SFT dataset from tau-retail tasks.

Creates a parquet with 'messages' column where each row is:
  [{"role": "system", "content": <retail wiki + tool instructions>},
   {"role": "user", "content": <task instruction>},
   {"role": "assistant", "content": <JSON tool call 1>},
   {"role": "user", "content": <tool output 1>},
   {"role": "assistant", "content": <JSON tool call 2>},
   ...]

The assistant messages are the canonical actions formatted as JSON tool calls.
The user messages (tool outputs) are simulated by actually running the actions
against a fresh tau-retail env instance.
"""

import argparse
import json
import os

import pandas as pd


def build_system_prompt():
    """Build the system prompt matching our env's format."""
    from tau_bench.envs.retail import MockRetailDomainEnv

    env = MockRetailDomainEnv(
        user_strategy="trivial",
        user_model="trivial",
        task_split="test",
        task_index=0,
    )
    wiki = env.wiki
    tools_info = env.tools_info

    tool_lines = []
    for t in tools_info:
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
        f"{wiki}\n\n"
        "You are a retail customer service agent.\n\n"
        "At each turn, output ONLY a single JSON object representing your next action:\n"
        '{"name": "<tool_name>", "arguments": {<arg_name>: <value>, ...}}\n\n'
        "Do not output anything other than the JSON object.\n\n"
        "Available tools:\n" + "\n".join(tool_lines) + "\n\n"
        "When you have completed the task, call the \"transfer_to_human_agents\" tool to end."
    )


def build_sft_data():
    """Build SFT conversations from tau-retail tasks with real tool outputs."""
    from tau_bench.envs.retail import MockRetailDomainEnv
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from tau_bench.envs.user import UserStrategy
    import tau_bench.envs.user as tau_user
    import tau_bench.envs.base as tau_base

    # Patch trivial user
    class TrivialUserSim(tau_user.BaseUserSimulationEnv):
        def reset(self, instruction=None):
            return instruction or ""
        def step(self, content):
            return "###STOP###"
        def get_total_cost(self):
            return 0.0

    _orig_load_user = tau_user.load_user
    def _patched_load_user(user_strategy, model=None, provider=None):
        us = user_strategy
        if isinstance(us, str) and us == "trivial":
            return TrivialUserSim()
        if hasattr(us, "value") and us.value == "trivial":
            return TrivialUserSim()
        return _orig_load_user(us, model, provider)
    tau_user.load_user = _patched_load_user
    tau_base.load_user = _patched_load_user

    system_prompt = build_system_prompt()
    rows = []

    for task_idx, task in enumerate(TASKS_TEST):
        try:
            env = MockRetailDomainEnv(
                user_strategy="trivial",
                user_model="trivial",
                task_split="test",
                task_index=task_idx,
            )
            env.reset(task_index=task_idx)

            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": task.instruction})

            # Execute canonical actions and record the conversation
            from tau_bench.types import Action
            success = True
            for action in task.actions:
                # Format the action as JSON (what the model should generate)
                action_json = json.dumps({"name": action.name, "arguments": action.kwargs})
                messages.append({"role": "assistant", "content": action_json})

                # Execute the action to get the tool output
                try:
                    resp = env.tau_env.step(action)
                    if resp.done:
                        break
                    messages.append({"role": "user", "content": str(resp.observation)})
                except Exception as e:
                    messages.append({"role": "user", "content": f"Error: {e}"})
                    success = False
                    break

            if success:
                rows.append({"messages": messages, "task_index": task_idx})
        except Exception as e:
            print(f"Task {task_idx} failed: {e}")
            continue

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/data/tau_retail_sft")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = build_sft_data()
    print(f"Built {len(rows)} SFT examples")

    df = pd.DataFrame(rows)
    output_path = os.path.join(args.output_dir, "tau_retail_sft.parquet")
    df.to_parquet(output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
