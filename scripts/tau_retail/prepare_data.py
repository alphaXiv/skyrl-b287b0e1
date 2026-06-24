"""Prepare tau-retail tasks as parquet datasets for SkyRL training.

Generates train and eval parquet files where each row has:
  - prompt: [{"role": "system", ...}, {"role": "user", "content": <instruction>}]
  - env_class: "tau_retail"
  - hint: serialized canonical action plan (Task.actions)
  - task_index: int

Usage:
    python scripts/tau_retail/prepare_data.py --output-dir /root/data/tau_retail
"""

import argparse
import json
import os

import pandas as pd


def serialize_hint(task) -> str:
    """Format canonical actions as model-output JSON tool calls.

    Each action becomes a JSON object the model would generate:
      {"name": "tool_name", "arguments": {...}}
    Separated by newlines. This matches the model's output format so the
    teacher's log-probs on student tokens are reasonable (no format mismatch).
    """
    lines = []
    for action in task.actions:
        kwargs_str = json.dumps(action.kwargs)
        lines.append(json.dumps({"name": action.name, "arguments": action.kwargs}))
    if task.outputs:
        lines.append(json.dumps({"name": "respond", "arguments": {"content": task.outputs[0] if task.outputs else ""}}))
    return "\n".join(lines)


def build_split(tasks, split_name):
    rows = []
    for idx, task in enumerate(tasks):
        rows.append({
            "prompt": [
                {"role": "system", "content": "retail"},
                {"role": "user", "content": task.instruction},
            ],
            "env_class": "tau_retail",
            "hint": serialize_hint(task),
            "task_index": idx,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/data/tau_retail")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from tau_bench.envs.retail.tasks_dev import TASKS_DEV

    train_rows = build_split(TASKS_TEST, "test")
    eval_rows = build_split(TASKS_DEV, "dev")

    for name, rows in [("train", train_rows), ("eval", eval_rows)]:
        path = os.path.join(args.output_dir, f"tau_retail_{name}.parquet")
        df = pd.DataFrame(rows)
        df.to_parquet(path)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
