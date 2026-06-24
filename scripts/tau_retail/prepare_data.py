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
    lines = ["Reference solution actions:"]
    for i, action in enumerate(task.actions, 1):
        kwargs_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in action.kwargs.items())
        lines.append(f"{i}. {action.name}({kwargs_str})")
    if task.outputs:
        lines.append(f"\nExpected outputs: {json.dumps(task.outputs)}")
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
