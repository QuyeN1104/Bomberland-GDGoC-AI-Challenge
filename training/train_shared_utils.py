"""Shared training utilities used across scripts.

Keep this module lightweight and dependency-free so it can be imported from
multiple training entrypoints without circular imports.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def csv_append(path: str, fieldnames: list[str], row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        safe_row = {k: row.get(k, "") for k in fieldnames}
        w.writerow(safe_row)


def save_checkpoint(path: str, model, optimizer, meta: dict, agent_type: str, num_actions: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(meta)
    meta.setdefault("num_actions", int(num_actions))
    payload = {
        "model_state_dict": model.state_dict(),
        "meta": meta,
        "agent_type": str(agent_type),
        "num_actions": int(meta["num_actions"]),
    }
    if meta.get("input_spec") is not None:
        payload["input_shape"] = meta["input_spec"]
        payload["input_spec"] = meta["input_spec"]
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, model, device: str, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}

