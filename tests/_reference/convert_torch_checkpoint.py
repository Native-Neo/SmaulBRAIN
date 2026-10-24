#!/usr/bin/env python3
"""Convert legacy PyTorch SmaulBRAIN checkpoints to the NumPy formats.

Usage:
    python tests/_reference/convert_torch_checkpoint.py INPUT OUTPUT

For an NCM ``.pt`` file, OUTPUT is a ``.safetensors`` file. For a legacy
core checkpoint directory (one containing ``manifest.json`` and
``params.pt``), OUTPUT is a new checkpoint directory.

This is a development-only tool: it imports PyTorch and is intentionally kept
outside the runtime package.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tree(value):
    if torch.is_tensor(value):
        return _array(value)
    if isinstance(value, dict):
        return {k: _tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree(v) for v in value)
    return value


class _ModelShim:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class _OptimizerShim:
    def __init__(self, payload):
        self._payload = payload

    def state_dict(self):
        return self._payload


def _convert_ncm(source: Path, target: Path) -> None:
    from ncm_checkpoint import save_model_checkpoint, save_training_checkpoint

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError(f"{source}: not an NCM checkpoint")
    config = types.SimpleNamespace(**payload["config"])
    model_state = {k: _array(v) for k, v in payload["model"].items()}
    if "optimizer" not in payload:
        save_model_checkpoint(target, config, _ModelShim(model_state))
        print(f"converted model checkpoint -> {target}")
        return

    optimizer = _tree(payload["optimizer"])
    if not (isinstance(optimizer, dict) and optimizer.get("type") == "SmaulOPT"):
        print(
            "warning: legacy optimizer is not SmaulOPT; writing model weights "
            "with fresh optimizer state",
            file=sys.stderr,
        )
        optimizer = {
            "type": "SmaulOPT",
            "state": {},
            "factored": {},
            "n_updates": int(payload.get("step", 0)),
            "hyperparameters": {},
        }
    save_training_checkpoint(
        target, config, _ModelShim(model_state), _OptimizerShim(optimizer),
        int(payload.get("step", 0)),
    )
    print(f"converted training checkpoint -> {target}")


def _convert_core(source: Path, target: Path) -> None:
    from smaulbrain import checkpoint as _ckpt

    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{source}: no manifest.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    params_path = source / "params.pt"
    optim_path = source / "optim.pt"
    if not params_path.is_file() or not optim_path.is_file():
        raise ValueError(
            f"{source}: expected params.pt and optim.pt (legacy .npz is already readable)"
        )
    params = {k: _array(v) for k, v in torch.load(
        params_path, map_location="cpu", weights_only=True).items()}
    optim = _tree(torch.load(optim_path, map_location="cpu", weights_only=True))
    _ckpt.save_checkpoint(str(target), manifest, params, optim)
    print(f"converted core checkpoint -> {target}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.input.is_dir():
        _convert_core(args.input, args.output)
    else:
        _convert_ncm(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
