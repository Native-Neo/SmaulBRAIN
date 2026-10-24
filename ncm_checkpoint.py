"""Safetensors checkpoint helpers for the NumPy NCM runtime.

The training and inference entry points keep their public paths, but the files
are safetensors containers instead of PyTorch pickles. Model tensors live under
``model::`` names; optimizer moment tensors live under ``optim::`` names. Small
JSON metadata (config, step, optimizer hyperparameters) is stored in the
safetensors header.
"""
from __future__ import annotations

import json
import os

import numpy as np

from smaulbrain import safetensors_np as _st
from smaulbrain.npgrad import Tensor

_MODEL_PREFIX = "model::"
_OPTIM_PREFIX = "optim::"


def _array(value):
    if isinstance(value, Tensor):
        value = value.data
    return np.ascontiguousarray(np.asarray(value))


def _torch_pickle(path):
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _read(path):
    path = os.fspath(path)
    if _torch_pickle(path):
        raise ValueError(
            f"{path} is a legacy PyTorch checkpoint; convert it with "
            "tests/_reference/convert_torch_checkpoint.py"
        )
    return _st.load_file(path)


def save_model_checkpoint(path, config, model):
    """Save only model weights plus the NCM config."""
    tensors = {
        _MODEL_PREFIX + name: _array(value)
        for name, value in model.state_dict().items()
    }
    metadata = {
        "kind": "ncm-model",
        "config": json.dumps(config.__dict__, separators=(",", ":")),
    }
    _st.save_file(tensors, path, metadata=metadata)
    return os.fspath(path)


def save_training_checkpoint(path, config, model, optimizer, step):
    """Save model, SmaulOPT state, config and global step atomically."""
    tensors = {
        _MODEL_PREFIX + name: _array(value)
        for name, value in model.state_dict().items()
    }
    opt_state = optimizer.state_dict()
    for name, state in opt_state["state"].items():
        for key, value in state.items():
            tensors[f"{_OPTIM_PREFIX}{name}::{key}"] = _array(value)
    opt_meta = {
        "factored": opt_state.get("factored", {}),
        "n_updates": int(opt_state.get("n_updates", 0)),
        "hyperparameters": opt_state.get("hyperparameters", {}),
    }
    metadata = {
        "kind": "ncm-train",
        "config": json.dumps(config.__dict__, separators=(",", ":")),
        "step": str(int(step)),
        "optimizer": json.dumps(opt_meta, separators=(",", ":")),
    }
    _st.save_file(tensors, path, metadata=metadata)
    return os.fspath(path)


def load_checkpoint(path):
    """Load a model or training checkpoint.

    Returns a dict with ``config`` (plain dict), ``model`` (state dict),
    ``optimizer`` (SmaulOPT-shaped payload or ``None``) and ``step``.
    """
    tensors, metadata = _read(path)
    if not metadata or "config" not in metadata:
        raise ValueError(f"{path}: missing NCM checkpoint metadata")
    try:
        config = json.loads(metadata["config"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid NCM config metadata") from exc
    model_state = {
        name[len(_MODEL_PREFIX):]: value
        for name, value in tensors.items()
        if name.startswith(_MODEL_PREFIX)
    }
    if not model_state:
        raise ValueError(f"{path}: checkpoint contains no model tensors")
    optimizer = None
    step = 0
    if metadata.get("kind") == "ncm-train":
        opt_meta = json.loads(metadata.get("optimizer", "{}"))
        state = {}
        for name, value in tensors.items():
            if not name.startswith(_OPTIM_PREFIX):
                continue
            param_name, _, key = name[len(_OPTIM_PREFIX):].partition("::")
            if not param_name or not key:
                raise ValueError(f"{path}: malformed optimizer tensor name {name!r}")
            state.setdefault(param_name, {})[key] = value
        optimizer = {
            "type": "SmaulOPT",
            "state": state,
            "factored": opt_meta.get("factored", {}),
            "n_updates": int(opt_meta.get("n_updates", 0)),
            "hyperparameters": opt_meta.get("hyperparameters", {}),
        }
        step = int(metadata.get("step", 0))
    return {"config": config, "model": model_state, "optimizer": optimizer, "step": step}
