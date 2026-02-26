"""
Serialization helpers for safe torch.load compatibility.
"""

import inspect
from typing import Any, Optional

import torch


def safe_torch_load(
    path: str,
    map_location: Optional[str] = None,
    weights_only: Optional[bool] = None,
) -> Any:
    """Call torch.load with optional weights_only when supported."""
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location

    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(path, **kwargs, weights_only=weights_only if weights_only is not None else False)

    return torch.load(path, **kwargs)
