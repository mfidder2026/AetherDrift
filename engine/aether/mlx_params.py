"""Typed access to a module's parameters (mlx.utils.tree_flatten is typed loosely)."""

from __future__ import annotations

from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def flat_params(module: nn.Module) -> list[tuple[str, mx.array]]:
    return cast(list[tuple[str, mx.array]], tree_flatten(module.parameters()))
