"""Run an mlx-lm model with Drift-managed layer weights (start-prompt-v2 §6.7).

The model architecture comes from mlx-lm; only the weights of the decoder layers are managed
here. MLX evaluates lazily, so without an explicit `mx.eval` per streamed layer the graph of
all layers would need all weights at once. Each streamed layer therefore: acquire weights →
compute → eval → hand the weights back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from ..mlx_params import flat_params
from ..runtime.errors import EngineError
from .experts import ExpertCache
from .residency import DenseResidency


class DriftLayer:
    """Stands in for a decoder layer inside the model's `layers` list."""

    def __init__(self, inner: nn.Module, layer: int, residency: DenseResidency) -> None:
        self.inner = inner
        self.layer = layer
        self.residency = residency
        self._placeholders = tree_unflatten(
            [(k, mx.zeros((1,), dtype=v.dtype)) for k, v in flat_params(inner)]
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        params = self.residency.acquire(self.layer)
        if self.layer in self.residency.pinned:
            self.inner.update(tree_unflatten(list(params.items())))
            return self.inner(*args, **kwargs)
        try:
            self.inner.update(tree_unflatten(list(params.items())))
            out = self.inner(*args, **kwargs)
            mx.eval(out)  # weights must be consumed before their slot is reused
            return out
        finally:
            self.inner.update(self._placeholders)
            self.residency.release(self.layer)

    def __getattr__(self, name: str) -> Any:  # e.g. cache construction peeking at attributes
        return getattr(self.inner, name)


def find_layers(model: nn.Module) -> list[Any]:
    for owner in (
        getattr(model, "model", None),
        model,
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        layers = getattr(owner, "layers", None) if owner is not None else None
        if isinstance(layers, list) and layers and isinstance(layers[0], nn.Module):
            return layers
    raise EngineError(
        "drift_unsupported",
        "Dit model wordt nog niet ondersteund door Drift.",
        hint="De lagen van dit model konden niet gevonden worden.",
        status=409,
    )


def build_drift_model(path: str, residency: DenseResidency) -> nn.Module:
    """Instantiate the architecture, load only the core weights, wrap every decoder layer."""
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(path), lazy=True)
    layers = find_layers(model)
    if len(layers) != len(residency.order):
        raise EngineError(
            "drift_unsupported", "Aantal lagen klopt niet met de modelbestanden.", status=409
        )

    for i, layer in enumerate(layers):
        expected = {k for k, _ in flat_params(layer)}
        have = {t.name for t in residency.units[residency.order[i]].tensors}
        if expected != have:
            missing = sorted(expected - have)[:3]
            raise EngineError(
                "drift_unsupported",
                "De modelbestanden passen niet bij de architectuur voor Drift.",
                hint=f"Ontbrekend in bestanden: {missing}",
                status=409,
            )

    # Wrap before evaluating anything so the lazy layer weights are never read from disk.
    for i, layer in enumerate(layers):
        layers[i] = DriftLayer(layer, residency.order[i], residency)
    core = [v for _, v in flat_params(model)]  # layers are no longer Modules
    mx.eval(core)
    return model


class DriftSwitch(nn.Module):
    """Replaces a layer's SwitchGLU: routed experts come from the expert cache.

    The router's choices must be known before experts can be loaded, so the indices are
    evaluated here (one sync point per MoE layer). Decode uses the layer's expert slots;
    prefill uses a transient full-size buffer (see experts.py).
    """

    def __init__(self, inner: nn.Module, layer: int, cache: ExpertCache, n_layers: int) -> None:
        super().__init__()
        self.inner = inner
        self._layer = layer
        self._cache = cache
        self._n_layers = n_layers
        self._first = layer == cache.order[0]
        self._last = layer == cache.order[-1]
        self._next: tuple[int, Any] | None = None  # (next MoE layer, its MoE block) for prefetch
        self._gate: Any = None  # this block's router, for Turbo (routers that return weights)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        cache = self._cache
        decode = indices.size == indices.shape[-1]  # one token
        share = cache.skip_share() if self._gate is not None else 0.0
        weights = None
        if share > 0:
            try:  # Turbo needs the routing weights: computed again here, the router is tiny.
                weights = _routing(self._gate, x, 1)[1]
            except ValueError:  # the router does not take the experts' input (latent MoE)
                self._gate = None
        pending = [indices] if weights is None else [indices, weights]
        predicted = None
        if decode and self._next is not None:
            # The next layer's router applied to this layer's input: the residual stream
            # changes little per layer, so this predicts most of its experts.
            try:
                predicted = _routing(self._next[1].gate, x, cache.plan.experts_per_token)
                pending += [a for a in predicted if a is not None]
            except ValueError:  # as above: no prediction for this model
                self._next = None
        mx.eval(pending)
        idx = np.array(indices)
        if decode and self._first:
            cache.tick()
        cache.stats.update("drift", current_layer=self._layer)
        keep = None
        if weights is not None:
            w = np.array(weights.astype(mx.float32))
            keep = cache.keep_mask(self._layer, idx, w, share)
        needed = np.unique(idx if keep is None else idx[keep])
        if decode:
            prefetch = None
            if predicted is not None and self._next is not None:
                guess = np.array(predicted[0])
                if share > 0 and predicted[1] is not None:  # Turbo: only what would be kept
                    pw = np.array(predicted[1].astype(mx.float32))
                    guess = guess[pw / pw.sum(axis=-1, keepdims=True) >= share]
                prefetch = (self._next[0], np.unique(guess))
            mapping = cache.ensure(self._layer, needed, decode=True, prefetch=prefetch)
            out = self.inner(x, mapping[indices])
        else:
            # Prefill: original indices on a full-size buffer (bit-identical to resident).
            self.inner.update(
                tree_unflatten(list(cache.prefill_weights(self._layer, needed).items()))
            )
            out = self.inner(x, indices)
            mx.eval(out)  # the prefill buffer is reused by the next layer
            self.inner.update(tree_unflatten(list(cache.layers[self._layer].tensors.items())))
        if keep is not None and not keep.all():
            # Skipped choices contribute nothing; the token's other experts carry its whole
            # routed weight (as if it had been routed to fewer experts). where(), not ×0: a
            # skipped choice ran on whatever weights its row held.
            scale = np.where(keep, w.sum(-1, keepdims=True) / (w * keep).sum(-1, keepdims=True), 0)
            scaled = out * mx.array(scale[..., None]).astype(out.dtype)
            out = mx.where(mx.array(keep[..., None]), scaled, 0)
        if self._last:
            cache.publish(self._layer, self._n_layers)
        return out


def _routing(gate: Any, x: mx.array, k: int) -> tuple[mx.array, mx.array | None]:
    """(experts, weights) the router picks for `x`; weights only for routers that return them
    (DeepSeek/GLM style). Plain linear routers give logits: their top-k, without weights."""
    out = gate(x)
    if isinstance(out, tuple):
        return out[0], out[1]
    return mx.argpartition(-out, kth=k - 1, axis=-1)[..., :k], None


def build_moe_model(path: str, cache: ExpertCache, progress: Any = None) -> nn.Module:
    """Architecture from mlx-lm; resident weights loaded now, routed experts via the cache."""
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(path), lazy=True)
    layers = find_layers(model)
    switches: list[tuple[int, Any, DriftSwitch]] = []
    for layer_idx, slots in sorted(cache.layers.items()):
        owner: Any = layers[layer_idx]
        *parents, leaf = slots.plan.module_path.split(".")
        for p in parents:
            owner = getattr(owner, p)
        inner = getattr(owner, leaf)
        expected = {k for k, _ in flat_params(inner)}
        if expected != set(slots.tensors):
            raise EngineError(
                "drift_unsupported",
                "De experts in de modelbestanden passen niet bij de architectuur.",
                hint=f"Verwacht {sorted(expected)[:4]}",
                status=409,
            )
        if any(getattr(m, "global_scale", None) is not None for m in inner.children().values()):
            raise EngineError(
                "drift_unsupported",
                "Experts met een eigen schaal (nvfp4) worden nog niet ondersteund door Drift.",
                status=409,
            )
        # Slot tensors replace the lazy file-backed expert weights before anything is read.
        inner.update(tree_unflatten(list(slots.tensors.items())))
        switch = DriftSwitch(inner, layer_idx, cache, len(layers))
        setattr(owner, leaf, switch)
        switches.append((layer_idx, owner, switch))

    # Each switch knows the next MoE block, for speculative prefetch.
    for (_, _, switch), (next_idx, next_block, _) in zip(switches, switches[1:], strict=False):
        if hasattr(next_block, "gate"):
            switch._next = (next_idx, next_block)
    for _, block, switch in switches:
        if hasattr(block, "gate"):
            switch._gate = block.gate

    # Everything still lazy now is the resident part: materialise it with progress.
    params = [v for _, v in flat_params(model)]
    total = sum(v.nbytes for v in params) or 1
    done, batch, batch_bytes = 0, [], 0
    for v in params:
        batch.append(v)
        batch_bytes += v.nbytes
        if batch_bytes >= 512 * 1024**2:
            mx.eval(batch)
            done += batch_bytes
            batch, batch_bytes = [], 0
            if progress:
                progress(done / total)
    mx.eval(batch)
    cache.publish(None, len(layers))
    return model
