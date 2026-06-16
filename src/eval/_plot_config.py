"""Plotting config: model list, colors, display-name overrides, step cutoff.

Load from a JSON file with ``PlotConfig.load(path)``; the schema is::

    {
      "models": [
        {"name": "BERT-tiny", "color": "#0d9488", "display_name": "BERT-small"},
        {"name": "OmniLearned-small", "color": "#ff7f0e"}
      ],
      "step_cutoff": null,
      "flops_xmin": null
    }

``display_name`` is optional per-model; when omitted :func:`plot_model_label` is used.
``step_cutoff`` (int or null) clips the x-axis of the log-steps plot only.
``flops_xmin`` (float or null) sets the minimum x-axis on log-FLOPs plots (``log10`` FLOPs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelEntry:
    name: str
    color: str
    display_name: str | None = None


@dataclass
class PlotConfig:
    models: list[ModelEntry] = field(default_factory=list)
    step_cutoff: int | None = None
    flops_xmin: float | None = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def colors(self) -> dict[str, str]:
        return {m.name: m.color for m in self.models}

    def label_for(self, name: str) -> str:
        from src.eval._constants import plot_model_label

        for m in self.models:
            if m.name == name and m.display_name is not None:
                return m.display_name
        return plot_model_label(name)

    def filter_dict(self, d: dict) -> dict:
        """Return a copy of *d* with only keys present in model_names()."""
        names = set(self.model_names())
        return {k: v for k, v in d.items() if k in names}

    def filter_dict_ordered(
        self,
        d: dict,
        tail: tuple[str, ...] = ("Transformer-xsmall", "Transformer-small", "MLP"),
    ) -> dict:
        """Like :meth:`filter_dict`, preserving :meth:`ordered_model_names` key order."""
        filtered = self.filter_dict(d)
        return {k: filtered[k] for k in self.ordered_model_names(tail) if k in filtered}

    def filter_nested(self, d: dict[str, Any]) -> dict[str, Any]:
        """Filter one level deep: ``{"Log1p": {model: ...}}`` → keep model keys."""
        out: dict[str, Any] = {}
        names = set(self.model_names())
        for loss, inner in d.items():
            if isinstance(inner, dict):
                out[loss] = {k: v for k, v in inner.items() if k in names}
            else:
                out[loss] = inner
        return out

    def ordered_model_names(
        self,
        tail: tuple[str, ...] = ("Transformer-xsmall", "Transformer-small", "MLP"),
    ) -> list[str]:
        """Config model order with *tail* names moved to the end (legend / draw order)."""
        tail_set = set(tail)
        names = self.model_names()
        ordered = [n for n in names if n not in tail_set]
        ordered.extend(n for n in tail if n in names)
        return ordered

    def filter_nested_ordered(
        self,
        d: dict[str, Any],
        tail: tuple[str, ...] = ("Transformer-xsmall", "Transformer-small", "MLP"),
    ) -> dict[str, Any]:
        """Like :meth:`filter_nested`, preserving :meth:`ordered_model_names` key order."""
        filtered = self.filter_nested(d)
        order_idx = {n: i for i, n in enumerate(self.ordered_model_names(tail))}
        out: dict[str, Any] = {}
        for loss, inner in filtered.items():
            if isinstance(inner, dict):
                keys = sorted(inner.keys(), key=lambda k: order_idx.get(k, len(order_idx)))
                out[loss] = {k: inner[k] for k in keys}
            else:
                out[loss] = inner
        return out

    def legend_labels(
        self,
        tail: tuple[str, ...] = ("Transformer-xsmall", "Transformer-small", "MLP"),
    ) -> list[str]:
        """Display labels in :meth:`ordered_model_names` order."""
        return [self.label_for(n) for n in self.ordered_model_names(tail)]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "PlotConfig":
        with open(path) as f:
            data = json.load(f)
        models = [
            ModelEntry(
                name=m["name"],
                color=m["color"],
                display_name=m.get("display_name"),
            )
            for m in data.get("models", [])
        ]
        return cls(
            models=models,
            step_cutoff=data.get("step_cutoff"),
            flops_xmin=data.get("flops_xmin"),
        )
