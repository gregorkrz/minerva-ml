"""Plotting config: model list, colors, display-name overrides, step cutoff.

Load from a JSON file with ``PlotConfig.load(path)``; the schema is::

    {
      "eval_split": "test",
      "models": [
        {"name": "BERT-tiny", "color": "#0d9488", "display_name": "BERT-small"},
        {"name": "BDT-binnedW-train", "color": "#ff1493", "split": "train",
         "source_name": "BDT-binnedW", "display_name": "BDT-binnedW (train)"}
      ],
      "step_cutoff": null,
      "flops_xmin": null
    }

``eval_split`` (default ``"test"``) is the dataset split for models without an
explicit ``split`` field.  Per-model ``split`` and ``source_name`` load results
from ``{split}_results/`` under a different checkpoint alias (e.g. train-set
overlay of ``BDT-binnedW`` as ``BDT-binnedW-train``).

``display_name`` is optional per-model; when omitted :func:`plot_model_label` is used.
``step_cutoff`` (int or null) clips every model at this training step (legacy global cap).
``flops_xmin`` (float or null) sets the minimum x-axis on log-FLOPs plots (``log10`` FLOPs);
prefer per-model ``flop_cut`` for curve truncation instead.

Per-model curve truncation (on each entry in ``models``)::

    {"name": "HyperScale-small", "step_cut_policy": "min_val_loss"}
    {"name": "HyperScale-small-rw", "step_cut_match": "HyperScale-small"}
    {"name": "MLP", "step_cut": 50000, "flop_cut": 16.5}
    {"name": "MLP", "classification_horizontal_ref": true}

``classification_horizontal_ref`` — on classification val-loss panels only, draw
this model as a horizontal dashed reference (like BDT) instead of a training curve.
Regression panels are unchanged.

``step_cut`` — explicit max training step (inclusive).
``flop_cut`` — max ``log10(cumulative FLOPs + 1)`` for that model on **FLOPs** panels only.
Scalar or per-task object::

    "flop_cut": 16.55
    "flop_cut": {"classification": 16.55, "regression": 16.8}

``log_step_cut`` — max ``log10(training steps + 1)`` for that model on **steps** panels only.
Same scalar / per-task schema as ``flop_cut``.
``flop_xmin`` — min ``log10(cumulative FLOPs + 1)`` for that model on **FLOPs** panels only.
Same scalar / per-task schema as ``flop_cut``.
``step_cut_policy`` — ``min_val_loss`` stops at the step of lowest mean val loss.
``step_cut_match`` — copy the resolved step cutoff from another model.

Legacy top-level ``curve_end`` is still supported for older configs::

    "curve_end": {
      "policy": "min_val_loss",
      "match": {"HyperScale-small-rw": "HyperScale-small"}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


TaskName = Literal["classification", "regression"]


@dataclass
class TaskScopedCut:
    """Per-task x-scale cut; a scalar applies to both classification and regression."""

    classification: float | None = None
    regression: float | None = None

    @classmethod
    def from_json(cls, raw: Any) -> TaskScopedCut | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            value = float(raw)
            return cls(classification=value, regression=value)
        if isinstance(raw, dict):
            clf = raw.get("classification")
            reg = raw.get("regression")
            return cls(
                classification=float(clf) if clf is not None else None,
                regression=float(reg) if reg is not None else None,
            )
        raise TypeError(
            "flop_cut / log_step_cut must be a number or "
            '{"classification": ..., "regression": ...} object'
        )

    def for_task(self, task: TaskName) -> float | None:
        return self.classification if task == "classification" else self.regression


@dataclass
class CurveEndConfig:
    """Per-model training-curve truncation (stop before overfitting)."""

    policy: str = "none"  # "min_val_loss" | "none"
    match: dict[str, str] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.policy == "min_val_loss" or bool(self.match)


@dataclass
class ModelEntry:
    name: str
    color: str
    display_name: str | None = None
    split: str | None = None
    source_name: str | None = None
    step_cut: float | None = None
    flop_cut: TaskScopedCut | None = None
    flop_xmin: TaskScopedCut | None = None
    log_step_cut: TaskScopedCut | None = None
    step_cut_policy: str | None = None
    step_cut_match: str | None = None
    classification_horizontal_ref: bool = False


@dataclass
class ModelCurveCuts:
    """Per-model training-curve truncation for steps / FLOPs plots."""

    step_cut: dict[str, float] = field(default_factory=dict)
    flop_cut: dict[str, float] = field(default_factory=dict)
    flop_xmin: dict[str, float] = field(default_factory=dict)
    log_step_cut: dict[str, float] = field(default_factory=dict)
    step_cut_policy: dict[str, str] = field(default_factory=dict)
    step_cut_match: dict[str, str] = field(default_factory=dict)
    legacy_global_min_val_loss: bool = False
    legacy_match: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_plot_config(cls, cfg: "PlotConfig", task: TaskName) -> "ModelCurveCuts":
        step_cut: dict[str, float] = {}
        flop_cut: dict[str, float] = {}
        flop_xmin: dict[str, float] = {}
        log_step_cut: dict[str, float] = {}
        step_cut_policy: dict[str, str] = {}
        step_cut_match: dict[str, str] = {}
        for m in cfg.models:
            if m.step_cut is not None:
                step_cut[m.name] = float(m.step_cut)
            if m.flop_cut is not None:
                value = m.flop_cut.for_task(task)
                if value is not None:
                    flop_cut[m.name] = value
            if m.flop_xmin is not None:
                value = m.flop_xmin.for_task(task)
                if value is not None:
                    flop_xmin[m.name] = value
            if m.log_step_cut is not None:
                value = m.log_step_cut.for_task(task)
                if value is not None:
                    log_step_cut[m.name] = value
            if m.step_cut_policy:
                step_cut_policy[m.name] = m.step_cut_policy
            if m.step_cut_match:
                step_cut_match[m.name] = m.step_cut_match
        legacy_global = (
            cfg.curve_end is not None and cfg.curve_end.policy == "min_val_loss"
        )
        legacy_match = (
            dict(cfg.curve_end.match) if cfg.curve_end is not None else {}
        )
        return cls(
            step_cut=step_cut,
            flop_cut=flop_cut,
            flop_xmin=flop_xmin,
            log_step_cut=log_step_cut,
            step_cut_policy=step_cut_policy,
            step_cut_match=step_cut_match,
            legacy_global_min_val_loss=legacy_global,
            legacy_match=legacy_match,
        )

    def is_active(self) -> bool:
        return bool(
            self.step_cut
            or             self.flop_cut
            or self.flop_xmin
            or self.log_step_cut
            or self.step_cut_policy
            or self.step_cut_match
            or self.legacy_global_min_val_loss
            or self.legacy_match
        )


@dataclass
class PlotConfig:
    models: list[ModelEntry] = field(default_factory=list)
    eval_split: str = "test"
    step_cutoff: int | None = None
    flops_xmin: float | None = None
    curve_end: CurveEndConfig = field(default_factory=CurveEndConfig)
    legend_column_stacks: list[list[str]] = field(default_factory=list)

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

    def model_split(self, name: str) -> str:
        for m in self.models:
            if m.name == name:
                return m.split or self.eval_split
        return self.eval_split

    def model_source(self, name: str) -> str:
        for m in self.models:
            if m.name == name:
                return m.source_name or m.name
        return name

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

    def model_curve_cuts(self, task: TaskName) -> ModelCurveCuts:
        """Per-model step / FLOP curve truncation specs for one task."""
        return ModelCurveCuts.from_plot_config(self, task)

    def horizontal_ref_names(self, task: str) -> set[str]:
        """Models drawn as horizontal val-loss references on *task* panels."""
        from src.eval._constants import (
            is_horizontal_reference_model,
            is_steps_plot_excluded_model,
        )

        names: set[str] = set()
        for m in self.models:
            if is_horizontal_reference_model(m.name) and not is_steps_plot_excluded_model(m.name):
                names.add(m.name)
            elif task == "classification" and m.classification_horizontal_ref:
                names.add(m.name)
        return names

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
                split=m.get("split"),
                source_name=m.get("source_name"),
                step_cut=m.get("step_cut"),
                flop_cut=TaskScopedCut.from_json(m.get("flop_cut")),
                flop_xmin=TaskScopedCut.from_json(m.get("flop_xmin")),
                log_step_cut=TaskScopedCut.from_json(m.get("log_step_cut")),
                step_cut_policy=m.get("step_cut_policy"),
                step_cut_match=m.get("step_cut_match"),
                classification_horizontal_ref=bool(
                    m.get("classification_horizontal_ref", False)
                ),
            )
            for m in data.get("models", [])
        ]
        stacks_raw = data.get("legend_column_stacks")
        if stacks_raw is None and data.get("legend_column_stack"):
            stacks_raw = [data["legend_column_stack"]]
        legend_column_stacks = [list(s) for s in (stacks_raw or [])]
        curve_raw = data.get("curve_end") or {}
        curve_end = CurveEndConfig(
            policy=curve_raw.get("policy", "none"),
            match=dict(curve_raw.get("match") or {}),
        )
        return cls(
            models=models,
            eval_split=data.get("eval_split", "test"),
            step_cutoff=data.get("step_cutoff"),
            flops_xmin=data.get("flops_xmin"),
            curve_end=curve_end,
            legend_column_stacks=legend_column_stacks,
        )
