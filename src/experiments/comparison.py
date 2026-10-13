"""Matched 3DR-Net and conventional 3D U-Net studies without another fit loop."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from .ablation import AblationResult, run_study
from .config import ExperimentConfig, ModelConfig
from .data import PreparedData, SubjectSplit
from ..evaluation.k_fold import FoldSplit


@dataclass(frozen=True)
class ModelVariant:
    """A named architecture; every non-model setting comes from the reference."""

    name: str
    model: ModelConfig

    def configuration(self, base: ExperimentConfig) -> ExperimentConfig:
        """Use an explicit model configuration without mutating shared settings."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError("Model names must be safe folder identifiers")
        config = replace(base, model=self.model)
        config.validate()
        return config


def comparison_plan(base: ExperimentConfig, variants: Sequence[ModelVariant]) -> pd.DataFrame:
    """Require matched widths, heads and scratch initialization for this baseline.

    The conventional U-Net disables residual/attention options. Parameter counts
    may differ: identical widths and epoch budgets do not imply equal capacity.
    """
    base.validate()
    if len(variants) != 2 or len({variant.name for variant in variants}) != 2:
        raise ValueError("Provide two uniquely named model variants")
    if {variant.model.architecture for variant in variants} != {"3drnet", "unet"}:
        raise ValueError("Compare one 3DR-Net with one U-Net")
    if base.initialization.mode != "scratch":
        raise ValueError("This matched comparison requires scratch initialization")

    rows = []
    for variant in variants:
        model = variant.configuration(base).model
        if model.filters != base.model.filters or model.classifier:
            raise ValueError("Match filter widths and omit the unsupported U-Net classifier")
        if not model.max_pooling:
            raise ValueError("Use conventional max pooling in the baseline comparison")
        if model.architecture == "unet" and (model.residual or model.attention):
            raise ValueError("The conventional U-Net baseline disables residual and attention")
        rows.append({"Variant": variant.name, "Architecture": model.architecture,
                     "Filters": model.filters, "Residual": model.residual,
                     "Attention": model.attention, "MaxPooling": model.max_pooling})
    return pd.DataFrame(rows)


def run_model_comparison(
    data: PreparedData, variants: Sequence[ModelVariant], *, study_dir: Path,
    mode: Literal["fixed_split", "cross_validation"] = "fixed_split",
    split: SubjectSplit | None = None, folds: list[FoldSplit] | None = None,
    seeds: Sequence[int] = (42,), resume: bool = False,
    callbacks: Sequence[Any] | None = None,
) -> AblationResult:
    """Use identical animals, training seeds, losses, crops and maximum budgets."""
    plan = comparison_plan(data.config, variants)
    return run_study(
        data=data, variants=variants, plan=plan, study_dir=study_dir, mode=mode,
        split=split, folds=folds, seeds=seeds, resume=resume, callbacks=callbacks,
    )


def model_complexity(base: ExperimentConfig, variants: Sequence[ModelVariant]) -> pd.DataFrame:
    """Build each existing model and count parameters; no training or FLOP estimate."""
    import tensorflow as tf
    import numpy as np
    from .training import build_model

    comparison_plan(base, variants)
    rows = []
    for variant in variants:
        tf.keras.backend.clear_session()
        config = variant.configuration(base)
        model = build_model(config)
        trainable = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
        rows.append({"Variant": variant.name, "Parameters": model.count_params(),
                     "TrainableParameters": trainable})
        del model
    tf.keras.backend.clear_session()
    return pd.DataFrame(rows)
