"""
Logging utilities: MLflow setup and tracker configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerate import Accelerator


def _require_mlflow(cfg) -> None:
    """Assert MLflow is configured. Raises if report_to != 'mlflow' or mlflow.experiment_name/run_name missing."""
    if cfg.report_to != "mlflow":
        raise ValueError("report_to must be 'mlflow'")
    mlflow_cfg = getattr(cfg, "mlflow", None)
    if not mlflow_cfg:
        raise ValueError("mlflow config required (mlflow: experiment_name: ..., run_name: ...)")
    if not getattr(mlflow_cfg, "experiment_name", None):
        raise ValueError("mlflow.experiment_name is required")
    if not getattr(mlflow_cfg, "run_name", None):
        raise ValueError("mlflow.run_name is required")


def setup_mlflow_log_with(log_root: str | Path, mlflow_cfg) -> "MLflowTracker":
    """
    Setup MLflow tracking. Returns MLflowTracker.
    MLflow store at project level: log_root/mlruns.
    Sets run_name from mlflow.run_name.
    """
    mlflow_dir = Path(log_root).resolve() / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = getattr(mlflow_cfg, "tracking_uri", None) or mlflow_dir.as_uri()
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    from accelerate.tracking import MLflowTracker
    return MLflowTracker(
        experiment_name=mlflow_cfg.experiment_name,
        logging_dir=str(mlflow_dir),
        run_name=mlflow_cfg.run_name,
    )


def set_mlflow_user_tag(user: str | None) -> None:
    """Set mlflow.user tag on the current run if user is set (skip when 'default')."""
    u = (user and str(user).strip()) or ""
    if not u or u == "default":
        return
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.set_tag("mlflow.user", str(user).strip())
    except Exception:
        pass


def get_output_dir(log_root: str | Path, experiment_name: str, run_name: str, user: str | None = None) -> Path:
    """Output dir for a run: log_root/experiments/{user}/{experiment_name}-{run_name}. User defaults to 'default' if null."""
    user_dir = user or "default"
    return Path(log_root).resolve() / "experiments" / user_dir / f"{experiment_name}-{run_name}"


def setup_tracking(cfg) -> tuple[str, "MLflowTracker"]:
    """
    Validate mlflow config and setup tracking. Sets cfg.output_dir.
    Returns (output_dir, log_with) for use with Accelerator.
    """
    _require_mlflow(cfg)
    mlflow_cfg = cfg.mlflow
    log_root = Path(getattr(cfg, "log_root", "logs")).resolve()
    experiment_name = mlflow_cfg.experiment_name
    run_name = mlflow_cfg.run_name
    run_user = getattr(mlflow_cfg, "user", None)
    output_dir = str(get_output_dir(log_root, experiment_name, run_name, user=run_user))
    cfg.output_dir = output_dir
    log_with = setup_mlflow_log_with(log_root, mlflow_cfg)
    return output_dir, log_with


def init_trackers(accelerator: "Accelerator", cfg) -> None:
    """
    Init MLflow trackers and set user tag. Call when accelerator.is_main_process.
    """
    mlflow_cfg = cfg.mlflow
    config_dict = {}
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            try:
                config_dict[k] = str(v)
            except Exception:
                config_dict[k] = repr(v)
    accelerator.init_trackers(mlflow_cfg.experiment_name, config=config_dict)
    set_mlflow_user_tag(getattr(mlflow_cfg, "user", None))


def log_dataset_input(
    class_name: str | None = None,
    kwargs: dict | None = None,
    name: str | None = None,
    source_path: str | Path | None = None,
    context: str = "training",
) -> None:
    """
    Log dataset to MLflow run. Records class and kwargs as params. If source_path is given,
    also logs as MetaDataset for lineage. Name from config or class_name fallback.
    """
    try:
        import mlflow

        dataset_name = name or (class_name.split(":")[-1] if class_name else "dataset")
        if class_name is not None:
            mlflow.log_param("dataset.class", class_name)
        if kwargs:
            mlflow.log_param("dataset.kwargs", str(kwargs))
        if source_path:
            import warnings

            from mlflow.data.dataset_source_registry import resolve_dataset_source
            from mlflow.data.meta_dataset import MetaDataset

            path = str(Path(source_path).resolve())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                src = resolve_dataset_source(path)
            dataset = MetaDataset(source=src, name=dataset_name)
            mlflow.log_input(dataset, context=context)
    except Exception:
        pass


def log_validation_images_to_mlflow(images: list, step: int, output_dir: str | Path) -> None:
    """
    Save validation images under output_dir/validation_images/ and log to MLflow.
    """
    import mlflow

    val_dir = Path(output_dir) / "validation_images"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = val_dir / f"validation_step{step}_img{i}.png"
        img.save(path)
        mlflow.log_artifact(str(path), artifact_path="validation")
