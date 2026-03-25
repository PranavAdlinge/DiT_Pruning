"""
YAML-based training config: load YAML, resolve class references (module:ClassName),
support extends (inherit from base config), and apply CLI overrides.
"""

import argparse
import importlib
import os
from pathlib import Path
from types import SimpleNamespace


def ns_to_kwargs(ns: SimpleNamespace | None, **overrides) -> dict:
    """Convert SimpleNamespace to dict for **kwargs, skipping private attrs."""
    if ns is None:
        return overrides
    d = {k: v for k, v in vars(ns).items() if not k.startswith("_")}
    d.update(overrides)
    return d


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_class(path: str):
    """Resolve 'module:ClassName' to callable class via importlib."""
    if ":" not in path:
        raise ValueError(f"Class path must be 'module:ClassName', got {path}")
    mod_path, cls_name = path.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def _deep_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert dict to SimpleNamespace for dot access."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and not any(str(kk).startswith("_") for kk in v):
            out[k] = _deep_namespace(v)
        else:
            out[k] = v
    return SimpleNamespace(**out)


def load_config(path: str | Path) -> SimpleNamespace:
    """Load YAML config. Supports extends: base_path for inheritance.
    Resolves class references (dataset, model, optimizer, loss)."""
    import yaml

    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Handle extends: load base config and deep-merge
    extends = raw.pop("extends", None)
    if extends:
        base_path = (path.parent / extends).resolve()
        if not base_path.exists():
            base_path = Path(extends).resolve()
        with open(base_path) as bf:
            base_raw = yaml.safe_load(bf) or {}
        base_raw.pop("extends", None)  # base cannot extend (avoid cycles)
        raw = _deep_merge(base_raw, raw)

    cfg = _deep_namespace(raw)

    # Resolve dataset class
    if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "class_name"):
        cfg.dataset._class = _resolve_class(cfg.dataset.class_name)

    # Resolve collate
    if hasattr(cfg, "collate") and cfg.collate and hasattr(cfg.collate, "class_name"):
        cfg.collate._fn = _resolve_class(cfg.collate.class_name)

    # Resolve pipeline class (top-level)
    if hasattr(cfg, "pipeline") and hasattr(cfg.pipeline, "class_name"):
        cfg.pipeline._class = _resolve_class(cfg.pipeline.class_name)

    # Resolve model classes
    if hasattr(cfg, "model"):
        for component in ("dit", "transformer", "vae", "scheduler"):
            if hasattr(cfg.model, component):
                comp = getattr(cfg.model, component)
                if hasattr(comp, "class_name"):
                    comp._class = _resolve_class(comp.class_name)
        # dit is primary; alias transformer -> dit for backwards compat
        if hasattr(cfg.model, "dit") and not hasattr(cfg.model, "transformer"):
            cfg.model.transformer = cfg.model.dit

    # Resolve distillation source_transformer if present
    if hasattr(cfg, "distillation") and hasattr(cfg.distillation, "source_transformer"):
        st = cfg.distillation.source_transformer
        if hasattr(st, "class_name"):
            st._class = _resolve_class(st.class_name)

    # Resolve optimizer class
    if hasattr(cfg, "optimizer") and hasattr(cfg.optimizer, "class_name"):
        cfg.optimizer._class = _resolve_class(cfg.optimizer.class_name)

    # Resolve loss class
    if hasattr(cfg, "loss") and hasattr(cfg.loss, "class_name"):
        cfg.loss._class = _resolve_class(cfg.loss.class_name)

    # Inject latent_channels and text_embed_hidden from base into dataset.kwargs
    if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "kwargs"):
        for key in ("latent_channels", "text_embed_hidden"):
            if hasattr(cfg, key) and not hasattr(cfg.dataset.kwargs, key):
                setattr(cfg.dataset.kwargs, key, getattr(cfg, key))

    return cfg


def parse_args(input_args=None) -> SimpleNamespace:
    """Parse CLI: --config required, optional overrides for output_dir, precomputed_data_dir, etc."""
    parser = argparse.ArgumentParser(description="Flux.2 Klein training (YAML config).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override mlflow.run_name")
    parser.add_argument("--precomputed_data_dir", type=str, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args(input_args) if input_args else parser.parse_args()
    cfg = load_config(args.config)
    cfg._config_path = str(Path(args.config).resolve())

    if args.precomputed_data_dir:
        if not hasattr(cfg, "dataset"):
            cfg.dataset = SimpleNamespace()
        if not hasattr(cfg.dataset, "kwargs"):
            cfg.dataset.kwargs = SimpleNamespace()
        cfg.dataset.kwargs.local = args.precomputed_data_dir
    if args.output_dir:
        if not hasattr(cfg, "mlflow"):
            cfg.mlflow = SimpleNamespace()
        cfg.mlflow.run_name = args.output_dir
    if args.max_train_steps is not None:
        cfg.train.max_steps = args.max_train_steps
    if args.resume_from_checkpoint is not None:
        cfg.resume_from_checkpoint = args.resume_from_checkpoint

    cfg.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return cfg
