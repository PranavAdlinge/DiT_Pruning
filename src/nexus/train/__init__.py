"""
Training: config-driven Flux.2 Klein on precomputed SSTK MDS data.

  from nexus.train import main
    main()  # Uses configs/klein4b/t2i_finetune.yaml, MDS from precompute.py
"""

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from .main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
