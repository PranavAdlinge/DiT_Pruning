#!/usr/bin/env python3
"""
Check that required packages are installed. Warns on missing deps; does not install.
Run: python check_setup.py
"""

REQUIRED = [
    ("torch", "torch"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("huggingface_hub", "huggingface_hub"),
    ("mlflow", "mlflow"),
    ("streaming", "mosaicml-streaming"),
    ("yaml", "pyyaml"),
    ("safetensors", "safetensors"),
]


def main():
    missing = []
    for module, pkg in REQUIRED:
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("WARN: missing packages (run: pip install -e .)")
        for pkg in missing:
            print(f"  - {pkg}")
        return 1

    print("OK: all required packages present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
