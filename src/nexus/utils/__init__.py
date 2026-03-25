"""
Shared utilities: io, logging, device/dtype helpers, train utils, etc.
"""

import torch

DATA_TYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
