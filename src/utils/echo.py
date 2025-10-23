# src/utils/echo.py

# Unified Echo Utility For Consistent, Grep-Friendly Logging
# Supports Custom Key Ordering And Optional Writer Function

# Standard Library
from typing import Dict, Iterable, Optional, Callable

# Third-Party
from tqdm import tqdm


def _fmt_val(key: str, val) -> str:
    # Format Values Based On Key Type For Readable Logging
    if isinstance(val, float):
        if key == "lr":
            return format(val, ".1e")                 # Learning Rate In Scientific Notation
        if key.startswith(("loss", "cls", "box", "obj", "rpn")):
            return format(val, ".4f")                 # Losses / Metrics With 4 Decimal Places
        return format(val, ".3f")                     # Other Floats With 3 Decimal Places
    return str(val)


def echo_line(
    tag: str,
    kv_pairs: Dict[str, object],
    order: Optional[Iterable[str]] = None,
    writer: Optional[Callable[[str], None]] = None
) -> None:
    # Print Or Write A Grep-Friendly Line With Tag + Key=Value Pairs
    order = list(order or [])                        # Force Key Order For Selected Keys
    tail_keys = sorted([k for k in kv_pairs if k not in order])
    keys = order + tail_keys                         # Append Remaining Keys Sorted
    fields = [f"{k}={_fmt_val(k, kv_pairs[k])}" for k in keys if k in kv_pairs]
    msg = f"{tag} | " + " | ".join(fields)          # Combine Tag With Key=Value Fields
    (writer or tqdm.write)(msg)                      # Default Writer Uses tqdm.write
