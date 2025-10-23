# src/utils/echo.py

# Unified Echo Utility: Consistent, Grep-Friendly Log Lines For CLS And DET

from typing import Dict, Iterable, Optional, Callable
from tqdm import tqdm


def _fmt_val(key: str, val) -> str:
    if isinstance(val, float):
        if key == "lr":                            # Learning Rates In Sci-Notation
            return format(val, ".1e")
        if key.startswith(("loss", "cls", "box", "obj", "rpn")):
            return format(val, ".4f")              # Losses/Metrics In 4dp
        return format(val, ".3f")                  # Other Floats In 3dp
    return str(val)


def echo_line(
    tag: str,
    kv_pairs: Dict[str, object],
    order: Optional[Iterable[str]] = None,
    writer: Optional[Callable[[str], None]] = None
) -> None:
    order = list(order or [])                      # Force Order For Some Keys
    tail_keys = sorted([k for k in kv_pairs if k not in order])
    keys = order + tail_keys                       # Remaining Keys Sorted
    fields = [f"{k}={_fmt_val(k, kv_pairs[k])}" for k in keys if k in kv_pairs]
    msg = f"{tag} | " + " | ".join(fields)         # Tag + Key=Val Fields
    (writer or tqdm.write)(msg)                    # Always Use tqdm.write
