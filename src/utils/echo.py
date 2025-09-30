# src/utils/echo.py
# Unified echo utility: consistent, grep-friendly log lines for CLS and DET
# Optional `writer` (e.g., tqdm.write) keeps progress bars intact

from typing import Dict, Iterable, Optional, Callable

def _fmt_val(key: str, val) -> str:
    if isinstance(val, float):
        if key == "lr":                            # learning rates in sci-notation
            return format(val, ".1e")
        if key.startswith(("loss", "cls", "box", "obj", "rpn")):
            return format(val, ".4f")              # losses/metrics in 4dp
        return format(val, ".3f")                  # other floats in 3dp
    return str(val)

def echo_line(
    tag: str,
    kv_pairs: Dict[str, object],
    order: Optional[Iterable[str]] = None,
    writer: Optional[Callable[[str], None]] = None
) -> None:
    order = list(order or [])                      # force order for some keys
    tail_keys = sorted([k for k in kv_pairs if k not in order])
    keys = order + tail_keys                       # remaining keys sorted
    fields = [f"{k}={_fmt_val(k, kv_pairs[k])}" for k in keys if k in kv_pairs]
    msg = f"{tag} | " + " | ".join(fields)         # TAG + key=val fields
    (writer or print)(msg)                         # default to print
