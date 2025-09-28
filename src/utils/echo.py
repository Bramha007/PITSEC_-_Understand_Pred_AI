# src/utils/echo.py
"""
Shared echo utility so CLS and DET print the same neat, grep-friendly lines.
Supports an optional `writer` (e.g., tqdm.write) to avoid breaking progress bars.
"""
from typing import Dict, Iterable, Optional, Callable

def _fmt_val(key: str, val) -> str:
    if isinstance(val, float):
        if key == "lr":
            return format(val, ".1e")
        if key.startswith(("loss", "cls", "box", "obj", "rpn")):
            return format(val, ".4f")
        return format(val, ".3f")
    return str(val)

def echo_line(
    tag: str,
    kv_pairs: Dict[str, object],
    order: Optional[Iterable[str]] = None,
    writer: Optional[Callable[[str], None]] = None
) -> None:
    order = list(order or [])
    tail_keys = sorted([k for k in kv_pairs.keys() if k not in order])
    keys = order + tail_keys
    fields = []
    for k in keys:
        if k not in kv_pairs:
            continue
        v = kv_pairs[k]
        fields.append(f"{k}={_fmt_val(k, v)}")
    msg = f"{tag} | " + " | ".join(fields)
    (writer or print)(msg)
