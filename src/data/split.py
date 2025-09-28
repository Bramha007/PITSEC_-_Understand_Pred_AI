# src/data/split.py
import random
from typing import List, Tuple

def subsample_pairs(
    pairs: List[Tuple[str, str]],
    fraction: float,
    seed: int = 42,
    max_items: int | None = None,
    allow_zero: bool = False,
) -> List[Tuple[str, str]]:
    """Deterministically pick a fraction (and optional cap) of (img, xml) pairs.

    Behavior:
      - fraction is clamped to [0,1].
      - If fraction >= 1.0 and no max_items is set, returns the original order.
      - For small positive fractions, we take at least 1 item.
      - If allow_zero=True and the (clamped) fraction is exactly 0, we take 0.
      - max_items (if provided) caps the final count.
    """
    n = len(pairs)
    if n == 0:
        return []

    clamped = max(0.0, min(1.0, float(fraction)))

    # Fast path: full set, unless capped by max_items.
    if clamped >= 1.0 and max_items is None:
        return pairs

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    if allow_zero and clamped == 0.0:
        take = 0
    else:
        # At least 1 for any positive fraction on small splits
        take = max(1, int(round(n * clamped))) if clamped > 0.0 else 0

    if max_items is not None:
        take = min(take, max_items)

    chosen = idxs[:take]
    return [pairs[i] for i in chosen]
