# src/data/split.py
# Utility For Deterministic Subsampling Of (Image, XML) Pairs

import random
from typing import List, Tuple


def subsample_pairs(
    pairs: List[Tuple[str, str]],
    fraction: float,
    seed: int = 42,
    max_items: int | None = None,
    allow_zero: bool = False,
) -> List[Tuple[str, str]]:
    # Deterministically Pick A Fraction (And Optional Cap) Of Pairs
    n = len(pairs)
    if n == 0:
        return []

    clamped = max(0.0, min(1.0, float(fraction)))

    # Fast Path: Full Set Unless Capped By max_items
    if clamped >= 1.0 and max_items is None:
        return pairs

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    # Compute Number Of Items To Take
    if allow_zero and clamped == 0.0:
        take = 0
    else:
        take = max(1, int(round(n * clamped))) if clamped > 0.0 else 0

    # Apply Max Items Cap
    if max_items is not None:
        take = min(take, max_items)

    # Select Subset
    chosen = idxs[:take]
    return [pairs[i] for i in chosen]
