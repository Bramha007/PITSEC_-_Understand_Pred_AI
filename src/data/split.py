# src/data/split.py

# Utilities For Subsampling And Randomly Selecting Image/Annotation Pairs
# Supports Deterministic Fractional Subsampling With Optional Max Items Cap
# Ensures Reproducible Selection Using Random Seeds

# Standard Library
import random
from typing import List, Tuple


# Subsample A Fraction Of Image/Annotation Pairs Deterministically
def subsample_pairs(
    pairs: List[Tuple[str, str]],
    fraction: float,
    seed: int = 42,
    max_items: int | None = None,
    allow_zero: bool = False,
) -> List[Tuple[str, str]]:
    # Returns A Random Subset Of Pairs Based On Fraction, Seed, And Max Items Limit
    n = len(pairs)
    if n == 0:
        return []

    # Clamp Fraction Between 0 And 1
    clamped = max(0.0, min(1.0, float(fraction)))

    # Fast Path: Return Full Set If Fraction >= 1 And No Max Items Limit
    if clamped >= 1.0 and max_items is None:
        return pairs

    # Initialize Random Generator With Seed For Determinism
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    # Determine Number Of Items To Take
    if allow_zero and clamped == 0.0:
        take = 0
    else:
        take = max(1, int(round(n * clamped))) if clamped > 0.0 else 0

    # Apply Optional Maximum Items Cap
    if max_items is not None:
        take = min(take, max_items)

    # Select Subset Of Pairs
    chosen = idxs[:take]
    return [pairs[i] for i in chosen]
