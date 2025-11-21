# import random
# from typing import List, Tuple

# def subsample_pairs(
#     pairs: List[Tuple[str, str]],
#     fraction: float,
#     seed: int = 42,
#     max_items: int | None = None,
# ) -> List[Tuple[str, str]]:
#     """Deterministically pick a fraction (and optional cap) of (img, xml) pairs."""
#     if fraction >= 1.0 and not max_items:
#         return pairs
#     rng = random.Random(seed)
#     idxs = list(range(len(pairs)))
#     rng.shuffle(idxs)
#     take = int(len(pairs) * max(0.0, min(1.0, fraction)))
#     if max_items is not None:
#         take = min(take if take > 0 else len(pairs), max_items)
#     chosen = idxs[:take] if take > 0 else idxs
#     return [pairs[i] for i in chosen]

# File: src/dataio/split_utils.py

from typing import List, Tuple
import torch

def subsample_pairs(
    pairs: List[Tuple[str, str]],
    fraction: float,
    seed: int = 42,
    max_items: int | None = None,
) -> List[Tuple[str, str]]:
    """
    Deterministically selects a fraction (and optional cap) of (img, xml) pairs
    using PyTorch's random generator for reproducibility.
    """
    total_len = len(pairs)
    if not total_len:
        return []

    # 1. Calculate number of items to select based on fraction
    take = int(total_len * max(0.0, min(1.0, fraction)))
    
    # 2. Apply max_items cap
    if max_items is not None:
        # If the fractional count is 0, default to taking all, then cap by max_items
        # This prevents returning 0 items if fraction is very small but max_items > 0
        take = min(max(take, 1) if take < 1 and max_items > 0 else take, max_items)

    # Final safeguard: ensure 'take' doesn't exceed total length
    take = min(take, total_len)

    # Handle edge case where we select everything
    if take >= total_len:
        return pairs
    
    # 3. Deterministic Shuffling using PyTorch Generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # torch.randperm returns a random permutation of indices
    indices = torch.randperm(total_len, generator=generator).tolist()
    
    # 4. Select the subsample
    chosen_indices = indices[:take]
    
    return [pairs[i] for i in chosen_indices]