# src/utils/determinism.py
import os, random, numpy as np, torch

def set_seed(seed: int, deterministic_cudnn: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def worker_init_fn(worker_id: int):
    base = int(os.environ.get("DATA_WORKER_SEED", "0"))
    s = base + worker_id
    random.seed(s)
    np.random.seed(s & 0xFFFFFFFF)
    torch.manual_seed(s)
