import torch
from captum.attr import IntegratedGradients, Saliency

def captum_ig(model_or_fn, x, target=None, steps=64, baseline_strategy="black"):
    if baseline_strategy == "black":
        baseline = torch.zeros_like(x)
    elif baseline_strategy == "avg":
        baseline = x.detach().mean(dim=(2,3), keepdim=True).expand_as(x)
    else:
        baseline = torch.zeros_like(x)
    ig = IntegratedGradients(model_or_fn)
    attr = ig.attribute(x, baselines=baseline, target=target, n_steps=steps)
    return attr

def captum_saliency(model, x, target=None):
    sal = Saliency(model)
    attr = sal.attribute(x, target=target)
    return attr
