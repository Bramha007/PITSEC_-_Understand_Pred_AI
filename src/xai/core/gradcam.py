# src/xai/core/gradcam.py
import torch
import torch.nn.functional as F

class GradCAM:
    """Lightweight Grad-CAM.
    target_module: the conv block to hook (e.g., model.layer4[-1]).
    No plotting or file I/O here.
    """
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self._fh = []
        self._feats = None
        self._grads = None
        self._register()

    def _save_feats(self, _, __, out): self._feats = out.detach()
    def _save_grads(self, _, grad_in, grad_out): self._grads = grad_out[0].detach()

    def _register(self):
        self._fh.append(self.target_module.register_forward_hook(self._save_feats))
        self._fh.append(self.target_module.register_full_backward_hook(self._save_grads))

    def remove(self):
        for h in self._fh: h.remove()
        self._fh = []

    @torch.no_grad()
    def _normalize(self, x, eps=1e-6):
        x = x - x.min()
        x = x / (x.max() + eps)
        return x

    def __call__(self, x, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)     # [B,C] or [1,C]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[:, target_class]
        score.backward(retain_graph=True)

        assert self._feats is not None and self._grads is not None, "GradCAM hooks did not fire."
        w = self._grads.mean(dim=(2,3), keepdim=True)   # [B,C,1,1]
        cam = (w * self._feats).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize(cam)
        return cam, target_class, logits.softmax(dim=1).detach()
