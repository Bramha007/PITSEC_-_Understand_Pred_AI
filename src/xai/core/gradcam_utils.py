import torch
import torch.nn.functional as F

class GradCAM:
    """
    Minimal Grad-CAM for torchvision ResNet.
    target_module: e.g., model.layer4[-1]
    """
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.handles = []
        self.feats = None
        self.grads = None
        self._register()

    def _save_feats(self, _, __, output):
        self.feats = output.detach()

    def _save_grads(self, _, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def _register(self):
        self.handles.append(self.target_module.register_forward_hook(self._save_feats))
        self.handles.append(self.target_module.register_full_backward_hook(self._save_grads))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def _normalize(self, x, eps=1e-6):
        x = x - x.min()
        x = x / (x.max() + eps)
        return x

    def __call__(self, x, target_class=None):
        """
        x: tensor [1,3,H,W] normalized for the model
        target_class: int (if None, picks argmax)
        returns heatmap in [0,1] as [1,1,H,W]
        """
        self.model.zero_grad()
        logits = self.model(x)                 # [1,C]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class]
        score.backward(retain_graph=True)

        assert self.feats is not None and self.grads is not None, "Hooks did not fire."
        weights = self.grads.mean(dim=(2,3), keepdim=True)          # [1,C,1,1]
        cam = (weights * self.feats).sum(dim=1, keepdim=True)       # [1,1,h,w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize(cam)
        return cam, int(target_class), logits.softmax(dim=1)[0].detach()
