import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_prediction(image_tensor, pred, gt=None, score_thr=0.5, save_path=None):
    img = image_tensor.permute(1,2,0).cpu().numpy()
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.imshow(img); ax.axis("off")
    # predictions
    for b, s in zip(pred["boxes"], pred["scores"]):
        if s < score_thr: continue
        x1,y1,x2,y2 = b.detach().cpu().numpy()
        ax.add_patch(patches.Rectangle((x1,y1),(x2-x1),(y2-y1), edgecolor="red",
                                       fill=False, lw=2))
    # ground-truth
    if gt is not None:
        for b in gt["boxes"]:
            x1,y1,x2,y2 = b.cpu().numpy()
            ax.add_patch(patches.Rectangle((x1,y1),(x2-x1),(y2-y1),
                                           fill=False, lw=1, linestyle="--", edgecolor="lime"))
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    else:
        plt.show()


