"""Utilities for OBB compiled-model inference and evaluation."""

import os

import numpy
import torch


def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)


def compute_metric_obb(output, target_cls, target_xywhr, iou_v):
    """OBB metrics: output (k, 7) [xywh, conf, cls, angle], target_xywhr (m, 5), target_cls (m,)."""
    from utils.obb_utils import batch_probiou, match_predictions_obb

    if output.shape[0] == 0:
        return torch.zeros(0, iou_v.numel(), dtype=torch.bool, device=output.device)
    pred_xywhr = torch.cat([output[:, :4], output[:, 6:7]], dim=-1)
    pred_cls = output[:, 5].cpu().numpy().astype(int)
    gt_cls_np = (
        target_cls.cpu().numpy().astype(int)
        if hasattr(target_cls, "cpu")
        else numpy.asarray(target_cls, dtype=int)
    )
    if target_xywhr.shape[0] == 0:
        return torch.zeros(output.shape[0], iou_v.numel(), dtype=torch.bool, device=output.device)
    iou_m = batch_probiou(pred_xywhr, target_xywhr)
    if iou_m.dim() == 1:
        iou_m = iou_m.unsqueeze(1)
    correct = match_predictions_obb(pred_cls, gt_cls_np, iou_m, iou_v.cpu().numpy())
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def nms_rotated(boxes, scores, threshold=0.45):
    """Rotated NMS: boxes (n, 5) xywhr, scores (n,). Returns indices to keep."""
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    from utils.obb_utils import batch_probiou

    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze(-1)
    return sorted_idx[pick]


def non_max_suppression_obb(outputs, confidence_threshold=0.001, iou_threshold=0.45):
    """OBB NMS: outputs (b, N, 4+nc+1) -> list of (k, 7) [xywh, conf, cls, angle]."""
    max_det = 300
    max_nms = 30000
    bs = outputs.shape[0]
    nc = outputs.shape[-1] - 5
    out_list = [torch.zeros((0, 7), device=outputs.device, dtype=outputs.dtype)] * bs
    pred = outputs.permute(0, 2, 1)
    for xi, x in enumerate(pred):
        xc = x[4 : 4 + nc].amax(0) > confidence_threshold
        x = x[:, xc].T
        if not x.shape[0]:
            continue
        box = x[:, :4]
        cls = x[:, 4 : 4 + nc]
        angle = x[:, -1:]
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), angle), 1)[conf.view(-1) > confidence_threshold]
        if not x.shape[0]:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        scores = x[:, 4]
        xywhr = torch.cat((x[:, :4], x[:, -1:]), dim=-1)
        keep = nms_rotated(xywhr, scores, iou_threshold)[:max_det]
        out_list[xi] = x[keep]
    return out_list


def smooth(y, f=0.1):
    nf = round(len(y) * f * 2) // 2 + 1
    p = numpy.ones(nf // 2)
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode="valid")


def plot_pr_curve(px, py, ap, names, save_dir):
    from matplotlib import pyplot

    if len(py) == 0:
        print("Warning: No data to plot for PR Curve.")
        return
    fig, ax = pyplot.subplots(1, 1, figsize=(5, 5), layout="constrained")
    py = numpy.stack(py, axis=1)

    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py, linewidth=1, color="grey")

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-Recall Curve")
    fig.legend(loc="outside lower center")
    fig.savefig(save_dir, dpi=250)
    pyplot.close(fig)


def plot_curve(px, py, names, save_dir, x_label="Confidence", y_label="Metric"):
    from matplotlib import pyplot

    figure, ax = pyplot.subplots(1, 1, figsize=(5, 5), layout="constrained")

    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")

    y = smooth(py.mean(0), f=0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.3f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{y_label}-Confidence Curve")
    figure.legend(loc="outside lower center")
    figure.savefig(save_dir, dpi=250)
    pyplot.close(figure)


def compute_ap(version, tp, conf, output, target, plot=False, names=(), eps=1e-16, save_dir=None):
    """Average precision from TP/conf/class arrays."""
    i = numpy.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    unique_classes, nt = numpy.unique(target, return_counts=True)
    nc = unique_classes.shape[0]

    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(start=0, stop=1, num=1000), []
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]
        no = i.sum()
        if no == 0 or nl == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (nl + eps)
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))
            x = numpy.linspace(start=0, stop=1, num=101)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)
            if plot and j == 0:
                py.append(numpy.interp(px, m_rec, m_pre))

    f1 = 2 * p * r / (p + r + eps)
    if plot:
        plot_names = []
        for c in unique_classes:
            c = int(c)
            if 0 <= c < len(names):
                plot_names.append(names[c])
            else:
                plot_names.append(f"Class_{c}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_pr_curve(px, py, ap, plot_names, save_dir=os.path.join(save_dir, "PR_curve.png"))
            plot_curve(px, f1, plot_names, save_dir=os.path.join(save_dir, "F1_curve.png"), y_label="F1")
            plot_curve(px, p, plot_names, save_dir=os.path.join(save_dir, "P_curve.png"), y_label="Precision")
            plot_curve(px, r, plot_names, save_dir=os.path.join(save_dir, "R_curve.png"), y_label="Recall")
        else:
            weight_dir = "./runs"
            os.makedirs(weight_dir, exist_ok=True)
            plot_pr_curve(px, py, ap, plot_names, save_dir=os.path.join(weight_dir, "PR_curve.png"))
            plot_curve(px, f1, plot_names, save_dir=os.path.join(weight_dir, "F1_curve.png"), y_label="F1")
            plot_curve(px, p, plot_names, save_dir=os.path.join(weight_dir, "P_curve.png"), y_label="Precision")
            plot_curve(px, r, plot_names, save_dir=os.path.join(weight_dir, "R_curve.png"), y_label="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    ap50, ap = ap[:, 0], ap.mean(1)
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


class Colors:
    def __init__(self):
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return c[::-1] if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


def draw_rotated_box(im, xywhr, index, label=""):
    """Draw OBB from xywhr (x, y, w, h, angle_rad)."""
    import cv2
    from utils.obb_utils import xywhr2xyxyxyxy

    t = torch.as_tensor(xywhr[:5], dtype=torch.float32).unsqueeze(0)
    corners = xywhr2xyxyxyxy(t).squeeze(0).cpu().numpy()
    pts = corners.astype(numpy.int32).reshape((-1, 1, 2))
    color = Colors()(index, True)
    cv2.polylines(im, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        th += 3
        x1, y1 = int(corners[:, 0].min()), int(corners[:, 1].min())
        outside = y1 < th + 3
        if x1 + tw > im.shape[1]:
            x1 = im.shape[1] - tw
        y2_label = y1 + th if outside else y1 - th
        cv2.rectangle(im, (x1, y1), (x1 + tw, y2_label), color, -1)
        cv2.putText(
            im,
            label,
            (x1, y1 + th - 3 if outside else y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    return im
