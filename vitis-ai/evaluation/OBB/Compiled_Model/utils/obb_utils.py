import math
import numpy
import torch


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode DFL distance + angle to rotated box (xywh format)."""
    lt, rb = pred_dist.split(2, dim=dim)
    cos_a = torch.cos(pred_angle)
    sin_a = torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x = xf * cos_a - yf * sin_a
    y = xf * sin_a + yf * cos_a
    xy = torch.cat([x, y], dim=dim)
    if dim == 1:
        if anchor_points.dim() == 2:
            anchor_points = anchor_points.T.unsqueeze(0)
        elif anchor_points.dim() == 3 and anchor_points.shape[1] != xy.shape[1]:
            anchor_points = anchor_points.permute(0, 2, 1)
    xy = xy + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def bbox2dist(anchor_points, bbox_xyxy, reg_max):
    """Convert bbox (x1,y1,x2,y2) to DFL target distances from anchor points."""
    x1y1, x2y2 = bbox_xyxy.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def xywh2xyxy(x):
    """Convert xywh to x1y1x2y2 (axis-aligned)."""
    assert x.shape[-1] >= 4
    y = torch.empty_like(x[..., :4])
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y


def xywhr2xyxyxyxy(center):
    """Convert xywhr (center, w, h, angle rad) to 4 corners (N, 4, 2)."""
    ctr = center[..., :2]
    w, h, angle = (center[..., i:i + 1] for i in range(2, 5))
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    vec1 = torch.cat([w / 2 * cos_a, w / 2 * sin_a], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_a, h / 2 * cos_a], dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.stack([pt1, pt2, pt3, pt4], dim=-2)


def _get_covariance_matrix(boxes):
    """Helper for probiou: boxes (..., 5) xywhr."""
    gbbs = torch.cat((torch.pow(boxes[..., 2:4], 2) / 12, boxes[..., 4:5]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    return (
        a * torch.cos(c) ** 2 + b * torch.sin(c) ** 2,
        a * torch.sin(c) ** 2 + b * torch.cos(c) ** 2,
        a * torch.cos(c) * torch.sin(c) - b * torch.sin(c) * torch.cos(c),
    )


def xyxyxyxy2xywhr(corners):
    """Convert 4 corners (N, 8) or (N, 4, 2) to xywhr (N, 5). angle in radians."""
    import cv2
    if hasattr(corners, 'numpy'):
        points = corners.cpu().numpy()
    else:
        points = corners
    points = points.reshape(-1, 4, 2)
    rboxes = []
    for pts in points:
        (x, y), (w, h), angle_deg = cv2.minAreaRect(pts.astype('float32'))
        rboxes.append([x, y, w, h, angle_deg * math.pi / 180])
    if hasattr(corners, 'cpu'):
        return torch.tensor(rboxes, dtype=corners.dtype, device=corners.device)
    return numpy.asarray(rboxes, dtype=points.dtype)


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """ProbIoU for rotated boxes (xywhr, last dim 5)."""
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou


def batch_probiou(obb1, obb2, eps=1e-7):
    """IoU matrix between two sets of OBBs. obb1 (N,5), obb2 (M,5) -> (N,M)."""
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    return 1 - hd


def match_predictions_obb(pred_cls, true_classes, iou_matrix, iouv):
    """Match predictions to GT using OBB IoU. Returns correct (N, len(iouv)) bool."""
    pred_cls = numpy.asarray(pred_cls)
    true_classes = numpy.asarray(true_classes)
    correct = numpy.zeros((len(pred_cls), len(iouv)), dtype=bool)
    correct_class = (pred_cls[:, None] == true_classes[None, :])
    if hasattr(iou_matrix, "cpu"):
        iou = (iou_matrix * torch.tensor(correct_class, dtype=iou_matrix.dtype, device=iou_matrix.device)).cpu().numpy()
    else:
        iou = numpy.asarray(iou_matrix) * correct_class
    for i, threshold in enumerate(iouv):
        matches = numpy.nonzero(iou >= threshold)
        matches = numpy.array(matches).T
        if matches.shape[0] > 0:
            if matches.shape[0] > 1:
                iou_vals = iou[matches[:, 0], matches[:, 1]]
                matches = matches[iou_vals.argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 0].astype(int), i] = True
    return correct
