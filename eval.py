import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / 'core'))

from models.pointpillars import PointPillars, PointPillarsConfig
from utils.dair_dataset   import DAIRDataset

def collate_fn(batch):
    return {
        'pillars':    torch.stack([b['pillars']    for b in batch]).float(),
        'coords':     torch.stack([b['coords']     for b in batch]).int(),
        'num_points': torch.stack([b['num_points'] for b in batch]).int(),
        'gt_boxes':   [b['gt_boxes'].float() for b in batch],
        'frame_id':   [b['frame_id']          for b in batch],
        'raw_points': [b['raw_points']         for b in batch],
    }

def decode_predictions(
    preds:              dict,
    score_threshold:    float = 0.1,
    nms_iou_threshold:  float = 0.3,
) -> list:
  
    import torchvision

    results = []

    for b in range(preds['cls_preds'].shape[0]):
        cls = preds['cls_preds'][b]   
        reg = preds['reg_preds'][b]   
        H, W = cls.shape[1], cls.shape[2]

        cls_r = (
            cls.permute(1, 2, 0)        
               .reshape(H * W, 2, 3)
               .reshape(H * W * 2, 3)
        )
        reg_r = (
            reg.permute(1, 2, 0)       
               .reshape(H * W, 2, 7)
               .reshape(H * W * 2, 7)
        )

        scores     = torch.sigmoid(cls_r)            
        max_scores, class_ids = scores.max(dim=1)    

        mask = max_scores > score_threshold
        if not mask.any():
            results.append(np.zeros((0, 9), dtype=np.float32))
            continue

        f_scores  = max_scores[mask].float()
        f_classes = class_ids[mask]
        f_boxes   = reg_r[mask].float()    

        x, y = f_boxes[:, 0], f_boxes[:, 1]
        l, w = f_boxes[:, 3], f_boxes[:, 4]
        xyxy = torch.stack([x - l / 2, y - w / 2,
                            x + l / 2, y + w / 2], dim=1)
        keep = torchvision.ops.nms(xyxy, f_scores, nms_iou_threshold)

        final = np.hstack([
            f_boxes[keep].cpu().numpy(),
            f_classes[keep].cpu().numpy().reshape(-1, 1),
            f_scores[keep].cpu().numpy().reshape(-1, 1),
        ]).astype(np.float32)

        results.append(final)

    return results     

def bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a[0] - a[3] * .5, a[1] - a[4] * .5
    ax2, ay2 = a[0] + a[3] * .5, a[1] + a[4] * .5
    bx1, by1 = b[0] - b[3] * .5, b[1] - b[4] * .5
    bx2, by2 = b[0] + b[3] * .5, b[1] + b[4] * .5

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter   = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    area_a  = (ax2 - ax1) * (ay2 - ay1)
    area_b  = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def match_frame(
    preds:         np.ndarray,   
    gt:            np.ndarray,  
    iou_threshold: float,
    class_id:      int,
):
    preds_c = preds[preds[:, 7] == class_id] if len(preds) else np.empty((0, 9))
    gt_c    = gt[gt[:, 7] == class_id]        if len(gt)    else np.empty((0, 8))

    if len(preds_c) == 0:
        return [], len(gt_c)

    order   = np.argsort(-preds_c[:, 8])
    preds_c = preds_c[order]

    matched_gt = set()
    tp_fp      = []

    for pred in preds_c:
        best_iou, best_j = -1., -1
        for j, g in enumerate(gt_c):
            if j in matched_gt:
                continue
            iou = bev_iou(pred[:7], g[:7])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0:
            tp_fp.append((float(pred[8]), True))
            matched_gt.add(best_j)
        else:
            tp_fp.append((float(pred[8]), False))

    return tp_fp, len(gt_c)


def compute_ap(tp_fp: list, n_gt: int) -> float:
    if n_gt == 0 or len(tp_fp) == 0:
        return 0.0

    tp_fp.sort(key=lambda x: -x[0])

    tp_cum = fp_cum = 0
    prec, rec = [], []
    for _, is_tp in tp_fp:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        prec.append(tp_cum / (tp_cum + fp_cum))
        rec.append(tp_cum / n_gt)

    ap = 0.0
    for thr in np.arange(0, 1.1, 0.1):
        p = [p for p, r in zip(prec, rec) if r >= thr]
        ap += max(p) if p else 0.0
    return ap / 11.0


def center_precision(
    preds:    np.ndarray,
    gt:       np.ndarray,
    radius:   float,
    class_id: int,
) -> tuple:

    preds_c = preds[preds[:, 7] == class_id] if len(preds) else np.empty((0, 9))
    gt_c    = gt[gt[:, 7] == class_id]        if len(gt)    else np.empty((0, 8))

    if len(preds_c) == 0:
        return 0.0, 0.0
    if len(gt_c) == 0:
        return 0.0, 0.0

    dists = np.linalg.norm(
        preds_c[:, :2][:, None, :] - gt_c[:, :2][None, :, :],
        axis=-1,
    )  

    prec = float((dists.min(axis=1) < radius).mean())

    rec  = float((dists.min(axis=0) < radius).mean())

    return prec, rec

GT_COLOR   = (0.3, 1.0, 0.3)
PRED_COLORS = {
    0: (0.0, 0.8, 1.0),   
    1: (1.0, 1.0, 0.0),   
    2: (1.0, 0.2, 1.0),   
}


def make_obb(box7: np.ndarray, color: tuple):
    import open3d as o3d
    center = box7[:3]
    extent = np.abs(box7[3:6])
    yaw    = box7[6]
    R      = o3d.geometry.get_rotation_matrix_from_xyz((0., 0., yaw))
    obb    = o3d.geometry.OrientedBoundingBox(center, R, extent)
    obb.color = color
    return obb


class Visualizer:
    def __init__(self):
        import open3d as o3d
        import matplotlib
        self._o3d  = o3d
        self._cmap = matplotlib.colormaps['turbo']

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="PointPillars Eval  |  GT=zelená  Pred=cyan/žltá/magenta",
            width=1400, height=800,
        )
        ro = self.vis.get_render_option()
        ro.point_size       = 2.0
        ro.background_color = np.array([0.05, 0.05, 0.05])

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        self.vis.add_geometry(axes)

        self._geoms   = []
        self._first   = True

    def update(
        self,
        points:       np.ndarray,
        gt_boxes:     np.ndarray,
        pred_boxes:   np.ndarray,
        frame_id:     str,
        metrics_str:  str = "",
    ) -> bool:
     
        pts = points[:, :3]
        if len(pts):
            h     = pts[:, 2]
            norm  = (h - h.min()) / (h.max() - h.min() + 1e-6)
            cols  = self._cmap(norm)[:, :3]
            self.pcd.points = self._o3d.utility.Vector3dVector(pts)
            self.pcd.colors = self._o3d.utility.Vector3dVector(cols)
            self.vis.update_geometry(self.pcd)
            if self._first:
                self.vis.reset_view_point(True)
                self._first = False

        for g in self._geoms:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self._geoms.clear()

        for b in gt_boxes:
            if np.abs(b).sum() < 1e-5:
                continue
            obb = make_obb(b[:7], GT_COLOR)
            self.vis.add_geometry(obb, reset_bounding_box=False)
            self._geoms.append(obb)

        for p in pred_boxes:
            cls_id = int(p[7])
            score  = p[8]
            color  = PRED_COLORS.get(cls_id, (1., 1., 1.))
            color  = tuple(c * (0.4 + 0.6 * score) for c in color)
            obb    = make_obb(p[:7], color)
            self.vis.add_geometry(obb, reset_bounding_box=False)
            self._geoms.append(obb)

        title = (f"Frame {frame_id}  "
                 f"| GT={len(gt_boxes)}  Pred={len(pred_boxes)}  "
                 f"| {metrics_str}")
        print(f"\r  {title}", end='', flush=True)

        return self.vis.poll_events() and (self.vis.update_renderer() or True)

    def destroy(self):
        self.vis.destroy_window()

def run_eval(args):
    cfg    = PointPillarsConfig()
    device = (torch.device('cuda')  if torch.cuda.is_available() else
              torch.device('mps')   if torch.backends.mps.is_available() else
              torch.device('cpu'))

    print(f"\n{'='*60}")
    print(f"  PointPillars Evaluácia")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")
    print(f"  Device     : {device}")
    print(f"  Score thr  : {args.score_thresh}")
    print(f"  NMS thr    : {args.nms_thresh}")
    print(f"{'='*60}\n")

    model = PointPillars(cfg).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nenájdený: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    epoch = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f"Načítaný checkpoint: epocha {epoch}, val_loss={val_loss:.4f}\n")

    dataset = DAIRDataset(split=args.split)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    max_frames = args.max_frames if args.max_frames > 0 else len(dataset)

    CLASS_NAMES = cfg.class_names   
    N_CLASSES   = len(CLASS_NAMES)

    tp_fp_05   = defaultdict(list)  
    tp_fp_07   = defaultdict(list)  
    n_gt       = defaultdict(int)

    cp_prec    = defaultdict(list)  
    cp_rec     = defaultdict(list)  

    total_gt   = 0
    total_pred = 0
    t_start    = time.time()

    viz = Visualizer() if not args.no_viz else None

    with torch.no_grad():
        for frame_idx, batch in enumerate(loader):
            if frame_idx >= max_frames:
                break

            pillars    = batch['pillars'].to(device)
            coords     = batch['coords'].to(device)
            num_points = batch['num_points'].to(device)
            gt_raw     = batch['gt_boxes'][0].numpy()    
            frame_id   = batch['frame_id'][0]
            raw_pts    = batch['raw_points'][0]

            preds = model(pillars, coords, num_points, batch_size=1)

            detections = decode_predictions(
                preds,
                score_threshold=args.score_thresh,
                nms_iou_threshold=args.nms_thresh,
            )[0]   

            gt_valid = gt_raw[np.abs(gt_raw).sum(axis=1) > 1e-5]

            total_gt   += len(gt_valid)
            total_pred += len(detections)

            for c in range(N_CLASSES):
                tp_fp_b05, ng = match_frame(detections, gt_valid, 0.5, c)
                tp_fp_b07, _  = match_frame(detections, gt_valid, 0.7, c)

                tp_fp_05[c].extend(tp_fp_b05)
                tp_fp_07[c].extend(tp_fp_b07)
                n_gt[c] += ng

                pr, re = center_precision(detections, gt_valid, radius=2.5, class_id=c)
                if ng > 0:
                    cp_prec[c].append(pr)
                    cp_rec[c].append(re)

            if viz is not None:
                pts_np = raw_pts if isinstance(raw_pts, np.ndarray) else raw_pts.numpy()
                n_pred_now = len(detections)
                metrics_s  = (f"Car GT={int((gt_valid[:,7]==0).sum())} "
                              f"Pred={int((detections[:,7]==0).sum()) if n_pred_now else 0}")
                keep = viz.update(pts_np, gt_valid, detections, frame_id, metrics_s)
                if not keep:
                    break
                time.sleep(0.05)

            if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                elapsed = time.time() - t_start
                fps     = (frame_idx + 1) / elapsed
                print(f"\n  [{frame_idx+1:4d}/{min(max_frames, len(dataset))}]"
                      f"  {fps:.1f} fps  "
                      f"  total pred={total_pred}  gt={total_gt}")

    if viz:
        viz.destroy()

    print(f"\n\n{'='*65}")
    print(f"  METRIKY  ({frame_idx+1} framov, split={args.split})")
    print(f"{'='*65}")
    header = f"{'Trieda':<14}  {'AP@0.5':>7}  {'AP@0.7':>7}  {'CP@2.5m':>8}  {'Recall@2.5m':>12}  {'#GT':>6}"
    print(header)
    print('-' * 65)

    ap05_all, ap07_all = [], []
    cp_all, cr_all     = [], []

    for c, name in enumerate(CLASS_NAMES):
        ap05 = compute_ap(tp_fp_05[c], n_gt[c])
        ap07 = compute_ap(tp_fp_07[c], n_gt[c])
        cprec = float(np.mean(cp_prec[c])) if cp_prec[c] else 0.0
        crec  = float(np.mean(cp_rec[c]))  if cp_rec[c]  else 0.0

        print(f"  {name:<12}  {ap05:>7.3f}  {ap07:>7.3f}  {cprec:>8.3f}  {crec:>12.3f}  {n_gt[c]:>6}")

        ap05_all.append(ap05)
        ap07_all.append(ap07)
        cp_all.append(cprec)
        cr_all.append(crec)

    print('-' * 65)
    print(f"  {'mAP / mean':<12}  "
          f"{np.mean(ap05_all):>7.3f}  "
          f"{np.mean(ap07_all):>7.3f}  "
          f"{np.mean(cp_all):>8.3f}  "
          f"{np.mean(cr_all):>12.3f}  "
          f"{sum(n_gt.values()):>6}")
    print(f"{'='*65}")

    elapsed = time.time() - t_start
    fps     = (frame_idx + 1) / elapsed
    print(f"\n  Čas: {elapsed:.1f}s  |  {fps:.1f} frames/s")
    print(f"  Celkovo predikovaných boxov: {total_pred}")
    print(f"  Celkovo GT boxov:            {total_gt}")
    print()

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='PointPillars evaluácia a vizualizácia'
    )
    p.add_argument(
        '--checkpoint',
        type=str,
        default='core/checkpoints/best.pth',
        help='Cesta k .pth súboru (default: core/checkpoints/best.pth)',
    )
    p.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val'],
        help='Dataset split (default: val)',
    )
    p.add_argument(
        '--score-thresh',
        type=float,
        default=0.1,
        help='Minimálne skóre pre detekciu (default: 0.1)',
    )
    p.add_argument(
        '--nms-thresh',
        type=float,
        default=0.3,
        help='NMS IoU prah (default: 0.3)',
    )
    p.add_argument(
        '--no-viz',
        action='store_true',
        help='Preskočiť Open3D vizualizáciu (použiť na serveri)',
    )
    p.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Limit počtu framov (0 = všetky)',
    )

    args = p.parse_args()
    run_eval(args)