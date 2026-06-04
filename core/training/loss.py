import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PointPillarsLoss(nn.Module):
    CLASS_RADIUS = {0: 2.0, 1: 0.6, 2: 1.0}   

    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0

    CLS_WEIGHT: float = 1.0
    REG_WEIGHT: float = 2.0
    DIR_WEIGHT: float = 0.2

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self._anchor_cache: Optional[torch.Tensor] = None
        self._anchor_cache_hw: Optional[tuple] = None

    def _get_anchor_xy(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if self._anchor_cache_hw == (H, W) and self._anchor_cache is not None:
            return self._anchor_cache.to(device)

        x_min, x_max = self.cfg.x_range
        y_min, y_max = self.cfg.y_range

        dx = (x_max - x_min) / W    
        dy = (y_max - y_min) / H    

        xs = torch.arange(W, dtype=torch.float32) * dx + x_min + dx * 0.5  
        ys = torch.arange(H, dtype=torch.float32) * dy + y_min + dy * 0.5  

        yy, xx = torch.meshgrid(ys, xs, indexing='ij')   
        xy = torch.stack([xx, yy], dim=-1).reshape(H * W, 2)  

        anchor_xy = xy.repeat_interleave(2, dim=0)        

        self._anchor_cache    = anchor_xy
        self._anchor_cache_hw = (H, W)
        return anchor_xy.to(device)

    def _assign_targets(
        self,
        anchor_xy: torch.Tensor,
        gt_boxes:  torch.Tensor,
    ):
        N      = anchor_xy.shape[0]
        device = anchor_xy.device

        cls_targets = torch.zeros(N, 3, device=device)
        reg_targets = torch.zeros(N, 7, device=device)
        pos_mask    = torch.zeros(N, dtype=torch.bool, device=device)

        M = gt_boxes.shape[0] if gt_boxes is not None else 0
        if M == 0:
            return cls_targets, reg_targets, pos_mask, ~pos_mask

        gt_boxes = gt_boxes.to(device).float()

        dists = torch.cdist(anchor_xy.float(), gt_boxes[:, :2])   

        cls_ids = gt_boxes[:, 7].long().clamp(0, 2).tolist()
        radii   = torch.tensor(
            [self.CLASS_RADIUS[c] for c in cls_ids],
            dtype=torch.float32, device=device
        )

        norm_dists        = dists / radii.unsqueeze(0)  
        min_norm, best_gt = norm_dists.min(dim=1)        

        pos_mask = min_norm < 1.0

        best_gt_data = gt_boxes[best_gt]                              
        gt_cls_int   = best_gt_data[:, 7].long().clamp(0, 2)         

        cls_targets = (
            F.one_hot(gt_cls_int, num_classes=3).float()
            * pos_mask.float().unsqueeze(1)
        )

        reg_targets = best_gt_data[:, :7]

        return cls_targets, reg_targets, pos_mask, ~pos_mask

    def _focal_loss(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.FOCAL_ALPHA
        gamma = self.FOCAL_GAMMA

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  

        p  = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1.0 - p)

        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, alpha),
            torch.full_like(targets, 1.0 - alpha),
        )

        loss = (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()
        return loss

    def forward(
        self,
        preds:          dict,
        gt_boxes_list:  List[torch.Tensor],
        batch_size:     int,
    ) -> dict:
        cls_preds = preds['cls_preds']   
        reg_preds = preds['reg_preds']   
        dir_preds = preds['dir_preds']   

        B      = batch_size
        H, W   = cls_preds.shape[2], cls_preds.shape[3]
        device = cls_preds.device

        num_anchors = 2  
        num_classes = 3
        N = H * W * num_anchors

        cls_p = (
            cls_preds.permute(0, 2, 3, 1)      
                     .reshape(B, H * W, num_anchors, num_classes)
                     .reshape(B, N, num_classes)
        )
        reg_p = (
            reg_preds.permute(0, 2, 3, 1)      
                     .reshape(B, H * W, num_anchors, 7)
                     .reshape(B, N, 7)
        )
        dir_p = (
            dir_preds.permute(0, 2, 3, 1)      
                     .reshape(B, H * W, num_anchors, 2)
                     .reshape(B, N, 2)
        )

        anchor_xy = self._get_anchor_xy(H, W, device)  

        cls_losses, reg_losses, dir_losses = [], [], []
        num_pos_total = 0

        for b in range(B):
            gt = gt_boxes_list[b]

            if isinstance(gt, torch.Tensor) and gt.numel() > 0:
                valid = gt.abs().sum(dim=1) > 1e-5
                gt    = gt[valid]

            cls_t, reg_t, pos_mask, neg_mask = self._assign_targets(anchor_xy, gt)

            num_pos_total += int(pos_mask.sum().item())

            cls_loss_b = self._focal_loss(cls_p[b], cls_t)

            if pos_mask.any():
                reg_loss_b = F.smooth_l1_loss(
                    reg_p[b][pos_mask],
                    reg_t[pos_mask].to(device),
                    beta=1.0,
                    reduction='mean',
                )

                dir_target = (reg_t[pos_mask, 6] < 0).long().to(device)
                dir_loss_b = F.cross_entropy(dir_p[b][pos_mask], dir_target)
            else:
                reg_loss_b = cls_preds.new_zeros(1).squeeze()
                dir_loss_b = cls_preds.new_zeros(1).squeeze()

            cls_losses.append(cls_loss_b)
            reg_losses.append(reg_loss_b)
            dir_losses.append(dir_loss_b)

        cls_loss = sum(cls_losses) / B
        reg_loss = sum(reg_losses) / B
        dir_loss = sum(dir_losses) / B

        total = (
            self.CLS_WEIGHT * cls_loss
            + self.REG_WEIGHT * reg_loss
            + self.DIR_WEIGHT * dir_loss
        )

        return {
            'total':   total,
            'cls':     cls_loss,
            'reg':     reg_loss,
            'dir':     dir_loss,
            'num_pos': num_pos_total // B,  
        }