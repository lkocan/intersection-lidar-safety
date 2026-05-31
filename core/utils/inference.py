import torch
import torchvision
import numpy as np

def decode_predictions(preds, cfg, score_threshold=0.5, nms_iou_threshold=0.1):
    cls_preds = preds['cls_preds'][0]  
    reg_preds = preds['reg_preds'][0]  
    
    scores = torch.sigmoid(cls_preds)
    max_scores, class_ids = torch.max(scores, dim=0)
    
    mask = max_scores > score_threshold
   
    if not mask.any():
        return np.zeros((0, 9), dtype=np.float32)

    filtered_scores = max_scores[mask]
    filtered_classes = class_ids[mask]
    filtered_boxes = reg_preds[:, mask].T  
    
    if 'dir_preds' in preds:
        dir_preds = preds['dir_preds'][0]
        filtered_dirs = torch.max(dir_preds[:, mask].T, dim=1)[1] 

    x = filtered_boxes[:, 0]
    y = filtered_boxes[:, 1]
    w = filtered_boxes[:, 3]
    l = filtered_boxes[:, 4]
    
    boxes_for_nms = torch.stack([
        x - w / 2, y - l / 2,
        x + w / 2, y + l / 2
    ], dim=1)

    keep_indices = torchvision.ops.nms(boxes_for_nms, filtered_scores, nms_iou_threshold)
    
    final_boxes = filtered_boxes[keep_indices].cpu().numpy()
    final_scores = filtered_scores[keep_indices].cpu().numpy().reshape(-1, 1)
    final_classes = filtered_classes[keep_indices].cpu().numpy().reshape(-1, 1)
    
    detections = np.hstack([final_boxes, final_classes, final_scores]).astype(np.float32)
    
    return detections