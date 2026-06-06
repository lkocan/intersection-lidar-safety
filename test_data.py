import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / 'core'))

from utils.dair_dataset  import DAIRDataset         
from tracking.tracker_3d import Tracker3D            

CLASS_COLORS = {
    0: (0.0, 1.0, 0.0),  
    1: (1.0, 1.0, 0.0),   
    2: (1.0, 0.0, 1.0),   
}

def main():
    dataset = DAIRDataset(split='val')
    print(f"[DAIRDataset] Načítaných {len(dataset)} vzoriek")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Intersection Safety Monitor",
        width=1280, height=720,
    )

    render_opt = vis.get_render_option()
    render_opt.point_size       = 2.0
    render_opt.background_color = np.array([0.05, 0.05, 0.05])

    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(axes)

    roi_cfg = _ROOT / 'roi_config.json'     
    if roi_cfg.exists():
        try:
            with open(roi_cfg) as f:
                polygon = json.load(f)['polygon']
            roi_pts   = [[pt[0], pt[1], -2.0] for pt in polygon]
            n         = len(roi_pts)
            lines_idx = [[i, (i + 1) % n] for i in range(n)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(roi_pts)
            ls.lines  = o3d.utility.Vector2iVector(lines_idx)
            ls.paint_uniform_color([0.0, 0.5, 1.0])
            vis.add_geometry(ls)
            print("ROI zóna načítaná.")
        except Exception as e:
            print(f"ROI nenačítaná: {e}")

    tracker = Tracker3D(max_age=15, min_hits=2, dist_threshold=3.0)

    track_history: dict[int, list] = {}

    current_geoms: list = []   

    print("\nSpúšťam vizualizáciu... ('Q' pre koniec, 'R' pre reset pohľadu)")

    for i in range(len(dataset)):
        data = dataset[i]

        pts = data['raw_points']
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().numpy()
        pts = pts[:, :3]

        if len(pts) > 0:
            heights = pts[:, 2]
            norm_h  = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
            colors  = plt.get_cmap('turbo')(norm_h)[:, :3]
            pcd_o3d.points = o3d.utility.Vector3dVector(pts)
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd_o3d)
            if i == 0:
                vis.reset_view_point(True)

        for g in current_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        current_geoms.clear()

        gt = data['gt_boxes']
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        valid = [b for b in gt if np.abs(b).sum() > 1e-5]
   
        detections = (
            np.array([np.append(b, 1.0) for b in valid], dtype=np.float32)
            if valid else np.empty((0, 9), dtype=np.float32)
        )

        active_tracks = tracker.update(detections)

        for t in active_tracks:
            x, y, z, l, w, h, yaw = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
            class_id = int(t[7])
            track_id = int(t[8])

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append([x, y, z])
            
            track_history[track_id] = track_history[track_id][-30:]

            center = np.array([x, y, z])
            extent = np.array([l, w, h])
            R      = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
            obb    = o3d.geometry.OrientedBoundingBox(center, R, extent)
            obb.color = CLASS_COLORS.get(class_id, (1.0, 1.0, 1.0))
            vis.add_geometry(obb, reset_bounding_box=False)
            current_geoms.append(obb)

            hist = track_history[track_id]
            if len(hist) >= 2:
                hist_pts   = [[p[0], p[1], p[2] + 0.5] for p in hist]
                hist_lines = [[k, k + 1] for k in range(len(hist_pts) - 1)]
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(hist_pts)
                ls.lines  = o3d.utility.Vector2iVector(hist_lines)
                ls.paint_uniform_color((1.0, 0.5, 0.0))
                vis.add_geometry(ls, reset_bounding_box=False)
                current_geoms.append(ls)

        active_ids = {int(t[8]) for t in active_tracks}
        gone_ids   = set(track_history.keys()) - active_ids
        for tid in gone_ids:
            del track_history[tid]

        keep_running = vis.poll_events()
        vis.update_renderer()
        if not keep_running:
            break

        time.sleep(0.05)

    vis.destroy_window()


if __name__ == '__main__':
    main()