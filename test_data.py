import os
import json
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy

from utils.preprocess import DAIRDataset
from tracking.tracker import Tracker3D

def main():
    dataset = DAIRDataset(split='val')
    print(f"[DAIRDataset] Načítaných {len(dataset)} vzoriek")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="M4 Intersection Safety Monitor", width=1280, height=720)
    
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0  
    render_opt.background_color = np.array([0.05, 0.05, 0.05])

    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)
    
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(axes)

    try:
        with open('roi_config.json', 'r') as f:
            roi_config = json.load(f)
            roi_polygon = roi_config['polygon'] 
        
        roi_points = [[pt[0], pt[1], -2.0] for pt in roi_polygon]
        lines = [[i, i+1] for i in range(len(roi_points)-1)]
        lines.append([len(roi_points)-1, 0])
        colors = [[0.0, 0.5, 1.0] for _ in range(len(lines))]
        
        roi_line_set = o3d.geometry.LineSet()
        roi_line_set.points = o3d.utility.Vector3dVector(roi_points)
        roi_line_set.lines = o3d.utility.Vector2iVector(lines)
        roi_line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(roi_line_set)
        print("ROI zóna úspešne načítaná.")
    except Exception as e:
        print(f"Upozornenie: Nepodarilo sa načítať ROI zónu z roi_config.json ({e})")

    print("\nSpúšťam vizualizáciu... ('Q' pre koniec, 'R' pre reset pohľadu)")

    tracker = Tracker3D(max_age=15, min_hits=2, dist_threshold=3.0)
    current_boxes = []
    current_lines = []

    for i in range(len(dataset)):
        data = dataset[i]
        
        raw_points = data['raw_points']
        if isinstance(raw_points, torch.Tensor):
            points_np = raw_points.cpu().numpy()
        else:
            points_np = raw_points
            
        points_np = points_np[:, :3]

        if len(points_np) > 0:
            heights = points_np[:, 2]
            norm_h = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
            colors = plt.get_cmap('turbo')(norm_h)[:, :3]

            pcd_o3d.points = o3d.utility.Vector3dVector(points_np)
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd_o3d)
            
            if i == 0:
                vis.reset_view_point(True)

        for geom in current_boxes + current_lines:
            vis.remove_geometry(geom, reset_bounding_box=False)
        current_boxes.clear()
        current_lines.clear()
        
        gt_boxes = data['gt_boxes']
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
            
        valid_boxes = []
        for b in gt_boxes:
            if np.sum(np.abs(b)) > 1e-5:
                det_with_score = np.append(b, [1.0]) 
                valid_boxes.append(det_with_score)
                
        detections = np.array(valid_boxes) if len(valid_boxes) > 0 else np.empty((0, 9))

        active_tracks = tracker.update(detections)

        for t in active_tracks:
            state = t.kf.x[:7]
            center = state[0:3]
            extent = state[3:6]
            yaw = state[6]
            
            R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            
            # Farby: Auto = Zelená, Chodec = Žltá, Cyklista = Fialová
            color_map = {0: (0.0, 1.0, 0.0), 1: (1.0, 1.0, 0.0), 2: (1.0, 0.0, 1.0)}
            obb.color = color_map.get(t.class_id, (1.0, 1.0, 1.0))
            
            vis.add_geometry(obb, reset_bounding_box=False)
            current_boxes.append(obb)

            if len(t.history) >= 2:
                hist_points = [[pt[0], pt[1], pt[2] + 0.5] for pt in t.history]
                lines = [[k, k+1] for k in range(len(hist_points)-1)]
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(hist_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color((1.0, 0.5, 0.0))
                
                vis.add_geometry(line_set, reset_bounding_box=False)
                current_lines.append(line_set)

        keep_running = vis.poll_events()
        vis.update_renderer()
        
        if not keep_running:
            break
            
        import time
        time.sleep(0.05)

    vis.destroy_window()

if __name__ == "__main__":
    main()
