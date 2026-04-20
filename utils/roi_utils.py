import numpy as np
from shapely.geometry import Point, Polygon

def filter_detections_by_roi(detections, roi_coords):
    """Filtruje detekcie (N, 9) podľa toho, či ležia vnútri polygonu."""
    if len(detections) == 0 or roi_coords is None:
        return detections
    
    poly = Polygon(roi_coords)
    mask = []
    for d in detections:
        # d[0]=x, d[1]=y
        mask.append(poly.contains(Point(d[0], d[1])))
    
    return detections[np.array(mask)]

def calculate_risk_score(current_track, all_tracks, roi_center=(0, 0)):
    """Vypočíta Risk Score pre konkrétny objekt na základe okolia."""
    # 1. Proximity risk (vzdialenosť k najbližšiemu susedovi)
    min_dist = 20.0
    for other in all_tracks:
        if other.id == current_track.id:
            continue
        dist = np.linalg.norm(current_track.kf.x[:2] - other.kf.x[:2])
        if dist < min_dist:
            min_dist = dist
    
    proximity_risk = np.exp(-0.3 * min_dist)

    # 2. Convergence risk (smerujú k sebe?)
    velocity_risk = 0.0
    v1 = current_track.kf.x[7:9]
    for other in all_tracks:
        if other.id == current_track.id:
            continue
        rel_pos = other.kf.x[:2] - current_track.kf.x[:2]
        rel_vel = v1 - other.kf.x[7:9]
        
        # Ak sa približujú, skalárny súčin bude kladný
        dist_norm = np.linalg.norm(rel_pos) + 1e-6
        closing_speed = np.dot(rel_vel, rel_pos / dist_norm)
        
        if closing_speed > 0:
            ttc = dist_norm / closing_speed
            velocity_risk = max(velocity_risk, np.exp(-0.5 * ttc))

    # Finálna vážená suma
    total_risk = (0.6 * velocity_risk) + (0.4 * proximity_risk)
    return np.clip(total_risk, 0, 1)
