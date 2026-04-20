import numpy as np
from scipy.optimize import linear_sum_assignment
from enum import Enum
from utils.roi_utils import filter_detections_by_roi

class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

class KalmanFilter3D:
    """Stav: [x, y, z, l, w, h, rot, vx, vy, vz]"""
    def __init__(self, bbox: np.ndarray):
        self.F = np.eye(10, dtype=np.float32)
        self.F[0, 7] = 1.0  # x = x + vx
        self.F[1, 8] = 1.0  # y = y + vy
        self.F[2, 9] = 1.0  # z = z + vz

        self.H = np.zeros((7, 10), dtype=np.float32)
        self.H[:7, :7] = np.eye(7)

        self.Q = np.eye(10, dtype=np.float32)
        self.Q[:3, :3] *= 0.1
        self.Q[3:7, 3:7] *= 0.05
        self.Q[7:, 7:] *= 0.1

        self.R = np.eye(7, dtype=np.float32)
        self.R[:3, :3] *= 0.5
        self.R[3:6, 3:6] *= 1.0
        self.R[6, 6] *= 0.5

        self.P = np.eye(10, dtype=np.float32) * 10.0
        self.x = np.zeros(10, dtype=np.float32)
        self.x[:7] = bbox

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:7]

    def update(self, bbox: np.ndarray):
        y = bbox - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(10) - (K @ self.H)) @ self.P

class Track:
    def __init__(self, bbox, class_id, score, track_id):
        self.id = track_id
        self.class_id = int(class_id)
        self.score = score
        self.kf = KalmanFilter3D(bbox)
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history = []
        self.smoothed_risk = 0.0
        self.alpha_risk = 0.2

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        pos = self.kf.predict()
        self.history.append(pos[:3].copy())
        if len(self.history) > 50:
            self.history.pop(0)
        return pos

    def update(self, bbox):
        self.hits += 1
        self.time_since_update = 0
        self.kf.update(bbox)

    def update_risk(self, new_risk):
        self.smoothed_risk = (1 - self.alpha_risk) * self.smoothed_risk + self.alpha_risk * new_risk

class Tracker3D:
    def __init__(self, max_age=10, min_hits=3, dist_threshold=4.0, roi_coords=None):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.roi_coords = roi_coords
        self.tracks = []
        self._next_id = 1

    def update(self, detections: np.ndarray):
        if self.roi_coords is not None:
            detections = filter_detections_by_roi(detections, self.roi_coords)

        predicted_states = [t.predict() for t in self.tracks]

        matched, unmatched_dets, unmatched_trks = self._associate(detections, predicted_states)

        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(detections[d_idx][:7])
            if self.tracks[t_idx].state == TrackState.TENTATIVE and self.tracks[t_idx].hits >= self.min_hits:
                self.tracks[t_idx].state = TrackState.CONFIRMED

        for d_idx in unmatched_dets:
            d = detections[d_idx]
            new_track = Track(d[:7], d[7], d[8], self._next_id)
            self.tracks.append(new_track)
            self._next_id += 1

        for t_idx in unmatched_trks:
            if self.tracks[t_idx].time_since_update > self.max_age:
                self.tracks[t_idx].state = TrackState.DELETED

        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def _associate(self, detections, predicted):
        n_det, n_trk = len(detections), len(self.tracks)
        if n_det == 0 or n_trk == 0:
            return [], list(range(n_det)), list(range(n_trk))

        cost = np.full((n_det, n_trk), 1e6, dtype=np.float32)

        for d_i, det in enumerate(detections):
            for t_i, trk in enumerate(self.tracks):
                if int(det[7]) != trk.class_id:
                    continue
                dist = np.linalg.norm(det[:2] - predicted[t_i][:2])
                if dist < self.dist_threshold:
                    cost[d_i, t_i] = dist

        row_ind, col_ind = linear_sum_assignment(cost)
        
        matched = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e6:
                matched.append((r, c))
            
        unmatched_dets = [i for i in range(n_det) if i not in [m[0] for m in matched]]
        unmatched_trks = [i for i in range(n_trk) if i not in [m[1] for m in matched]]
        
        return matched, unmatched_dets, unmatched_trks
