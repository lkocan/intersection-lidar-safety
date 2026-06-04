import os
import json
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

_CORE_DIR    = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _CORE_DIR.parent

DAIR_ROOT  = PROJECT_ROOT / 'data'
PCD_DIR    = DAIR_ROOT / 'pcd'
LABEL_DIR  = DAIR_ROOT / 'label' / 'virtuallidar'
SPLIT_FILE = DAIR_ROOT / 'split_data.json'
GT_DB_DIR  = PROJECT_ROOT / 'gt_database'


def _p(path_obj) -> str:
    return str(path_obj)


@dataclass
class Box3D:
    obj_type: str
    x:        float
    y:        float
    z:        float
    length:   float
    width:    float
    height:   float
    rotation: float

    CLASS_MAP = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    @property
    def class_id(self) -> int:
        return self.CLASS_MAP.get(self.obj_type, -1)


def load_labels(label_path: str) -> List[Box3D]:
    if not os.path.exists(label_path):
        return []
    with open(label_path, encoding='utf-8') as f:
        data = json.load(f)
    boxes = []
    for obj in data:
        obj_type = obj.get('type', '')
        if obj_type not in Box3D.CLASS_MAP:
            continue
        loc  = obj['3d_location']
        dims = obj['3d_dimensions']
        boxes.append(Box3D(
            obj_type = obj_type,
            x        = float(loc['x']),
            y        = float(loc['y']),
            z        = float(loc['z']),
            length   = float(dims['l']),
            width    = float(dims['w']),
            height   = float(dims['h']),
            rotation = float(obj.get('rotation', 0.0)),
        ))
    return boxes


class GTSampler:
    def __init__(self, db_path: Path = GT_DB_DIR):
        self.db_path   = db_path
        self.db_info   = {}
        self.class_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        info_path      = db_path / 'gt_database_info.pkl'
        if not info_path.exists():
            print(f"[GTSampler] {info_path} nenájdený — augmentácia vypnutá")
            return
        with open(_p(info_path), 'rb') as f:
            self.db_info = pickle.load(f)

    def sample(self, existing_boxes: np.ndarray, max_ped: int = 8, max_cyc: int = 8):
        sampled_pts, sampled_boxes = [], []
        all_boxes = existing_boxes.copy() if len(existing_boxes) else np.empty((0, 8))
        for cls_name, max_count in [('Pedestrian', max_ped), ('Cyclist', max_cyc)]:
            available = self.db_info.get(cls_name, [])
            if not available:
                continue
            n       = min(len(available), max_count)
            choices = np.random.choice(len(available), n, replace=False)
            for idx in choices:
                info    = available[idx]
                pts     = np.load(_p(self.db_path / info['filepath']))
                l, w, h = info['box_dims']
                x       = float(np.random.uniform(10, 70))
                y       = float(np.random.uniform(-35, 35))
                z       = -1.5
                yaw     = float(np.random.uniform(0, 2 * np.pi))
                if len(all_boxes):
                    dists = np.linalg.norm(all_boxes[:, :2] - [x, y], axis=1)
                    if dists.min() < 3.5:
                        continue
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                R = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
                new_pts = pts.copy()
                new_pts[:, :3] = pts[:, :3] @ R.T + [x, y, z]
                new_box = np.array([x, y, z, l, w, h, yaw,
                                    self.class_map[cls_name]], dtype=np.float32)
                sampled_pts.append(new_pts)
                sampled_boxes.append(new_box)
                all_boxes = np.vstack([all_boxes, new_box])
        return sampled_pts, sampled_boxes


class DAIRDataset(Dataset):
    def __init__(self, split: str = 'train'):
        import sys
        sys.path.insert(0, _p(_CORE_DIR))
        from utils.preprocess import (
            DAIRPillarConfig, filter_points,
            filter_ground_plane, create_pillars, load_pcd,
        )
        self.cfg             = DAIRPillarConfig()
        self._filter_points  = filter_points
        self._filter_ground  = filter_ground_plane
        self._create_pillars = create_pillars
        self._load_pcd       = load_pcd

        if not SPLIT_FILE.exists():
            raise FileNotFoundError(
                f"Split file nenájdený: {SPLIT_FILE}\n"
                f"Skontroluj cestu k datasetu: {DAIR_ROOT}"
            )

        with open(_p(SPLIT_FILE)) as f:
            all_ids = json.load(f)[split]

        self.split      = split
        self.ids        = [fid for fid in all_ids if (PCD_DIR / f'{fid}.pcd').exists()]
        self.gt_sampler = GTSampler() if split == 'train' else None

        print(f'[DAIRDataset] {split}: {len(self.ids)} vzoriek')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        fid    = self.ids[idx]
        points = self._load_pcd(_p(PCD_DIR / f'{fid}.pcd'))
        points = self._filter_points(points, self.cfg)
        points = self._filter_ground(points, self.cfg)

        boxes    = load_labels(_p(LABEL_DIR / f'{fid}.json'))
        gt_boxes = (
            np.array(
                [[b.x, b.y, b.z, b.length, b.width, b.height, b.rotation, b.class_id]
                 for b in boxes if b.class_id >= 0],
                dtype=np.float32,
            ) if boxes else np.zeros((0, 8), dtype=np.float32)
        )

        if self.gt_sampler is not None:
            s_pts, s_boxes = self.gt_sampler.sample(gt_boxes)
            if s_pts:
                points   = np.vstack([points] + s_pts)
                gt_boxes = (np.vstack([gt_boxes, s_boxes])
                            if len(gt_boxes) else np.array(s_boxes, dtype=np.float32))

        pillars, coords, num_pts = self._create_pillars(points, self.cfg)

        return {
            'pillars':    torch.from_numpy(pillars),
            'coords':     torch.from_numpy(coords),
            'num_points': torch.from_numpy(num_pts),
            'gt_boxes':   torch.from_numpy(gt_boxes),
            'frame_id':   fid,
            'raw_points': points,
        }