import numpy as np
import torch
import open3d as o3d
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class PillarConfig:
    voxel_size: Tuple[float, float] = (0.16, 0.16)

    x_range: Tuple[float, float] = (-70.4, 70.4)
    y_range: Tuple[float, float] = (-40.0, 40.0)
    z_range: Tuple[float, float] = (-3.0,   3.0)

    max_points_per_pillar: int = 32
    max_pillars: int = 12000

    normalize_intensity: bool = True

    remove_ground: bool = True           
    ground_threshold: float = 0.25       
    ransac_iterations: int = 250

@dataclass
class DAIRPillarConfig(PillarConfig):
    voxel_size: Tuple[float, float] = (0.2, 0.2)
    x_range: Tuple[float, float] = (0.0, 200.0)
    y_range: Tuple[float, float] = (-50.0, 50.0)
    z_range: Tuple[float, float] = (-3.0, 3.0)
    max_pillars: int = 12000
    remove_ground: bool = False

def filter_ground_plane(points: np.ndarray, cfg: PillarConfig) -> np.ndarray:
    if len(points) < 100 or not cfg.remove_ground:
        return points

    z_mask = points[:, 2] < (cfg.z_range[0] + 1.5)
    floor_candidates = points[z_mask]
    
    if len(floor_candidates) < 100:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(floor_candidates[:, :3])
    
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=cfg.ground_threshold,
            ransac_n=3,
            num_iterations=cfg.ransac_iterations
        )
        
        a, b, c, d = plane_model

        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        
        mask_not_ground = distances > cfg.ground_threshold
        
        return points[mask_not_ground]
    
    except Exception as e:
        print(f"[Varovanie] RANSAC filter zlyhal, ignorujem: {e}")
        return points


def filter_points(points: np.ndarray, cfg: PillarConfig) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    mask = (
        (x >= cfg.x_range[0]) & (x <= cfg.x_range[1]) &
        (y >= cfg.y_range[0]) & (y <= cfg.y_range[1]) &
        (z >= cfg.z_range[0]) & (z <= cfg.z_range[1])
    )
    pts = points[mask].astype(np.float32)

    if cfg.normalize_intensity and len(pts) > 0:
        intensity = pts[:, 3]
        if intensity.max() > 1.0:
            pts[:, 3] = intensity / 255.0

    return pts

def create_pillars(
    points: np.ndarray,
    cfg: PillarConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    P_max = cfg.max_pillars
    M = cfg.max_points_per_pillar
    dx, dy = cfg.voxel_size
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range

    empty = (
        np.zeros((P_max, M, 9), dtype=np.float32),
        np.zeros((P_max, 2),    dtype=np.int32),
        np.zeros(P_max,         dtype=np.int32),
    )

    if len(points) == 0:
        return empty

    pts = points.astype(np.float32)

    nx = int((x_max - x_min) / dx)
    ny = int((y_max - y_min) / dy)

    ix = ((pts[:, 0] - x_min) / dx).astype(np.int32)
    iy = ((pts[:, 1] - y_min) / dy).astype(np.int32)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not valid.any():
        return empty

    pts, ix, iy = pts[valid], ix[valid], iy[valid]

    key = iy * nx + ix  

    order = np.argsort(key, kind='stable')
    pts, key = pts[order], key[order]
    ix, iy = ix[order], iy[order]

    unique_keys, inverse, counts = np.unique(
        key, return_inverse=True, return_counts=True
    )
    P_actual = len(unique_keys)

    if P_actual > P_max:
        top_pids = np.argsort(-counts)[:P_max]
        keep = np.isin(inverse, top_pids)
        pts, key, ix, iy = pts[keep], key[keep], ix[keep], iy[keep]
        unique_keys, inverse, counts = np.unique(
            key, return_inverse=True, return_counts=True
        )
        P_actual = len(unique_keys)

    offsets = np.concatenate([[0], np.cumsum(counts)])  

    within_idx = np.arange(len(pts), dtype=np.int32) - offsets[inverse]

    keep = within_idx < M
    pts_f      = pts[keep]
    within_f   = within_idx[keep]
    inv_f      = inverse[keep]

    if len(pts_f) == 0:
        return empty

    p_ix = (unique_keys % nx).astype(np.int32)
    p_iy = (unique_keys // nx).astype(np.int32)
    cx = (x_min + (p_ix + 0.5) * dx).astype(np.float32)   
    cy = (y_min + (p_iy + 0.5) * dy).astype(np.float32)

    z_sum = np.zeros(P_actual, dtype=np.float64)
    np.add.at(z_sum, inverse, pts[:, 2].astype(np.float64))
    z_mean = (z_sum / counts).astype(np.float32)          

    gm_x = float(pts[:, 0].mean())
    gm_y = float(pts[:, 1].mean())

    N_f = len(pts_f)
    feat = np.empty((N_f, 9), dtype=np.float32)
    feat[:, 0] = pts_f[:, 0]                       # x
    feat[:, 1] = pts_f[:, 1]                       # y
    feat[:, 2] = pts_f[:, 2]                       # z
    feat[:, 3] = pts_f[:, 3]                       # intensity
    feat[:, 4] = pts_f[:, 0] - cx[inv_f]           # Δx od centra pilára
    feat[:, 5] = pts_f[:, 1] - cy[inv_f]           # Δy od centra pilára
    feat[:, 6] = pts_f[:, 2] - z_mean[inv_f]       # Δz od priemerného z
    feat[:, 7] = pts_f[:, 0] - gm_x                # Δx od globálneho stredu
    feat[:, 8] = pts_f[:, 1] - gm_y                # Δy od globálneho stredu

    pillar_out = np.zeros((P_max, M, 9), dtype=np.float32)
    pillar_out[inv_f, within_f, :] = feat

    coords_out = np.zeros((P_max, 2), dtype=np.int32)
    coords_out[:P_actual, 0] = p_ix
    coords_out[:P_actual, 1] = p_iy

    num_pts_out = np.zeros(P_max, dtype=np.int32)
    num_pts_out[:P_actual] = np.minimum(counts, M)

    return pillar_out, coords_out, num_pts_out

class PointCloudPreprocessor:
    def __init__(
        self,
        cfg: PillarConfig = None,
        device: str = 'cpu',
    ):
        self.cfg = cfg or PillarConfig()
        self.device = torch.device(device)

    def __call__(self, points: np.ndarray) -> dict:
        pts = filter_points(points, self.cfg)
        pts = filter_ground_plane(pts, self.cfg)
        pillars, coords, num_pts = create_pillars(pts, self.cfg)

        return {
            'pillars':    torch.from_numpy(pillars).unsqueeze(0).to(self.device),
            'coords':     torch.from_numpy(coords).unsqueeze(0).to(self.device),
            'num_points': torch.from_numpy(num_pts).unsqueeze(0).to(self.device),
        }

    def warmup(self, n_frames: int = 5):
        dummy = np.random.randn(20000, 4).astype(np.float32)
        dummy[:, 3] = np.abs(dummy[:, 3])
        for _ in range(n_frames):
            _ = self(dummy)
        print(f"[Preprocessor] Warmup hotový ({n_frames} frames)")

    def benchmark(self, n_frames: int = 100) -> dict:
        import time
        dummy = np.random.randn(25000, 4).astype(np.float32)
        dummy[:, 3] = np.abs(dummy[:, 3])

        self.warmup(n_frames=3)

        times = []
        for _ in range(n_frames):
            t0 = time.perf_counter()
            _ = self(dummy)
            times.append((time.perf_counter() - t0) * 1000)

        arr = np.array(times)
        result = {
            'mean_ms':  float(arr.mean()),
            'p50_ms':   float(np.percentile(arr, 50)),
            'p95_ms':   float(np.percentile(arr, 95)),
            'max_ms':   float(arr.max()),
        }
        print(
            f"[Preprocessor] Benchmark ({n_frames} frames, ~25k bodov): "
            f"mean={result['mean_ms']:.1f}ms  "
            f"p95={result['p95_ms']:.1f}ms  "
            f"max={result['max_ms']:.1f}ms"
        )
        return result

def load_pcd(pcd_path: str) -> np.ndarray:
    import os
    import struct

    if not os.path.exists(pcd_path):
        return np.zeros((0, 4), dtype=np.float32)

    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            if line.startswith('DATA'):
                data_type = line.split()[1].strip()
                break
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]

        num_points = int(header.get('POINTS', [0])[0])
        fields  = header.get('FIELDS', ['x', 'y', 'z'])
        sizes   = [int(s) for s in header.get('SIZE',  ['4'] * len(fields))]
        types   = header.get('TYPE',  ['F'] * len(fields))
        counts  = [int(c) for c in header.get('COUNT', ['1'] * len(fields))]

        col_offset = {}
        offset = 0
        for name, cnt in zip(fields, counts):
            col_offset[name] = offset
            offset += cnt
        total_cols = offset

        fmt_char = {
            ('F', 4): 'f', ('F', 8): 'd',
            ('I', 4): 'i', ('I', 2): 'h', ('I', 1): 'b',
            ('U', 4): 'I', ('U', 2): 'H', ('U', 1): 'B',
        }

        if data_type == 'ascii':
            rows = []
            for _ in range(num_points):
                row = f.readline().decode('utf-8', errors='ignore').strip().split()
                if row:
                    rows.append([float(v) for v in row])
            data = (
                np.array(rows, dtype=np.float32) if rows
                else np.zeros((0, total_cols), dtype=np.float32)
            )

        elif data_type == 'binary_compressed':
            compressed_size   = struct.unpack('<I', f.read(4))[0]
            uncompressed_size = struct.unpack('<I', f.read(4))[0]
            raw = _lzf_decompress(f.read(compressed_size), uncompressed_size)
            columns, byte_pos = [], 0
            for t, s, c, _name in zip(types, sizes, counts, fields):
                ch        = fmt_char.get((t, s), 'f')
                col_bytes = s * c * num_points
                col_data  = np.frombuffer(
                    raw[byte_pos: byte_pos + col_bytes],
                    dtype=np.dtype('<' + ch)
                ).reshape(num_points, c).astype(np.float32)
                columns.append(col_data)
                byte_pos += col_bytes
            data = np.concatenate(columns, axis=1)

        else:
            row_fmt  = '<' + ''.join(
                fmt_char.get((t, s), 'f') * c
                for t, s, c in zip(types, sizes, counts)
            )
            row_size = struct.calcsize(row_fmt)
            raw      = f.read(row_size * num_points)
            usable   = (len(raw) // row_size) * row_size
            if usable == 0:
                return np.zeros((0, 4), dtype=np.float32)
            data = np.array(
                list(struct.iter_unpack(row_fmt, raw[:usable])),
                dtype=np.float32,
            )

    if data.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x = data[:, col_offset.get('x', 0)]
    y = data[:, col_offset.get('y', 1)]
    z = data[:, col_offset.get('z', 2)]
    if 'intensity' in col_offset:
        intensity = data[:, col_offset['intensity']]
    else:
        intensity = np.zeros(len(x), dtype=np.float32)
    return np.stack([x, y, z, intensity], axis=1)


def _lzf_decompress(data: bytes, output_size: int) -> bytes:
    output  = bytearray(output_size)
    in_pos  = 0
    out_pos = 0
    while in_pos < len(data):
        ctrl = data[in_pos]
        in_pos += 1
        if ctrl < 32:
            length = ctrl + 1
            output[out_pos: out_pos + length] = data[in_pos: in_pos + length]
            in_pos  += length
            out_pos += length
        else:
            length = ctrl >> 5
            if length == 7:
                length += data[in_pos]
                in_pos += 1
            length += 2
            ref_offset = ((ctrl & 0x1F) << 8) + data[in_pos] + 1
            in_pos += 1
            ref_pos = out_pos - ref_offset
            for i in range(length):
                output[out_pos] = output[ref_pos + i]
                out_pos += 1
    return bytes(output)

if __name__ == '__main__':
    print("=== Smoke test: PointCloudPreprocessor ===\n")

    preproc_prod  = PointCloudPreprocessor(cfg=PillarConfig(),     device='cpu')
    preproc_dair  = PointCloudPreprocessor(cfg=DAIRPillarConfig(), device='cpu')

    dummy = np.random.randn(25000, 4).astype(np.float32)
    dummy[:, 3] = np.abs(dummy[:, 3])

    out = preproc_prod(dummy)
    print("Produkčná konfigurácia (360° ROI):")
    print(f"  pillars:    {out['pillars'].shape}")
    print(f"  coords:     {out['coords'].shape}")
    print(f"  num_points: {out['num_points'].shape}")

    out2 = preproc_dair(dummy)
    print("\nDAIR konfigurácia (0–200 m ROI):")
    print(f"  pillars:    {out2['pillars'].shape}")

    print()
    preproc_prod.benchmark(n_frames=50)