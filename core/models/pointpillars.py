import torch
import torch.nn as nn
import torch.nn.functional as F

class PointPillarsConfig:
    x_range = (0, 200)
    y_range = (-50, 50)
    z_range = (-3, 3)

    voxel_size             = (0.2, 0.2, 6.0)
    max_points_per_pillar  = 32
    max_pillars            = 12000

    num_classes  = 3
    class_names  = ['Car', 'Pedestrian', 'Cyclist']
    anchors = {
        'Car':         [4.3,  1.95, 1.6,  0],
        'Pedestrian':  [0.55, 0.55, 1.65, 0],
        'Cyclist':     [1.76, 0.6,  1.65, 0],
    }

    learning_rate = 1e-3
    batch_size    = 4
    num_epochs    = 80 

class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, pillars, coords, num_points, batch_size):
        p = pillars.transpose(1, 2) 
        x = F.relu(self.bn(self.conv(p.unsqueeze(-1)))) 
        x = torch.max(x, dim=2)[0].squeeze(-1)         

        return x

class Backbone(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels, 64, 3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 128, 3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class DetectionHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=3, num_anchors=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors 

        self.cls_head = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        self.reg_head = nn.Conv2d(in_channels, 7 * num_anchors, 1)
        self.dir_head = nn.Conv2d(in_channels, 2 * num_anchors, 1)

    def forward(self, x):
        cls_preds = self.cls_head(x)
        reg_preds = self.reg_head(x)
        dir_preds = self.dir_head(x)
        return {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds
        }

class PointPillars(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pfn      = PillarFeatureNet()
        self.backbone = Backbone()
        self.head     = DetectionHead(num_classes=cfg.num_classes)
        self.cfg      = cfg

        self.nx = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.voxel_size[0])
        self.ny = int((cfg.y_range[1] - cfg.y_range[0]) / cfg.voxel_size[1])

    def forward(self, pillars, coords, num_points, batch_size):
        B, P, N, C = pillars.shape
        pillars = pillars.view(B * P, N, C)
        
        features = self.pfn(pillars, coords, num_points, batch_size) 
        
        canvas = torch.zeros(batch_size, 64, self.ny, self.nx, device=features.device)
        
        for b in range(batch_size):
            batch_mask = (coords[b].sum(dim=1) != 0) 
            b_coords = coords[b][batch_mask].long()
            b_features = features.view(batch_size, P, 64)[b][batch_mask]
            
            canvas[b, :, b_coords[:, 1], b_coords[:, 0]] = b_features.t()

        x = self.backbone(canvas)
        out = self.head(x)
        
        return out

if __name__ == '__main__':
    cfg = PointPillarsConfig()
    model = PointPillars(cfg)
    print("PointPillars model úspešne inicializovaný.")
