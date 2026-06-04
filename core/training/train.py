import sys
import os
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # = core/
sys.path.insert(0, str(BASE_DIR))

from models.pointpillars import PointPillars, PointPillarsConfig
from training.loss        import PointPillarsLoss
from utils.dair_dataset   import DAIRDataset           

def collate_fn(batch):
    return {
        'pillars':    torch.stack([b['pillars']    for b in batch]).float(),
        'coords':     torch.stack([b['coords']     for b in batch]).int(),
        'num_points': torch.stack([b['num_points'] for b in batch]).int(),
        'gt_boxes':   [b['gt_boxes'].float() for b in batch],
        'frame_id':   [b['frame_id']          for b in batch],
    }

def train():
    cfg = PointPillarsConfig()

    if torch.cuda.is_available():
        device  = torch.device('cuda')
        use_amp = True
    elif torch.backends.mps.is_available():
        device  = torch.device('mps')
        use_amp = False
    else:
        device  = torch.device('cpu')
        use_amp = False

    print(f"\n{'='*50}")
    print(f"  PointPillars Training")
    print(f"  Device   : {device}  |  AMP: {use_amp}")
    print(f"  Batch    : {cfg.batch_size}  |  Epochs: {cfg.num_epochs}")
    print(f"  LR       : {cfg.learning_rate}")
    print(f"{'='*50}\n")

    
    n_workers = 0   
    if device.type == 'cuda':
        n_workers = int(os.environ.get('NUM_WORKERS', 4))
    elif os.environ.get('COLAB_GPU'):
        n_workers = 2

    train_ds = DAIRDataset(split='train')
    val_ds   = DAIRDataset(split='val')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(n_workers > 0),
        drop_last=True,      
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn,   
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(n_workers > 0),
    )

    print(f"Train: {len(train_ds)} vzoriek  "
          f"({len(train_loader)} batchov)\n"
          f"Val  : {len(val_ds)} vzoriek  "
          f"({len(val_loader)} batchov)\n")

    model     = PointPillars(cfg).to(device)
    criterion = PointPillarsLoss(cfg).to(device)   

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametre modelu: {total_params:,}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=cfg.num_epochs,
        pct_start=0.05,         
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == 'cuda'))

    save_dir = BASE_DIR / 'checkpoints'
    save_dir.mkdir(exist_ok=True)

    best_val_loss = float('inf')
    start_epoch   = 0

    last_ckpt = save_dir / 'last.pth'
    if last_ckpt.exists():
        ckpt        = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"[Resume] Načítaný checkpoint z epochy {start_epoch}, "
              f"val_loss={best_val_loss:.4f}\n")

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        t_start      = time.time()
        epoch_losses = {'total': [], 'cls': [], 'reg': [], 'dir': []}
        avg_pos      = []

        for i, batch in enumerate(train_loader):
            pillars    = batch['pillars'].to(device, non_blocking=True)
            coords     = batch['coords'].to(device, non_blocking=True)
            num_points = batch['num_points'].to(device, non_blocking=True)
            gt_boxes   = [g.to(device) for g in batch['gt_boxes']]

            optimizer.zero_grad(set_to_none=True)

            autocast_device = device.type if device.type != 'mps' else 'cpu'
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                preds  = model(pillars, coords, num_points,
                               batch_size=pillars.shape[0])
                losses = criterion(preds, gt_boxes,
                                   batch_size=pillars.shape[0])

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k].append(losses[k].item())
            avg_pos.append(losses.get('num_pos', 0))

            if i % 20 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  E{epoch+1:03d} [{i:4d}/{len(train_loader)}]"
                    f"  loss={losses['total'].item():.4f}"
                    f"  cls={losses['cls'].item():.4f}"
                    f"  reg={losses['reg'].item():.4f}"
                    f"  dir={losses['dir'].item():.4f}"
                    f"  pos={losses.get('num_pos', 0):4d}"
                    f"  lr={lr_now:.2e}"
                )

        model.eval()
        val_losses = {'total': [], 'cls': [], 'reg': [], 'dir': []}

        with torch.no_grad():
            for batch in val_loader:
                pillars    = batch['pillars'].to(device)
                coords     = batch['coords'].to(device)
                num_points = batch['num_points'].to(device)
                gt_boxes   = [g.to(device) for g in batch['gt_boxes']]

                preds  = model(pillars, coords, num_points,
                               batch_size=pillars.shape[0])
                losses = criterion(preds, gt_boxes,
                                   batch_size=pillars.shape[0])

                for k in val_losses:
                    val_losses[k].append(losses[k].item())

        avg_train = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        avg_val   = {k: sum(v) / len(v) for k, v in val_losses.items()}
        elapsed   = time.time() - t_start

        print(
            f"\nEpoch {epoch+1:03d}/{cfg.num_epochs}"
            f"  [{elapsed:.0f}s]"
            f"  train={avg_train['total']:.4f}"
            f"  val={avg_val['total']:.4f}"
            f"  (cls={avg_val['cls']:.4f}"
            f"  reg={avg_val['reg']:.4f}"
            f"  dir={avg_val['dir']:.4f})"
            f"  avg_pos={sum(avg_pos)/len(avg_pos):.0f}"
        )

        checkpoint = {
            'epoch':      epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_loss':   avg_val['total'],
            'cfg':        cfg.__dict__,
        }

        torch.save(checkpoint, save_dir / 'last.pth')

        if avg_val['total'] < best_val_loss:
            best_val_loss = avg_val['total']
            torch.save(checkpoint, save_dir / 'best.pth')
            print(f"Nový best model  (val_loss={best_val_loss:.4f})\n")
        else:
            print()

    print(f"\nTréning hotový. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpointy: {save_dir}")


if __name__ == '__main__':
    train()