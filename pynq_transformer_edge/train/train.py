"""
CIFAR-10 training script for TinyViT.
Usage:
    python train.py --epochs 100 --lr 3e-4 --batch-size 128
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import get_model

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_dataloaders(batch_size, num_workers=4):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    val_set   = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            out = model(x); loss = criterion(out, y)
            loss.backward(); optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x); loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return loss_sum / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--embed-dim',  type=int,   default=64)
    parser.add_argument('--depth',      type=int,   default=6)
    parser.add_argument('--warmup',     type=int,   default=10)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders(args.batch_size)
    model = get_model(embed_dim=args.embed_dim, depth=args.depth).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # Warmup + cosine decay
    def lr_lambda(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        progress = (ep - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"[{epoch:3d}/{args.epochs}] tr={tr_acc:.2f}% va={va_acc:.2f}% "
              f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'model_kwargs': {'embed_dim': args.embed_dim, 'depth': args.depth},
                        'val_acc': va_acc}, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  -> Best saved ({va_acc:.2f}%)")

    print(f"\nBest val accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
