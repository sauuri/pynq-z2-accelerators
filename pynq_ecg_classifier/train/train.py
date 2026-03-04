"""
ECG arrhythmia classification training.
Usage:
    python train.py --epochs 50
    python train.py --epochs 50 --lr 0.001 --batch-size 128
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import get_model
from dataset import get_dataloaders, CLASS_NAMES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X); loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * X.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += X.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        correct += preds.eq(y).sum().item(); total += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
    acc = 100.0 * correct / total

    # Per-class accuracy and F1
    n_cls = len(CLASS_NAMES)
    tp = np.zeros(n_cls); fp = np.zeros(n_cls); fn = np.zeros(n_cls)
    for p, t in zip(all_preds, all_targets):
        if p == t: tp[t] += 1
        else: fp[p] += 1; fn[t] += 1
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description='ECG arrhythmia classifier')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, test_loader = get_dataloaders(data_dir, args.batch_size)

    model = get_model().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet1D parameters: {total:,}")

    # Class-weighted loss for imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_acc, te_f1   = evaluate(model, test_loader, device)
        scheduler.step()
        macro_f1 = te_f1.mean()
        print(f"[{epoch:3d}/{args.epochs}] "
              f"train={tr_acc:.2f}% test={te_acc:.2f}% F1={macro_f1:.4f} "
              f"({time.time()-t0:.1f}s)")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': te_acc,
                'macro_f1': float(macro_f1),
                'per_class_f1': te_f1.tolist(),
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  -> Saved best (acc={te_acc:.2f}%, F1={macro_f1:.4f})")

    # Final per-class report
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    _, f1 = evaluate(model, test_loader, device)
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print("Per-class F1:")
    for name, score in zip(CLASS_NAMES, f1):
        print(f"  {name:<25} F1={score:.4f}")


if __name__ == '__main__':
    main()
