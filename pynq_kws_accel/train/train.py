"""
KWS training on Google Speech Commands.
Usage:
    python train.py --arch dscnn --epochs 50
    python train.py --arch bcresnet --epochs 60
"""

import argparse, os, time
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X); loss = criterion(out, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * X.size(0)
        correct += out.argmax(1).eq(y).sum().item(); total += X.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        correct += model(X).argmax(1).eq(y).sum().item(); total += y.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',       default='dscnn', choices=['dscnn', 'bcresnet'])
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=0.01)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--no-augment', action='store_true')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}, arch: {args.arch}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, val_loader = get_dataloaders(
        data_dir, args.batch_size, augment=not args.no_augment)

    model = get_model(arch=args.arch, num_classes=NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"[{epoch:3d}/{args.epochs}] train={tr_acc:.2f}% val={va_acc:.2f}% "
              f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'epoch': epoch, 'arch': args.arch,
                        'model_state_dict': model.state_dict(), 'val_acc': va_acc},
                       os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  -> Best ({va_acc:.2f}%)")

    print(f"\nBest val accuracy: {best_acc:.2f}%")
    print(f"Classes: {CLASS_NAMES}")


if __name__ == '__main__':
    main()
