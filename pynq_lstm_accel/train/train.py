"""
UCI HAR training script for LSTM / CNN-LSTM classifier.
Usage:
    python train.py --epochs 50 --arch lstm
    python train.py --epochs 50 --arch cnn_lstm --lr 0.001
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from model import get_model
from dataset import get_dataloaders, ACTIVITY_LABELS

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += X.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += X.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def per_class_accuracy(model, loader, device):
    model.eval()
    n_classes = len(ACTIVITY_LABELS)
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        for c in range(n_classes):
            mask = y == c
            correct[c] += preds[mask].eq(y[mask]).sum().item()
            total[c] += mask.sum().item()
    return (correct / total.clamp(min=1)) * 100


def main():
    parser = argparse.ArgumentParser(description='Train LSTM on UCI HAR')
    parser.add_argument('--arch', type=str, default='lstm',
                        choices=['lstm', 'cnn_lstm'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    args = parser.parse_args()

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, test_loader, train_set = get_dataloaders(
        data_dir, batch_size=args.batch_size)

    model = get_model(
        arch=args.arch,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {args.arch} | Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)
        scheduler.step(test_acc)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
              f"({elapsed:.1f}s)")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'model_kwargs': {
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                },
                'norm_mean': train_set.mean,
                'norm_std': train_set.std,
                'test_acc': test_acc,
            }, save_path)
            print(f"  -> Saved best model (test_acc={test_acc:.2f}%)")

    # Per-class accuracy
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    per_class = per_class_accuracy(model, test_loader, device)
    print(f"\nBest model test accuracy: {best_acc:.2f}%")
    print("\nPer-class accuracy:")
    for label, acc in zip(ACTIVITY_LABELS, per_class):
        print(f"  {label:<25} {acc:.2f}%")


if __name__ == '__main__':
    main()
