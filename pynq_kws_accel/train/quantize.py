"""
PTQ Static Quantization for DS-CNN KWS model.
Usage:
    python quantize.py
"""

import argparse, os, copy, io
import torch
from model import get_model
from dataset import get_dataloaders, NUM_CLASSES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    parser.add_argument('--output',     default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    arch = ckpt.get('arch', 'dscnn')
    model = get_model(arch=arch, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    print(f"Float model [{arch}] (val_acc={ckpt.get('val_acc',0):.2f}%)")

    # Fuse (DS-CNN only — BCResNet fusing is more complex)
    if arch == 'dscnn' and hasattr(model, 'fuse_model'):
        model.fuse_model()
        print("Fused Conv+BN+ReLU.")

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, _ = get_dataloaders(data_dir, batch_size=64, num_workers=0, augment=False)
    n_seen = 0; model.eval()
    print("Calibrating...")
    with torch.no_grad():
        for X, _ in train_loader:
            model(X); n_seen += X.size(0)
            if n_seen >= 2000: break
    print(f"Calibrated with {n_seen} samples.")

    q_model = copy.deepcopy(model)
    torch.quantization.convert(q_model, inplace=True)

    def sz(m): b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6
    torch.save({'arch': arch, 'model_state_dict': q_model.state_dict(),
                'val_acc': ckpt.get('val_acc')}, args.output)
    print(f"Float: {sz(model):.2f} MB → Quantized: {sz(q_model):.2f} MB")
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
