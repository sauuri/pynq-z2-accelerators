"""
PTQ Static Quantization for ResNet1D ECG classifier.
Usage:
    python quantize.py
"""

import argparse, os, copy, io
import torch
import numpy as np
from model import get_model
from dataset import get_dataloaders

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    parser.add_argument('--output',     default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    parser.add_argument('--calib-n',    type=int, default=2000)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = get_model()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Float model (test_acc={ckpt.get('test_acc',0):.2f}%, F1={ckpt.get('macro_f1',0):.4f})")

    # Fuse Conv+BN+ReLU
    model.fuse_model()

    # Quantize
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with real data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, _ = get_dataloaders(data_dir, batch_size=128, oversample=False)
    n_seen = 0
    model.eval()
    with torch.no_grad():
        for X, _ in train_loader:
            model(X); n_seen += X.size(0)
            if n_seen >= args.calib_n: break
    print(f"Calibrated with {n_seen} samples.")

    q_model = copy.deepcopy(model)
    torch.quantization.convert(q_model, inplace=True)

    def sz(m):
        b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(q_model.state_dict(), args.output)

    float_sz = os.path.getsize(args.checkpoint) / 1e6
    quant_sz  = sz(q_model)
    print(f"Float: {float_sz:.2f} MB → Quantized: {quant_sz:.2f} MB "
          f"(reduction: {(1-quant_sz/float_sz)*100:.1f}%)")
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
