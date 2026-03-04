"""
Post-Training Static Quantization for TinyViT.
Note: LayerNorm and Softmax in Transformers require special handling.
We quantize Linear layers and patch embedding Conv2d.

Usage:
    python quantize.py
"""

import argparse
import os
import copy

import torch
import torchvision
import torchvision.transforms as transforms

from model import get_model

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_calib_loader(n=1000, batch=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    sub = torch.utils.data.Subset(ds, range(n))
    return torch.utils.data.DataLoader(sub, batch_size=batch, shuffle=False)


def calibrate(model, loader):
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    parser.add_argument('--output',     default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    parser.add_argument('--calib-n',    type=int, default=1000)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_kwargs = ckpt.get('model_kwargs', {})
    model = get_model(**model_kwargs)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded float model (val_acc={ckpt.get('val_acc', 0):.2f}%)")

    # For Transformers: quantize only linear + conv layers
    # LayerNorm stays in FP32 (quantization-unfriendly)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare
    torch.quantization.prepare(model, inplace=True)
    print("Calibrating...")
    calibrate(model, get_calib_loader(args.calib_n))

    # Convert
    q_model = copy.deepcopy(model)
    torch.quantization.convert(q_model, inplace=True)
    print("Converted to INT8.")

    torch.save(q_model.state_dict(), args.output)
    print(f"Saved: {args.output}")

    # Size comparison
    import io
    def sz(m):
        b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6
    print(f"Float: {os.path.getsize(args.checkpoint)/1e6:.2f} MB → Quantized: {sz(q_model):.2f} MB")


if __name__ == '__main__':
    main()
