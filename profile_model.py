#!/usr/bin/env python3
"""Count parameters and FLOPs of the headline model (resize head + strip-pooling).

The network reads its module switches from environment variables AT IMPORT TIME,
so they are set here before importing the network. Headline config matches
make_figures.sh's OURS row:  HEAD=resize USE_SP=1 SP_MODE=add USE_BIM=0 USE_AG=0.

FLOPs are reported as GMACs at 1x3x224x224 (the convention these segmentation
papers use when they print "#FLOPs (G)"). Run in the `swinunet` env:

    conda run -n swinunet python profile_model.py
"""
import os
# --- must be set BEFORE importing the network (switches are module-level) ---
os.environ["HEAD"] = "resize"
os.environ["USE_SP"] = "1"
os.environ["SP_MODE"] = "add"
os.environ["USE_BIM"] = "0"
os.environ["USE_AG"] = "0"

import argparse
import torch
from config import get_config
from networks.vision_transformer import SwinUnet

CFG = "configs/swin_tiny_patch4_window7_224_lite.yaml"
IMG, NCLS = 224, 9   # Synapse: 8 organs + background


def build():
    args = argparse.Namespace(
        cfg=CFG, opts=None, batch_size=24, zip=False, cache_mode="part",
        resume=None, accumulation_steps=None, use_checkpoint=False,
        amp_opt_level="O1", tag=None, eval=False, throughput=False,
    )
    cfg = get_config(args)
    net = SwinUnet(cfg, img_size=IMG, num_classes=NCLS).eval()
    return net


def main():
    net = build()
    x = torch.randn(1, 3, IMG, IMG)

    total = sum(p.numel() for p in net.parameters())
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\n=== Ours (resize head + strip-pooling), input 1x3x{IMG}x{IMG}, {NCLS} classes ===")
    print(f"Params      : {total/1e6:.2f} M  (trainable {train/1e6:.2f} M)")

    # encoder vs decoder vs head/context split, by top-level submodule of swin_unet
    su = net.swin_unet
    for name, mod in su.named_children():
        p = sum(t.numel() for t in mod.parameters())
        if p:
            print(f"   - {name:<16} {p/1e6:6.2f} M")

    # FLOPs (GMACs) via fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(net, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        print(f"FLOPs(fvcore): {fca.total()/1e9:.2f} G (MACs)")
    except Exception as e:
        print("fvcore failed:", repr(e))

    # cross-check with thop
    try:
        from thop import profile
        macs, params = profile(net, inputs=(x,), verbose=False)
        print(f"FLOPs(thop)  : {macs/1e9:.2f} G (MACs) | params {params/1e6:.2f} M")
    except Exception as e:
        print("thop failed:", repr(e))


if __name__ == "__main__":
    main()
