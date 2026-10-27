"""Train the tabular baseline or the multimodal model.

  python train.py --model tabular    --features physical
  python train.py --model multimodal --features physical --backbone efficientnet_b0
"""
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch

from cyclone_mm import config as C
from cyclone_mm.data import CycloneDataset, TabularPreprocessor, load_samples, make_loader, split
from cyclone_mm.engine import fit, predict, regression_metrics
from cyclone_mm.models import MultimodalModel, TabularModel


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(kind: str, n_features: int, backbone: str, pretrained: bool = True):
    if kind == "tabular":
        return TabularModel(n_features)
    return MultimodalModel(n_features, backbone=backbone, pretrained=pretrained)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["tabular", "multimodal"], required=True)
    ap.add_argument("--features", choices=list(C.FEATURE_SETS), default="physical")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--split", choices=["storm", "temporal"], default="storm")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="head / MLP learning rate")
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--no-flips", action="store_true", help="rotation-only augmentation")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="run directory name under outputs/")
    args = ap.parse_args()

    mm = args.model == "multimodal"
    epochs = args.epochs or (25 if mm else 300)
    bs = args.batch_size or (32 if mm else 128)
    lr = args.lr or 1e-3
    run = args.name or (f"{args.model}_{args.features}" + (f"_{args.backbone}" if mm else ""))
    out = C.OUTPUTS / run
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    features = C.FEATURE_SETS[args.features]
    df = load_samples(require_images=True)
    train_df, val_df, test_df = split(df, args.split, seed=args.seed)
    print(f"[{run}] train {len(train_df)} ({train_df.storm_id.nunique()} storms)  "
          f"val {len(val_df)} ({val_df.storm_id.nunique()})  test {len(test_df)} ({test_df.storm_id.nunique()})  "
          f"device {device}")

    prep = TabularPreprocessor(features).fit(train_df)
    flips = not args.no_flips
    tr_ds = CycloneDataset(train_df, prep, with_images=mm, train=True, flips=flips)
    va_ds = CycloneDataset(val_df, prep, with_images=mm)
    te_ds = CycloneDataset(test_df, prep, with_images=mm)
    w = args.workers if mm else 0
    tr_dl = make_loader(tr_ds, bs, shuffle=True, workers=w)
    va_dl = make_loader(va_ds, bs * 2, shuffle=False, workers=w)
    te_dl = make_loader(te_ds, bs * 2, shuffle=False, workers=w)

    model = build_model(args.model, len(features), args.backbone).to(device)
    params = model.param_groups(args.lr_backbone, lr) if mm else model.parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=epochs * len(tr_dl), pct_start=0.1)

    model, history = fit(model, tr_dl, va_dl, prep, device, epochs=epochs, optimizer=opt, scheduler=sched,
                         patience=6 if mm else 40, log=lambda s: print(f"[{run}] {s}", flush=True))

    results = {}
    for name, ds, dl in [("val", va_ds, va_dl), ("test", te_ds, te_dl)]:
        pred = prep.inverse_y(predict(model, dl, device))
        results[name] = regression_metrics(ds.df[C.TARGET], pred)
        ds.df.drop(columns=["image_path"]).assign(pred_kt=pred).to_csv(out / f"predictions_{name}.csv", index=False)
    print(f"[{run}] val  {results['val']}\n[{run}] test {results['test']}")

    torch.save({"model": args.model, "backbone": args.backbone, "features": features,
                "feature_set": args.features, "split": args.split, "state_dict": model.state_dict(),
                "prep": prep, "args": vars(args)}, out / "model.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
