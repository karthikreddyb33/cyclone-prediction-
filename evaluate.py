"""Score trained runs on the 2025 test set and write a comparison table.

  python evaluate.py                       # every outputs/*/model.pt
  python evaluate.py --runs tabular_physical multimodal_physical_efficientnet_b0
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cyclone_mm import config as C  # noqa: E402
from cyclone_mm.data import CycloneDataset, load_samples, make_loader  # noqa: E402
from cyclone_mm.engine import predict, regression_metrics  # noqa: E402
from train import build_model  # noqa: E402


def load_run(run_dir, device):
    ck = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    model = build_model(ck["model"], len(ck["features"]), ck["backbone"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def label(ck) -> str:
    kind = "Tabular MLP" if ck["model"] == "tabular" else f"Multimodal ({ck['backbone']})"
    return f"{kind} [{ck['feature_set']}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [C.OUTPUTS / r for r in args.runs] if args.runs else sorted(p.parent for p in C.OUTPUTS.glob("*/model.pt"))

    test = load_samples(require_images=True)
    test = test[test["year"] == C.TEST_YEAR].reset_index(drop=True)
    y = test[C.TARGET].to_numpy()

    rows, preds = [], {}
    for rd in runs:
        model, ck = load_run(rd, device)
        mm = ck["model"] == "multimodal"
        ds = CycloneDataset(test, ck["prep"], with_images=mm)
        p = ck["prep"].inverse_y(predict(model, make_loader(ds, 64, False, workers=4 if mm else 0), device))
        name = label(ck)
        preds[name] = p
        rows.append({"Model": name, "Run": rd.name, **regression_metrics(y, p)})

    table = pd.DataFrame(rows).sort_values("RMSE")
    out = C.OUTPUTS / "comparison_2025"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "comparison_2025.csv", index=False)
    md = table.drop(columns=["Run"]).to_markdown(index=False, floatfmt=".3f")
    (out / "comparison_2025.md").write_text(f"# 2025 test set ({len(test)} fixes, "
                                            f"{test.storm_id.nunique()} systems)\n\n{md}\n")
    print(f"\n2025 TEST SET  -  {len(test)} fixes from {test.storm_id.nunique()} systems "
          f"(actual wind {y.min():.0f}-{y.max():.0f} kt)\n")
    print(md)

    # Per-storm MAE and actual-vs-predicted time series.
    detail = test[["storm_id", "name", "datetime", "lat", "lon", C.TARGET]].copy()
    for k, p in preds.items():
        detail[k] = np.round(p, 1)
    detail.to_csv(out / "predictions_2025.csv", index=False)
    per_storm = detail.groupby(["storm_id", "name"]).apply(
        lambda g: pd.Series({"fixes": len(g), "peak_kt": g[C.TARGET].max(),
                             **{f"MAE {k}": np.abs(g[k] - g[C.TARGET]).mean() for k in preds}}),
        include_groups=False)
    per_storm.to_csv(out / "per_storm_2025.csv")
    print("\nPer-system MAE (kt):\n" + per_storm.round(1).to_string())

    named = [s for s, n in detail.groupby("storm_id")["name"].first().items() if n != "UNNAMED"]
    fig, axes = plt.subplots(len(named), 1, figsize=(10, 3.2 * len(named)), squeeze=False)
    for ax, sid in zip(axes[:, 0], named):
        g = detail[detail.storm_id == sid]
        ax.plot(g.datetime, g[C.TARGET], "k-o", ms=3, lw=2, label="IMD best track (actual)")
        for k in preds:
            ax.plot(g.datetime, g[k], "-", lw=1.4, label=k)
        ax.set_title(f"{g.name.iloc[0]} ({sid})")
        ax.set_ylabel("Max sustained wind (kt)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "storms_2025.png", dpi=120)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
