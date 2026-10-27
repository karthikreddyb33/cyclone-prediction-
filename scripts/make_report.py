"""Build a self-contained training report from outputs/ (run after scripts/run_training.sh).

  python scripts/make_report.py   ->  outputs/training_report/{report.html, *.png}
"""
import base64
import html
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclone_mm import config as C  # noqa: E402

LOGS = C.OUTPUTS / "training_logs"
OUT = C.OUTPUTS / "training_report"
RUNS = {  # run dir -> (label, log file)
    "tabular_physical": ("Tabular MLP [physical]", "01_tabular_physical.log"),
    "tabular_no_pressure": ("Tabular MLP [no_pressure]", "02_tabular_no_pressure.log"),
    "multimodal_physical_efficientnet_b0": ("Multimodal EfficientNet-B0 [physical]", "03_multimodal_physical.log"),
    "multimodal_no_pressure_efficientnet_b0": ("Multimodal EfficientNet-B0 [no_pressure]", "04_multimodal_no_pressure.log"),
}


def png_tag(path: Path) -> str:
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{path.stem}">'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {r: v for r, v in RUNS.items() if (C.OUTPUTS / r / "metrics.json").exists()}

    # Training curves: train loss + validation RMSE per epoch, one row per run.
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 3.0 * len(runs)), squeeze=False)
    for (run, (label, _)), (a1, a2) in zip(runs.items(), axes):
        h = pd.read_csv(C.OUTPUTS / run / "history.csv")
        best = h.loc[h.val_RMSE.idxmin()]
        a1.plot(h.epoch, h.train_loss, color="#3b6fb6")
        a1.set(title=f"{label} - train loss (SmoothL1, scaled)", xlabel="epoch")
        a2.plot(h.epoch, h.val_RMSE, color="#c4572a")
        a2.axvline(best.epoch, ls="--", c="grey", lw=1)
        a2.set(title=f"validation RMSE (kt) - best {best.val_RMSE:.2f} @ epoch {int(best.epoch)}", xlabel="epoch")
        for a in (a1, a2):
            a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=110)
    plt.close(fig)

    # Actual vs predicted on the 2025 test set.
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), squeeze=False)
    for (run, (label, _)), ax in zip(runs.items(), axes[0]):
        p = pd.read_csv(C.OUTPUTS / run / "predictions_test.csv")
        lim = [0, max(p[C.TARGET].max(), p.pred_kt.max()) + 5]
        ax.scatter(p[C.TARGET], p.pred_kt, s=10, alpha=0.6)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlim=lim, ylim=lim, xlabel="actual wind (kt)", ylabel="predicted (kt)", title=label)
        ax.title.set_fontsize(9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "test_2025_scatter.png", dpi=110)
    plt.close(fig)

    rows = []
    for run, (label, _) in runs.items():
        m = json.loads((C.OUTPUTS / run / "metrics.json").read_text())
        h = pd.read_csv(C.OUTPUTS / run / "history.csv")
        rows.append({"Model": label, "Epochs run": len(h),
                     **{f"Val {k}": m["val"][k] for k in ("RMSE", "MAE", "R2")},
                     **{f"Test 2025 {k}": m["test"][k] for k in ("RMSE", "MAE", "R2", "Bias")}})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)

    env = (LOGS / "00_environment.txt").read_text()
    sections = []
    for run, (label, log) in runs.items():
        text = (LOGS / log).read_text(errors="replace")
        sections.append(f"<details><summary>{html.escape(label)} — full log ({log})</summary>"
                        f"<pre>{html.escape(text)}</pre></details>")
    extra = ""
    if (LOGS / "05_evaluate.log").exists():
        extra += f"<h2>2025 evaluation (evaluate.py)</h2><pre>{html.escape((LOGS / '05_evaluate.log').read_text())}</pre>"
    storms = C.OUTPUTS / "comparison_2025" / "storms_2025.png"
    if storms.exists():
        extra += f"<h2>2025 storms: actual vs predicted</h2>{png_tag(storms)}"

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cyclone Training Report</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}pre{{background:#f5f5f5;padding:10px;overflow-x:auto;font-size:12px}}
img{{max-width:100%}}h1{{margin-bottom:0}}
</style></head><body>
<h1>Multimodal cyclone intensity estimation — training report</h1>
<p>North Indian Ocean, IMD best track + GridSat-B1 IR + OISST. Train/val = 2013-2024 (split by storm), test = 2025 season.</p>
<h2>Environment and timing</h2><pre>{html.escape(env)}</pre>
<h2>Results</h2>{table.to_html(index=False, float_format=lambda v: f"{v:.3f}")}
<p><b>physical</b> includes central pressure, which IMD derives from wind, so it nearly gives away the target;
<b>no_pressure</b> is the honest test of what the satellite image adds.</p>
<h2>Training curves</h2>{png_tag(OUT / "training_curves.png")}
<h2>2025 test: actual vs predicted</h2>{png_tag(OUT / "test_2025_scatter.png")}
{extra}
<h2>Training logs (every epoch)</h2>{''.join(sections)}
</body></html>"""
    (OUT / "report.html").write_text(page, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"wrote {OUT / 'report.html'}")


if __name__ == "__main__":
    main()
