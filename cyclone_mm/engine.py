"""Training / evaluation loops and metrics."""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
            "Bias": float(np.mean(y_pred - y_true)),
            "N": int(len(y_true))}


def _forward(model, batch, device):
    tab = batch["tabular"].to(device)
    img = batch["image"].to(device) if "image" in batch else None
    return model(tab, img)


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None, grad_clip=1.0) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        if len(batch["target"]) < 2:  # BatchNorm needs >1 sample
            continue
        y = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(_forward(model, batch, device), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """Predictions in the *scaled* target space, in dataset order."""
    model.eval()
    out = np.zeros(len(loader.dataset), np.float32)
    for batch in loader:
        out[batch["idx"].numpy()] = _forward(model, batch, device).squeeze(1).cpu().numpy()
    return out


def fit(model, train_loader, val_loader, prep, device, *, epochs, optimizer, scheduler=None,
        loss_fn=None, patience=8, log=print):
    """Train with early stopping on validation RMSE (knots). Returns best model + history."""
    loss_fn = loss_fn or nn.SmoothL1Loss(beta=0.5)
    y_val = val_loader.dataset.df["wind_kt"].to_numpy()
    best, best_rmse, bad, history = None, float("inf"), 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        val_pred = prep.inverse_y(predict(model, val_loader, device))
        m = regression_metrics(y_val, val_pred)
        history.append({"epoch": ep, "train_loss": tr_loss, **{f"val_{k}": v for k, v in m.items()}})
        improved = m["RMSE"] < best_rmse - 1e-3
        if improved:
            best, best_rmse, bad = copy.deepcopy(model.state_dict()), m["RMSE"], 0
        else:
            bad += 1
        log(f"epoch {ep:3d}  train_loss {tr_loss:.4f}  val RMSE {m['RMSE']:.2f}  MAE {m['MAE']:.2f}  "
            f"R2 {m['R2']:.3f}  {time.time() - t0:.0f}s{'  *' if improved else ''}")
        if bad >= patience:
            log(f"early stop (no val improvement for {patience} epochs)")
            break
    model.load_state_dict(best)
    return model, history
