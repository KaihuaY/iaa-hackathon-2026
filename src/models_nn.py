"""CANN-style neural residual learner: GLM first, then an MLP learns a bounded log-link residual on top of it."""
import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from models import GLMTweedie

DEFAULT = dict(hidden=128, dropout=0.1, lr=1e-3, batch=4096, epochs=30, patience=3, weight_decay=1e-5,
               tweedie_variance_power=1.5, seed=7, threads=8)
CLIP_X, CLIP_R = 5.0, 3.0


def _cat_cols(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


class _Net(nn.Module):
    def __init__(self, n_num, card, hidden, dropout):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n + 1, min(16, math.ceil(n ** 0.5) + 1)) for n in card])
        d = n_num + sum(e.embedding_dim for e in self.embs)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, xn, xc):
        parts = [e(xc[:, i]) for i, e in enumerate(self.embs)] + [xn]
        return torch.clamp(self.mlp(torch.cat(parts, 1)).squeeze(1), -CLIP_R, CLIP_R)


def _deviance(y, mu, p):
    return 2 * (y.pow(2 - p) / ((1 - p) * (2 - p)) - y * mu.pow(1 - p) / (1 - p) + mu.pow(2 - p) / (2 - p))


class NNBoost:
    def __init__(self, params=None):
        params = dict(params or {})
        self.glm = GLMTweedie(**params.pop("glm", {}))
        self.params = {**DEFAULT, **params}
        self.best_iter = None
        self.scale = 1.0

    # ---- preprocessing -------------------------------------------------
    def _fit_prep(self, X):
        self.cats = _cat_cols(X)
        self.nums = [c for c in X.columns if c not in self.cats]
        A = X[self.nums].to_numpy(np.float32)
        self.med = np.nan_to_num(np.nanmedian(A, axis=0), nan=0.0).astype(np.float32)
        self.miss = [i for i in range(A.shape[1]) if np.isnan(A[:, i]).any()]
        A = np.where(np.isnan(A), self.med, A)
        self.mu_, self.sd_ = A.mean(0), A.std(0)
        self.sd_ = np.where(self.sd_ < 1e-8, 1.0, self.sd_).astype(np.float32)
        self.levels = {c: {v: i + 1 for i, v in enumerate(X[c].cat.categories)} for c in self.cats}

    def _prep(self, X):
        A = X[self.nums].to_numpy(np.float32)
        nan = np.isnan(A)
        ind = nan[:, self.miss].astype(np.float32)
        A = np.where(nan, self.med, A)
        A = np.clip((A - self.mu_) / self.sd_, -CLIP_X, CLIP_X).astype(np.float32)
        xn = torch.from_numpy(np.ascontiguousarray(np.hstack([A, ind])))
        cols = []
        for c in self.cats:
            codes = pd.Categorical(X[c], categories=list(self.levels[c])).codes.astype(np.int64) + 1
            cols.append(codes)       # NaN / unseen -> -1 + 1 = 0
        xc = torch.from_numpy(np.stack(cols, 1)) if cols else torch.zeros((len(X), 0), dtype=torch.int64)
        return xn, xc

    def _resid(self, xn, xc, bs=65536):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xn), bs):
                out.append(self.net(xn[i:i + bs], xc[i:i + bs]))
        return torch.cat(out).numpy().astype(np.float64)

    def _val_dev(self, xn, xc, y, w, m, bs=65536):
        p, tot = self.params["tweedie_variance_power"], 0.0
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(xn), bs):
                mu = torch.exp(m[i:i + bs] + self.net(xn[i:i + bs], xc[i:i + bs]))
                tot += float((w[i:i + bs] * _deviance(y[i:i + bs], mu, p)).sum())
        return tot / float(w.sum())

    # ---- fit / predict ---------------------------------------------------
    def fit(self, X, y, w, expo, X_val=None, y_val=None, w_val=None, expo_val=None, n_iter=None):
        P = self.params
        torch.set_num_threads(int(P["threads"]))
        torch.manual_seed(int(P["seed"]))
        np.random.seed(int(P["seed"]))
        self.glm.fit(X, y, w, expo)
        self._fit_prep(X)
        xn, xc = self._prep(X)
        yt, wt = torch.tensor(np.asarray(y), dtype=torch.float32), torch.tensor(np.asarray(w), dtype=torch.float32)
        mt = torch.tensor(np.log(self.glm.predict(X, expo)), dtype=torch.float32)
        card = [len(self.levels[c]) for c in self.cats]
        self.net = _Net(xn.shape[1], card, int(P["hidden"]), float(P["dropout"]))
        use_val = X_val is not None and n_iter is None
        if use_val:
            vn, vc = self._prep(X_val)
            vy, vw = torch.tensor(np.asarray(y_val), dtype=torch.float32), torch.tensor(np.asarray(w_val), dtype=torch.float32)
            vm = torch.tensor(np.log(self.glm.predict(X_val, expo_val)), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(P["lr"]), weight_decay=float(P["weight_decay"]))
        n, bs, pw = len(xn), int(P["batch"]), P["tweedie_variance_power"]
        n_ep = int(n_iter) if n_iter is not None else int(P["epochs"])
        best, best_ep, best_state, bad = np.inf, 0, None, 0
        for ep in range(1, n_ep + 1):
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                mu = torch.exp(mt[idx] + self.net(xn[idx], xc[idx]))
                dev = _deviance(yt[idx], mu, pw)
                loss = (wt[idx] * dev).sum() / wt[idx].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float((wt[idx] * dev.detach()).sum())
            msg = f"epoch {ep} train dev {tot / float(wt.sum()):.5f}"
            if use_val:
                vd = self._val_dev(vn, vc, vy, vw, vm)
                msg += f" val dev {vd:.5f}"
                if vd < best:
                    best, best_ep, bad = vd, ep, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
            print(msg, flush=True)
            if use_val and bad >= int(P["patience"]):
                break
        if use_val:
            self.net.load_state_dict(best_state)
            self.best_iter = best_ep
        else:
            self.best_iter = n_ep
        self.scale = 1.0
        self.scale = float(np.sum(w * y) / np.sum(w * self.predict(X, expo)))
        return self

    def predict(self, X, expo):
        xn, xc = self._prep(X)
        return self.scale * np.exp(np.log(self.glm.predict(X, expo)) + self._resid(xn, xc))

    def importance(self):
        return pd.Series(dtype=float)
