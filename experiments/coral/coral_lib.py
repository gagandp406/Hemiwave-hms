"""CoRaL — Committee-of-Latent-Raters Learning for HMS / IIIC vote distributions.

Core library shared by the screening scripts:
  * CommitteeHead: shared "perception" logits z(x); R latent raters, each a shifted and
    tempered softmax of z (the per-class logit-shift calibration model of Jing et al. 2023,
    Neurology, used here as a learnable layer); two rater pools (low-vote pool = 119 mixed
    raters, expert pool = 20 fellowship-trained experts) with their own mixing weights over
    the latent raters. Output = expected vote share of the pool's committee.
  * Likelihoods on raw vote counts: KL (baseline), multinomial NLL, Dirichlet-multinomial
    with learned per-pool concentration.
  * kNN vote-memory retrieval (committee evidence from labelled neighbours).
  * Metrics: Kaggle-style KLD, patient-level paired bootstrap.

No code is copied from public notebooks. Python >= 3.9, torch >= 2.0, numpy, pandas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTE_COLS = [f"{c}_vote" for c in CLASSES]
HQ_MIN_VOTES = 10          # HMS: vote counts are 1-7 or 10-28; none are 8 or 9
POOL_LOW, POOL_EXPERT = 0, 1


# ----------------------------------------------------------------------------- metrics
def kld_rows(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Per-row KL(y || p) with 0*log0 = 0 and predictions clipped like the Kaggle metric."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    p = p / p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * (np.log(np.where(y > 0, y, 1.0)) - np.log(p)), 0.0)
    return t.sum(1)


def votes_to_share(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / v.sum(1, keepdims=True)


def paired_patient_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, patient: np.ndarray,
                             n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean(loss_b - loss_a) with a 95% CI from resampling patients (pooled over rows).

    Negative delta = B is better than A.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patient, return_inverse=True)
    d = loss_b - loss_a
    sum_d = np.bincount(inv, weights=d, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(uniq), len(uniq))
        draws[i] = sum_d[idx].sum() / cnt[idx].sum()
    return {
        "delta": float(d.mean()),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "p_better": float((draws < 0).mean()),
    }


# ----------------------------------------------------------------------------- model
class CommitteeHead(nn.Module):
    """mu_pool(x) = sum_r pi_pool,r(x) * softmax(tau_r * (z(x) + b_pool + theta_r)).

    n_raters=1, n_pools=1  -> plain softmax head (baseline)
    n_raters=1, n_pools=2  -> pool-conditional softmax (pool-specific class bias)
    n_raters>1             -> latent-rater committee (mixture of shifted/tempered softmaxes)
    gate=True              -> sample-specific mixing weights (Guan et al. 2018 style, latent raters)
    """

    def __init__(self, d_in: int, n_cls: int = 6, n_raters: int = 1, n_pools: int = 2,
                 gate: bool = False, hidden: int = 0, dropout: float = 0.0,
                 init_W: Optional[np.ndarray] = None, init_b: Optional[np.ndarray] = None,
                 theta_init_std: float = 0.5, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_raters, self.n_pools = n_raters, n_pools
        self.drop = nn.Dropout(dropout)
        if hidden > 0:
            self.base = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_cls))
        else:
            self.base = nn.Linear(d_in, n_cls)
            if init_W is not None:
                with torch.no_grad():
                    self.base.weight.copy_(torch.as_tensor(np.asarray(init_W).T, dtype=torch.float32))
                    self.base.bias.copy_(torch.as_tensor(np.asarray(init_b), dtype=torch.float32))
        th = torch.randn(n_raters, n_cls, generator=g) * (theta_init_std if n_raters > 1 else 0.0)
        th = th - th.mean(1, keepdim=True)
        self.theta = nn.Parameter(th)
        self.log_tau = nn.Parameter(torch.zeros(n_raters))
        self.pool_bias = nn.Parameter(torch.zeros(n_pools, n_cls))
        self.pool_logits = nn.Parameter(torch.zeros(n_pools, n_raters))
        self.gate = nn.Linear(d_in, n_raters) if (gate and n_raters > 1) else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.log_kappa = nn.Parameter(torch.full((n_pools,), math.log(30.0)))

    def forward(self, h: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        """Returns log mu (B, C)."""
        pool = pool.clamp(max=self.n_pools - 1)
        z = self.base(self.drop(h)) + self.pool_bias[pool]
        zr = (z[:, None, :] + self.theta[None]) * self.log_tau.exp()[None, :, None]
        logp_r = F.log_softmax(zr, dim=-1)                               # (B, R, C)
        a = self.pool_logits[pool]
        if self.gate is not None:
            a = a + self.gate(h)
        logpi = F.log_softmax(a, dim=-1)                                 # (B, R)
        return torch.logsumexp(logpi[..., None] + logp_r, dim=1)         # (B, C)

    def kappa(self, pool: torch.Tensor) -> torch.Tensor:
        return self.log_kappa.exp()[pool.clamp(max=self.n_pools - 1)]


# ----------------------------------------------------------------------------- losses
def loss_kl(v: torch.Tensor, logmu: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = v.sum(1, keepdim=True)
    y = v / n
    t = torch.where(y > 0, y * (torch.log(y.clamp_min(1e-12)) - logmu), torch.zeros_like(y))
    return (t.sum(1) * w).sum() / w.sum().clamp_min(1e-12)


def loss_multinomial(v: torch.Tensor, logmu: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # -log Mult(v | n, mu) up to a constant; normalised per vote so the scale ~ cross-entropy
    nll = -(v * logmu).sum(1)
    return (nll * w).sum() / (v.sum(1) * w).sum().clamp_min(1e-12)


def loss_dirmult(v: torch.Tensor, logmu: torch.Tensor, kappa: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # -log DirMult(v | kappa*mu) up to a constant (multinomial coefficient dropped)
    mu = logmu.exp().clamp_min(1e-6)
    alpha = kappa[:, None] * mu
    n = v.sum(1)
    ll = (torch.lgamma(kappa) - torch.lgamma(n + kappa)
          + (torch.lgamma(v + alpha) - torch.lgamma(alpha)).sum(1))
    return -(ll * w).sum() / (n * w).sum().clamp_min(1e-12)


# ----------------------------------------------------------------------------- training
@dataclass
class Stage:
    mask: str = "all"            # "all" | "hq" | "ge2"
    loss: str = "multinomial"    # "kl" | "multinomial" | "dirmult"
    epochs: int = 20
    lr: float = 3e-3
    weight: str = "one"          # "one" | "n_over_20"  (row weights inside the loss)


@dataclass
class HeadConfig:
    name: str
    n_raters: int = 1
    n_pools: int = 2
    gate: bool = False
    hidden: int = 0
    dropout: float = 0.0
    anchor_l2: float = 1e-3      # L2 pull of the base layer toward its initial (VIPEEGNet) weights
    wd: float = 1e-4
    batch: int = 512
    stages: List[Stage] = field(default_factory=lambda: [Stage()])
    select_last_stage: bool = True   # choose last-stage epoch count on an inner patient split
    seed: int = 0


def _row_mask(kind: str, n: np.ndarray) -> np.ndarray:
    if kind == "all":
        return np.ones_like(n, dtype=bool)
    if kind == "hq":
        return n >= HQ_MIN_VOTES
    if kind == "ge2":
        return n >= 2
    raise ValueError(kind)


def _pools(n: np.ndarray) -> np.ndarray:
    return np.where(n >= HQ_MIN_VOTES, POOL_EXPERT, POOL_LOW).astype(np.int64)


def _make_head(cfg: HeadConfig, d_in: int, init_W, init_b) -> CommitteeHead:
    torch.manual_seed(cfg.seed)
    return CommitteeHead(d_in, n_raters=cfg.n_raters, n_pools=cfg.n_pools, gate=cfg.gate,
                         hidden=cfg.hidden, dropout=cfg.dropout, init_W=init_W, init_b=init_b,
                         seed=cfg.seed)


@torch.no_grad()
def predict(head: CommitteeHead, X: np.ndarray, pool: int = POOL_EXPERT, device="cpu",
            bs: int = 4096) -> np.ndarray:
    head.eval()
    out = []
    for i in range(0, len(X), bs):
        h = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        pl = torch.full((len(h),), pool, dtype=torch.long, device=device)
        out.append(head(h, pl).exp().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _run_stages(head: CommitteeHead, cfg: HeadConfig, X, V, device, eval_fn=None,
                last_stage_epochs: Optional[int] = None):
    """Train through cfg.stages. If eval_fn is given, returns per-epoch scores of the last stage."""
    n = V.sum(1)
    pools = _pools(n)
    W0 = None
    if isinstance(head.base, nn.Linear):
        W0 = (head.base.weight.detach().clone(), head.base.bias.detach().clone())
    scores = []
    rng = np.random.default_rng(cfg.seed)
    for si, st in enumerate(cfg.stages):
        is_last = si == len(cfg.stages) - 1
        epochs = last_stage_epochs if (is_last and last_stage_epochs is not None) else st.epochs
        idx_all = np.where(_row_mask(st.mask, n))[0]
        opt = torch.optim.AdamW(head.parameters(), lr=st.lr, weight_decay=cfg.wd)
        steps = max(1, math.ceil(len(idx_all) / cfg.batch)) * max(epochs, 1)
        warm = max(1, int(0.1 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s, w=warm, T=steps: min(1.0, (s + 1) / w) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(s, T) / T))))
        for ep in range(epochs):
            head.train()
            perm = rng.permutation(idx_all)
            for i in range(0, len(perm), cfg.batch):
                b = perm[i:i + cfg.batch]
                h = torch.as_tensor(X[b], dtype=torch.float32, device=device)
                v = torch.as_tensor(V[b], dtype=torch.float32, device=device)
                pl = torch.as_tensor(pools[b], device=device)
                w = torch.ones(len(b), device=device)
                if st.weight == "n_over_20":
                    w = v.sum(1) / 20.0
                logmu = head(h, pl)
                if st.loss == "kl":
                    loss = loss_kl(v, logmu, w)
                elif st.loss == "multinomial":
                    loss = loss_multinomial(v, logmu, w)
                elif st.loss == "dirmult":
                    loss = loss_dirmult(v, logmu, head.kappa(pl), w)
                else:
                    raise ValueError(st.loss)
                if W0 is not None and cfg.anchor_l2 > 0:
                    loss = loss + cfg.anchor_l2 * (((head.base.weight - W0[0]) ** 2).sum()
                                                   + ((head.base.bias - W0[1]) ** 2).sum())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
            if is_last and eval_fn is not None:
                scores.append(eval_fn(head))
    return scores


def fit_head(cfg: HeadConfig, X: np.ndarray, V: np.ndarray, patient: np.ndarray,
             init_W=None, init_b=None, device="cpu", inner_frac: float = 0.15):
    """Fit on training rows only. The last stage's epoch count is chosen on an inner
    patient-level split of the training rows (HQ KLD), then the head is refit on all rows.
    The outer validation fold is never used for any choice."""
    best_ep = None
    if cfg.select_last_stage and cfg.stages[-1].epochs > 1:
        rng = np.random.default_rng(cfg.seed + 17)
        up = np.unique(patient)
        inner_p = set(rng.choice(up, max(1, int(len(up) * inner_frac)), replace=False).tolist())
        is_in = np.array([p in inner_p for p in patient])
        tr, iv = ~is_in, is_in & (V.sum(1) >= HQ_MIN_VOTES)
        if iv.sum() >= 50:
            head = _make_head(cfg, X.shape[1], init_W, init_b).to(device)
            Xi, Yi = X[iv], votes_to_share(V[iv])
            scores = _run_stages(head, cfg, X[tr], V[tr], device,
                                 eval_fn=lambda hd: float(kld_rows(Yi, predict(hd, Xi, device=device)).mean()))
            best_ep = int(np.argmin(scores)) + 1
    head = _make_head(cfg, X.shape[1], init_W, init_b).to(device)
    _run_stages(head, cfg, X, V, device, last_stage_epochs=best_ep)
    return head, best_ep


# ----------------------------------------------------------------------------- retrieval
@torch.no_grad()
def knn_topk(Q: np.ndarray, K: np.ndarray, k: int, device="cpu", bs: int = 2048):
    """Cosine top-k of queries Q against keys K. Returns (sims, idx) with shape (Nq, k)."""
    Kt = F.normalize(torch.as_tensor(K, dtype=torch.float32, device=device), dim=1)
    sims, idxs = [], []
    for i in range(0, len(Q), bs):
        q = F.normalize(torch.as_tensor(Q[i:i + bs], dtype=torch.float32, device=device), dim=1)
        s, j = (q @ Kt.T).topk(k, dim=1)
        sims.append(s.cpu().numpy()); idxs.append(j.cpu().numpy())
    return np.concatenate(sims), np.concatenate(idxs)


def knn_vote_share(sims: np.ndarray, idx: np.ndarray, V_keys: np.ndarray, k: int, tau: float,
                   prior: Optional[np.ndarray] = None, prior_strength: float = 0.0) -> np.ndarray:
    """Committee evidence from neighbours: similarity-weighted pooled vote counts.

    Pooling raw counts (not shares) lets a 20-vote neighbour weigh more than a 3-vote one,
    i.e. the neighbours act as extra raters. With a prior (the model's own mu) this is the
    Dirichlet posterior mean (sum_i w_i v_i + a*mu) / (sum_i w_i n_i + a).
    """
    s, j = sims[:, :k], idx[:, :k]
    w = np.exp((s - s[:, :1]) / tau)                       # (Nq, k), stable softmax numerator
    w = w / w.sum(1, keepdims=True)
    Vn = V_keys[j]                                         # (Nq, k, C)
    num = (w[..., None] * Vn).sum(1)
    den = (w * Vn.sum(2)).sum(1, keepdims=True)
    if prior is not None and prior_strength > 0:
        num = num + prior_strength * prior
        den = den + prior_strength
    return num / den


def blend(p_model: np.ndarray, p_knn: np.ndarray, lam: float, mode: str = "arith") -> np.ndarray:
    if mode == "arith":
        p = (1 - lam) * p_model + lam * p_knn
    else:
        lp = (1 - lam) * np.log(np.clip(p_model, 1e-12, 1)) + lam * np.log(np.clip(p_knn, 1e-12, 1))
        p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- configs
def standard_configs(lr: float = 3e-3, ep1: int = 15, ep2: int = 30) -> List[HeadConfig]:
    """The screening ladder. Each rung changes one thing relative to the one above it."""
    return [
        # A: VIPEEGNet-style label handling on a fresh head (two-stage KL, n/20 weights, then >=10 votes)
        HeadConfig("A_softmax_kl_2stage", n_raters=1, n_pools=1,
                   stages=[Stage("all", "kl", ep1, lr, "n_over_20"), Stage("hq", "kl", ep2, lr / 3)]),
        # B: one stage on all rows, multinomial likelihood (votes as counts), single pool
        HeadConfig("B_softmax_mult_1pool", n_raters=1, n_pools=1,
                   stages=[Stage("all", "multinomial", ep2, lr)]),
        # C: + pool-conditional bias (low-vote raters vs experts)
        HeadConfig("C_pool_softmax_mult", n_raters=1, n_pools=2,
                   stages=[Stage("all", "multinomial", ep2, lr)]),
        # D: + latent-rater committee (mixture of 8 shifted/tempered softmaxes)
        HeadConfig("D_coral_R8_mult", n_raters=8, n_pools=2,
                   stages=[Stage("all", "multinomial", ep2, lr)]),
        # E: + Dirichlet-multinomial likelihood (learned per-pool overdispersion)
        HeadConfig("E_coral_R8_dm", n_raters=8, n_pools=2,
                   stages=[Stage("all", "dirmult", ep2, lr)]),
        # F: + sample-specific committee mixing (gated)
        HeadConfig("F_coral_R8_dm_gate", n_raters=8, n_pools=2, gate=True,
                   stages=[Stage("all", "dirmult", ep2, lr)]),
        # G: E followed by a short expert-only refinement (keeps the committee, tunes on HQ rows)
        HeadConfig("G_coral_R8_dm_then_hq", n_raters=8, n_pools=2,
                   stages=[Stage("all", "dirmult", ep2, lr), Stage("hq", "dirmult", ep2, lr / 3)]),
    ]
