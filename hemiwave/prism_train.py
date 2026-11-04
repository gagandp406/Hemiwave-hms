"""Training / evaluation pipeline for PRISM-EEG on HMS (PyTorch).

Data: reuses VIPEEGNet's EEG cache (18 bipolar channels, clip +-1024 -> 0.5-45 Hz filtfilt, 200 Hz) and VIPEEGNet's
row table (train.csv deduplicated on eeg_id + votes, 20,183 rows) and patient split, so predictions line up row by row
with the archived baseline. Everything after the waveform (representation, network, head, loss) is PRISM's own.
"""
import os, gc, json, time, math, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from prism import PRISM, kl_loss, dm_nll, CLASSES

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
FS, WIN = 200, 10_000


# ------------------------------------------------------------------------------------------------ data
def load_tables(data_dir, split_csv, smoke=False):
    df = pd.read_csv(f'{data_dir}/train.csv')
    rows = df.drop_duplicates(['eeg_id'] + TARGETS).reset_index(drop=True)[['eeg_id', 'patient_id'] + TARGETS].copy()
    rows['votes'] = rows[TARGETS].sum(1)
    pf = pd.read_csv(split_csv)[['patient_id', 'fold']]
    rows = rows.merge(pf, on='patient_id', how='left')
    assert rows['fold'].notna().all(), 'patient missing from split file'
    rows['fold'] = rows['fold'].astype(int)
    if smoke:
        keep = pd.Series(rows.patient_id.unique()).sample(min(250, rows.patient_id.nunique()), random_state=0)
        rows = rows[rows.patient_id.isin(keep)].reset_index(drop=True)
    lookup = {}
    for key, g in df[['eeg_id'] + TARGETS + ['eeg_sub_id', 'eeg_label_offset_seconds']].groupby(['eeg_id'] + TARGETS, sort=False):
        lookup[tuple(int(k) for k in key)] = (g['eeg_sub_id'].to_numpy(), g['eeg_label_offset_seconds'].to_numpy())
    return rows, lookup


def load_eegs(cache_dir, eeg_ids):
    from tqdm.auto import tqdm
    return {int(e): np.load(f'{cache_dir}/{e}.npy', mmap_mode=None) for e in tqdm(eeg_ids, desc='load EEG cache')}


class HMSRows(Dataset):
    def __init__(self, rows, lookup, eegs, mode='train', shift_s=5.0, rev_p=0.5, chdrop_p=0.5, gain_p=0.5):
        self.eid = rows['eeg_id'].to_numpy()
        self.raw = rows[TARGETS].to_numpy().astype(np.int64)
        self.lookup, self.eegs, self.mode = lookup, eegs, mode
        self.shift_s, self.rev_p, self.chdrop_p, self.gain_p = shift_s, rev_p, chdrop_p, gain_p

    def __len__(self):
        return len(self.eid)

    def start(self, i, rng=None):
        eid = int(self.eid[i]); sub, offs = self.lookup[(eid,) + tuple(int(v) for v in self.raw[i])]
        if self.mode == 'train':
            r = offs[rng.integers(len(offs))] + rng.uniform(-self.shift_s, self.shift_s)
            return float(np.clip(r, 0, max(self.eegs[eid].shape[1] / FS - 50, 0)))
        return float(offs[np.argsort(sub, kind='stable')[len(offs) // 2]])   # same window VIPEEGNet validates on

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.mode == 'train' else None
        eid = int(self.eid[i]); r = self.start(i, rng)
        x = self.eegs[eid][:, round(r * FS):round(r * FS) + WIN]
        if x.shape[1] < WIN:
            x = np.pad(x, ((0, 0), (0, WIN - x.shape[1])))
        x = np.clip(x, -1024, 1024).astype(np.float32)
        if self.mode == 'train':
            if rng.random() < self.rev_p:
                x = x[:, ::-1]
            if rng.random() < 0.5:                               # polarity flip (RPS maps are sign-invariant anyway)
                x = -x
            if rng.random() < self.gain_p:                       # per-channel gain jitter (+-20%)
                x = x * rng.uniform(0.8, 1.25, size=(18, 1)).astype(np.float32)
            if rng.random() < self.chdrop_p:
                x = x.copy(); x[rng.choice(18, size=rng.integers(1, 3), replace=False)] = 0
        v = self.raw[i].astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(v / v.sum()), torch.from_numpy(v)


# ------------------------------------------------------------------------------------------------ training
class EMA:
    def __init__(self, model, decay=0.998):
        self.decay, self.n = decay, 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))   # warm-up: early steps are not dominated by the random init
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.float(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def load_into(self, model):
        model.load_state_dict({k: v.to(dtype=model.state_dict()[k].dtype) for k, v in self.shadow.items()})


def make_opt(model, lr_backbone, lr_head, wd, steps, warmup):
    bb = [p for n, p in model.named_parameters() if n.startswith('enc.')]
    hd = [p for n, p in model.named_parameters() if not n.startswith('enc.')]
    opt = torch.optim.AdamW([{'params': bb, 'lr': lr_backbone}, {'params': hd, 'lr': lr_head}], weight_decay=wd)
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval(); out = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            lp = model(x).float()
            if tta:
                lp = (lp + model(x.flip(-1)).float()) / 2
                lp = lp - torch.logsumexp(lp, 1, keepdim=True)
        out.append(lp.exp().cpu().numpy())
    return np.concatenate(out)


def kld_rows(t, p, eps=1e-7):
    t = np.clip(t, eps, 1); p = np.clip(p, eps, 1)
    return np.sum(t * np.log(t / p), 1)


def run_fold(k, rows, lookup, eegs, cfg, out_dir, w_dir, device='cuda', log=print):
    torch.manual_seed(cfg['seed'] + k); np.random.seed(cfg['seed'] + k); random.seed(cfg['seed'] + k)
    tr, va = rows[rows.fold != k].reset_index(drop=True), rows[rows.fold == k].reset_index(drop=True)
    model = PRISM(backbone=cfg['backbone'], pretrained=cfg['pretrained'], head=cfg['head'], fusion=cfg['fusion'],
                  repr=cfg['repr'], norm=cfg.get('norm', 'global'), specaug=cfg.get('specaug', True),
                  drop=cfg.get('drop', 0.2), drop_path=cfg.get('drop_path', 0.1),
                  wave_width=cfg.get('wave_width', 1.0)).to(device)
    log_kappa = torch.nn.Parameter(torch.tensor(math.log(10.0), device=device))
    ema = EMA(model, cfg['ema'])
    dl_kw = dict(num_workers=cfg['workers'], pin_memory=device == 'cuda', persistent_workers=cfg['workers'] > 0)
    va_hq = va[va.votes >= 10].reset_index(drop=True)
    va_loader = DataLoader(HMSRows(va_hq, lookup, eegs, 'valid'), batch_size=cfg['batch'] * 2, **dl_kw)
    history = []
    stages = [('stage1', tr, cfg['epochs1'], cfg['lr1'], True), ('stage2', tr[tr.votes >= 10].reset_index(drop=True),
                                                                   cfg['epochs2'], cfg['lr2'], False)]
    for name, df, epochs, lr, weighted in stages:
        if epochs <= 0:
            continue
        loader = DataLoader(HMSRows(df, lookup, eegs, 'train'), batch_size=cfg['batch'], shuffle=True, drop_last=True,
                            **dl_kw)
        steps = epochs * len(loader)
        opt, sch = make_opt(model, lr, lr * cfg['head_lr_mult'], cfg['wd'], steps, len(loader) if name == 'stage1' else 0)
        if name == 'stage1' and cfg['loss'] == 'kl+dm':
            opt.add_param_group({'params': [log_kappa], 'lr': 1e-2, 'weight_decay': 0})
        for ep in range(epochs):
            model.train(); t0 = time.time(); tot = 0.0; nb = 0
            for x, t, v in loader:
                x, t, v = x.to(device, non_blocking=True), t.to(device), v.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                    lp = model(x)
                w = v.sum(1) / 20 if weighted else None
                loss = kl_loss(lp.float(), t, w)
                if name == 'stage1' and cfg['loss'] == 'kl+dm':
                    loss = loss + cfg['dm_weight'] * dm_nll(lp.float(), v, log_kappa)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
                opt.step(); sch.step(); ema.update(model)
                tot += loss.item(); nb += 1
            # monitoring only (EMA weights); no checkpoint is selected on validation
            cur = {kk: vv.clone() for kk, vv in model.state_dict().items()}
            ema.load_into(model); p = predict(model, va_loader, device); model.load_state_dict(cur)
            vk = float(kld_rows(va_hq[TARGETS].values / va_hq.votes.values[:, None], p).mean())
            history.append({'stage': name, 'epoch': ep + 1, 'train_loss': tot / max(nb, 1), 'val_hq_kld_ema': vk,
                            'seconds': time.time() - t0})
            log(f'  fold {k} {name} ep {ep + 1}/{epochs}  train {tot / max(nb, 1):.4f}  val>=10 (EMA) {vk:.4f}'
                f'  {time.time() - t0:.0f}s' + (f'  kappa {log_kappa.exp().item():.1f}' if cfg['loss'] == 'kl+dm' else ''))
    ema.load_into(model)
    torch.save(model.state_dict(), f'{w_dir}/fold{k}_ema.pt')
    # out-of-fold predictions for ALL validation rows, same order as the VIPEEGNet archive
    all_loader = DataLoader(HMSRows(va, lookup, eegs, 'valid'), batch_size=cfg['batch'] * 2, **dl_kw)
    res = {}
    for tag, tta in (('', False), ('_tta', True)):
        p = predict(model, all_loader, device, tta=tta)
        yt = (va[TARGETS].values / va.votes.values[:, None]).astype(np.float32)
        d = f'{out_dir}{tag}'; os.makedirs(d, exist_ok=True)
        hq = va.votes.values >= 10
        for fn, m in ((f'fold{k}_oof_all.npz', np.ones(len(va), bool)), (f'fold{k}_oof.npz', hq)):
            np.savez(f'{d}/{fn}', yt=yt[m], yp=p[m].astype(np.float32), wt=va.votes.values[m].astype(np.float32),
                     yp_raw_logits=np.log(np.clip(p[m], 1e-8, 1)).astype(np.float32), eeg_ids=va.eeg_id.values[m],
                     patient_ids=va.patient_id.values[m], temperature=np.array([1.0], np.float32))
        res[f'kld_hq{tag}'] = float(kld_rows(yt[hq], p[hq]).mean())
        res[f'kld_all{tag}'] = float(kld_rows(yt, p).mean())
    res.update(fold=k, n_valid_hq=int(hq.sum()), n_valid_all=len(va), history=history,
               kappa=float(log_kappa.exp().item()))
    del model, ema; gc.collect(); torch.cuda.empty_cache() if device == 'cuda' else None
    return res
