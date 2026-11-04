"""PRISM-EEG — Periodicity-Rhythm Images with Symmetric hemispheric Modelling.

A new model for the HMS IIIC task, built from the clinical (ACNS) definitions of the six classes rather than
from generic spectrogram/raw-EEG images:

  * LPD / GPD = Periodic Discharges  -> waveforms that REPEAT at a near-regular interval
  * LRDA/GRDA = Rhythmic Delta       -> near-sinusoidal 0.5-4 Hz activity
  * L vs G    = Lateralized vs Generalized -> a question of LEFT-RIGHT ASYMMETRY
  * Seizure   = rhythmic/periodic activity that EVOLVES over >= 10 s

1. RPS image (per bipolar channel, on GPU, 4 s frames every 0.25 s):
     R = autocorrelation of the signal          (lag 0-2 s)  -> rhythm: ridges at the oscillation period
     P = autocorrelation of the Hilbert envelope (lag 0-4 s)  -> periodicity: ridges at the discharge interval
     S = log power spectrum 0.5-20 Hz                        -> frequency content
   Autocorrelations are amplitude- and polarity-invariant by construction; S is normalised once per sample
   (norm='global', v2) so relative power between channels/hemispheres is kept.
2. Symmetric hemispheric encoder: left chains and mirror-ordered right chains go through ONE shared pretrained
   CNN; fusion uses (L+R)/2 and |L-R| per time token, plus a midline stream. Swapping hemispheres leaves the
   output exactly unchanged (laterality side is irrelevant to the six classes).
3. Evolution transformer over the 8 time tokens (+ first differences) for seizure-type evolution.
4. ACNS-factorised head: P(pattern in {SZ, PD, RDA, Other}) x P(lateralized | PD), P(lateralized | RDA).
   Same 5 degrees of freedom as a 6-way softmax, but laterality is predicted from asymmetry features.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FS = 200
CLASSES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
# channel order of the VIPEEGNet 18-channel cache (eeg_cache_v2):
#  0-3  LL: Fp1-F7 F7-T3 T3-T5 T5-O1     4-7  LP: Fp1-F3 F3-C3 C3-P3 P3-O1
#  8-9  midline: Fz-Cz Cz-Pz             10-13 RP: Fp2-F4 F4-C4 C4-P4 P4-O2
#  14-17 RL: Fp2-F8 F8-T4 T4-T6 T6-O2
LEFT = [0, 1, 2, 3, 4, 5, 6, 7]          # LL then LP
RIGHT = [14, 15, 16, 17, 10, 11, 12, 13]  # RL then RP  -> row i of RIGHT mirrors row i of LEFT
MID = [8, 9]


# ----------------------------------------------------------------------------------------------- features
class RPSFeatures(nn.Module):
    def __init__(self, win_s=4.0, hop_s=0.25, env_win_s=8.0, n_bins=64, fmin=0.5, fmax=20.0, env_smooth_s=0.1,
                 norm='global'):
        super().__init__()
        assert norm in ('global', 'channel')
        self.norm = norm   # 'global': one mean/std per sample -> keeps power differences between channels (asymmetry)
        self.win, self.hop, self.ewin = int(win_s * FS), int(hop_s * FS), int(env_win_s * FS)
        self.n_bins, self.fmin, self.fmax = n_bins, fmin, fmax
        self.smooth = max(1, int(env_smooth_s * FS))
        self.register_buffer('hann', torch.hann_window(self.win, periodic=False), persistent=False)
        self.register_buffer('hann_e', torch.hann_window(self.ewin, periodic=False), persistent=False)

    def _frames(self, x, win):
        x = F.pad(x, (win // 2, win // 2), mode='reflect')
        return x.unfold(-1, win, self.hop)                       # (..., F, win)

    def _acf(self, fr, window, max_lag):
        fr = fr - fr.mean(-1, keepdim=True)
        fr = fr * window
        n = 2 * fr.shape[-1]
        p = torch.fft.rfft(fr, n=n).abs().pow(2)
        r = torch.fft.irfft(p, n=n)[..., :max_lag]
        r = r / r[..., :1].clamp_min(1e-6)
        r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
        B = r.shape[:-1]
        r = F.adaptive_avg_pool1d(r.reshape(-1, 1, max_lag), self.n_bins)
        return r.reshape(*B, self.n_bins)

    @torch.no_grad()
    def forward(self, x):
        """x: (B, C, T) float, microvolt-scale filtered EEG. Returns (B, C, 3, n_bins, F)."""
        x = x.float()
        B, C, T = x.shape
        # R: signal autocorrelation, lags 0-2 s
        fr = self._frames(x, self.win)                           # (B, C, F, win)
        r_sig = self._acf(fr, self.hann, self.win // 2)
        # S: log power spectrum 0.5-20 Hz
        spec = torch.fft.rfft((fr - fr.mean(-1, keepdim=True)) * self.hann).abs().pow(2)
        freqs = torch.fft.rfftfreq(self.win, 1 / FS).to(x.device)
        band = (freqs >= self.fmin) & (freqs <= self.fmax)
        s = torch.log(spec[..., band] + 1e-3)
        Fn = s.shape[2]
        s = F.interpolate(s.reshape(-1, 1, s.shape[-1]), size=self.n_bins, mode='linear', align_corners=False)
        s = s.reshape(B, C, Fn, self.n_bins)
        dims = (1, 2, 3) if self.norm == 'global' else (2, 3)
        s = (s - s.mean(dims, keepdim=True)) / (s.std(dims, keepdim=True) + 1e-4)
        s = torch.clamp(s / 3, -1, 1)
        # P: envelope autocorrelation, lags 0-4 s
        X = torch.fft.fft(x - x.mean(-1, keepdim=True), dim=-1)
        h = torch.zeros(T, device=x.device)
        h[0] = 1
        if T % 2 == 0:
            h[T // 2] = 1; h[1:T // 2] = 2
        else:
            h[1:(T + 1) // 2] = 2
        env = torch.fft.ifft(X * h, dim=-1).abs()
        env = F.avg_pool1d(env.reshape(B * C, 1, T), self.smooth, stride=1, padding=self.smooth // 2,
                           count_include_pad=False)[..., :T].reshape(B, C, T)
        fe = self._frames(env, self.ewin)[:, :, :Fn]
        r_env = self._acf(fe, self.hann_e, self.ewin // 2)
        out = torch.stack([r_sig, r_env, s], dim=2)              # (B, C, 3, F, n_bins)
        return out.transpose(-1, -2).contiguous()                # (B, C, 3, n_bins, F)


def stack_channels(rps, idx, width=256):
    """(B, C, 3, H, F) -> image (B, 3, len(idx)*H, width): channels stacked top-to-bottom, lag/freq axis flipped so
    lag 0 / low frequency sits at the bottom of each band."""
    x = rps[:, idx].flip(-2)                                     # (B, n, 3, H, F)
    B, n, c, H, Fn = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(B, c, n * H, Fn)
    return F.interpolate(x, size=(n * H, width), mode='bilinear', align_corners=False)


# ----------------------------------------------------------------------------------------------- morphology stream
class ResBlock1D(nn.Module):
    def __init__(self, ci, co, stride=2, k=7):
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, k, stride, k // 2, bias=False); self.b1 = nn.BatchNorm1d(co)
        self.c2 = nn.Conv1d(co, co, k, 1, k // 2, bias=False); self.b2 = nn.BatchNorm1d(co)
        self.skip = nn.Sequential(nn.Conv1d(ci, co, 1, stride, bias=False), nn.BatchNorm1d(co))

    def forward(self, x):
        return F.gelu(self.b2(self.c2(F.gelu(self.b1(self.c1(x))))) + self.skip(x))


class MorphEncoder(nn.Module):
    """Shared 1D CNN applied to every bipolar channel of the raw 200 Hz waveform (sharpness, shape, amplitude).
    Returns tokens (B, C, n_tok, d) — avg+max pooled so isolated sharp transients survive pooling — and, if acf=True,
    learned-feature periodicity tokens: the autocorrelation (lag 0-3 s) of the 25 Hz feature maps inside each of the
    n_tok windows. A discharge detector that fires periodically produces a peak at the discharge interval."""
    def __init__(self, d=192, n_tok=8, scale=64.0, width=1.0, acf=False, acf_bins=15, acf_lag_s=3.0, d_acf=96):
        super().__init__()
        chs = [int(round(c * width)) for c in (32, 48, 64, 96, 128, 160)] + [d]
        self.stem = nn.Sequential(nn.Conv1d(1, chs[0], 7, 2, 3, bias=False), nn.BatchNorm1d(chs[0]), nn.GELU())
        self.low = nn.Sequential(*[ResBlock1D(chs[i - 1], chs[i]) for i in (1, 2, 3)])      # -> 25 Hz, chs[3]
        self.high = nn.Sequential(*[ResBlock1D(chs[i - 1], chs[i]) for i in (4, 5, 6)])     # -> ~3 Hz, d
        self.n_tok, self.scale, self.d, self.acf = n_tok, scale, d, acf
        if acf:
            self.acf_lag = int(acf_lag_s * 25)
            self.acf_bins = acf_bins
            self.acf_proj = nn.Sequential(nn.Linear(chs[3] * acf_bins, d_acf), nn.GELU())
            self.d_acf = d_acf

    def _acf_tokens(self, z):
        N, c, T = z.shape
        L = T // self.n_tok
        w = z[..., :L * self.n_tok].reshape(N, c, self.n_tok, L)
        w = w - w.mean(-1, keepdim=True)
        p = torch.fft.rfft(w.float(), n=2 * L).abs().pow(2)
        r = torch.fft.irfft(p, n=2 * L)[..., :min(self.acf_lag, L) + 1]
        r = r[..., 1:] / r[..., :1].clamp_min(1e-6)                                    # lags 1..max, normalised
        r = F.adaptive_avg_pool1d(r.reshape(-1, 1, r.shape[-1]), self.acf_bins).reshape(N, c, self.n_tok, -1)
        return self.acf_proj(r.permute(0, 2, 1, 3).reshape(N, self.n_tok, -1).to(z.dtype))   # (N, n_tok, d_acf)

    def forward(self, x):
        B, C, T = x.shape
        z = torch.clamp(x.float() / self.scale, -16, 16).reshape(B * C, 1, T)   # fixed uV scaling keeps amplitude
        z = self.low(self.stem(z))                                              # (B*C, c3, ~1250) at 25 Hz
        a = self._acf_tokens(z).reshape(B, C, self.n_tok, -1) if self.acf else None
        z = self.high(z)
        z = F.adaptive_avg_pool1d(z, self.n_tok) + F.adaptive_max_pool1d(z, self.n_tok)
        return z.reshape(B, C, self.d, self.n_tok).transpose(2, 3), a           # (B, C, n_tok, d), (B, C, n_tok, d_acf)


# ----------------------------------------------------------------------------------------------- model
class PRISM(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True, d=256, n_layers=2, heads=4, drop=0.2,
                 drop_path=0.1, width=256, head='acns', fusion='sym', repr='rps', norm='global', specaug=True,
                 wave_width=1.0):
        super().__init__()
        assert head in ('acns', 'flat') and fusion in ('sym', 'concat') and \
            repr in ('rps', 'spec', 'rps+wave', 'wave', 'wave+acf')
        self.use_img = repr in ('rps', 'spec', 'rps+wave')
        self.use_wave = repr in ('rps+wave', 'wave', 'wave+acf')
        self.use_acf = repr == 'wave+acf'
        assert not (self.use_wave and fusion == 'concat'), 'concat ablation is defined for the image stream only'
        self.head_type, self.fusion, self.repr = head, fusion, repr   # 'flat' / 'concat' / 'spec' = ablations
        import timm
        self.width = width
        self.feat = RPSFeatures(norm=norm)
        self.specaug = specaug
        if self.use_img:
            self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='',
                                         drop_path_rate=drop_path)
            C = self.enc.num_features
            cfg = getattr(self.enc, 'pretrained_cfg', {}) or {}
        else:
            self.enc, C, cfg = None, 0, {}
        mean = torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get('std', (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        self.register_buffer('img_mean', mean, persistent=False)
        self.register_buffer('img_std', std, persistent=False)
        DW = 192
        self.morph = MorphEncoder(d=DW, n_tok=width // 32, width=wave_width, acf=self.use_acf) if self.use_wave else None
        DA = self.morph.d_acf if self.use_acf else 0
        in_tok = (3 * C if self.use_img else 0) + (3 * DW if self.use_wave else 0) + 3 * DA
        in_lat = (2 * C if self.use_img else 0) + (2 * DW if self.use_wave else 0) + 2 * DA
        self.proj = nn.Sequential(nn.Linear(in_tok, d), nn.GELU())
        self.dmix = nn.Linear(2 * d, d)
        self.n_tok = width // 32
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True,
                                           activation='gelu')
        self.tx = nn.TransformerEncoder(layer, n_layers)
        self.pattern = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, 4 if head == 'acns' else 6))
        self.lat = nn.Sequential(nn.LayerNorm(in_lat), nn.Dropout(drop), nn.Linear(in_lat, d), nn.GELU(),
                                 nn.Linear(d, 2))                                               # PD, RDA laterality

    def _img(self, rps, idx):
        if self.repr == 'spec':                                  # ablation: spectrum only, no periodicity maps
            rps = rps[:, :, 2:3].expand(-1, -1, 3, -1, -1)
        x = stack_channels(rps, idx, self.width)                 # values in [-1, 1]
        x = (x + 1) / 2                                          # -> [0, 1] like an image
        return (x - self.img_mean) / self.img_std

    def _tokens(self, img):
        f = self.enc(img)                                        # (B, C, h, w)
        if f.shape[-1] != self.n_tok:
            f = F.adaptive_avg_pool2d(f, (f.shape[-2], self.n_tok))
        return f.mean(2).transpose(1, 2)                         # (B, w, C)

    def _mask(self, rps, n=2, max_w=20):
        # training-only time masking on the RPS frames (same frames for every channel, up to 5 s each)
        B, Fn = rps.shape[0], rps.shape[-1]
        keep = torch.ones(B, Fn, device=rps.device)
        for _ in range(n):
            w = torch.randint(0, max_w + 1, (B,), device=rps.device)
            s0 = (torch.rand(B, device=rps.device) * (Fn - w)).long()
            ar = torch.arange(Fn, device=rps.device)[None]
            keep = keep * ~((ar >= s0[:, None]) & (ar < (s0 + w)[:, None]))
        return rps * keep[:, None, None, None, :]

    def forward(self, x, return_parts=False):
        B = x.shape[0]
        parts, lat_in = [], []
        if self.use_img:
            rps = self.feat(x)
            if self.training and self.specaug:
                rps = self._mask(rps)
            lr = self._tokens(torch.cat([self._img(rps, LEFT), self._img(rps, RIGHT)]))
            tl, tr = lr[:B], lr[B:]
            tm = self._tokens(self._img(rps, MID))
            if self.fusion == 'sym':
                s, a = (tl + tr) / 2, (tl - tr).abs()
            else:                                                # ablation: plain concatenation (not flip-invariant)
                s, a = tl, tr
            parts += [s, a, tm]; lat_in += [a.mean(1), s.mean(1)]
        if self.use_wave:
            mt, at = self.morph(x)                               # (B, 18, w, DW), (B, 18, w, DA) or None
            for t in ([mt, at] if at is not None else [mt]):
                L, R, M = t[:, LEFT], t[:, RIGHT], t[:, MID]     # mirrored rows: L[:, i] <-> R[:, i]
                ws, wa = (L + R) / 2, (L - R).abs()              # per electrode pair, then pooled over the 8 pairs
                ws, wa, wm = ws.mean(1) + ws.amax(1), wa.mean(1) + wa.amax(1), M.mean(1)
                parts += [ws, wa, wm]; lat_in += [wa.mean(1), ws.mean(1)]
        tok = self.proj(torch.cat(parts, -1))
        prev = torch.cat([tok[:, :1], tok[:, :-1]], 1)
        tok = self.dmix(torch.cat([tok, tok - prev], -1)) + self.pos
        tok = self.tx(tok)
        pooled = torch.cat([tok.mean(1), tok.amax(1)], -1)
        if self.head_type == 'flat':
            return F.log_softmax(self.pattern(pooled).float(), -1)
        lp = F.log_softmax(self.pattern(pooled).float(), -1)
        z = self.lat(torch.cat(lat_in, -1)).float()
        ls, ln = F.logsigmoid(z), F.logsigmoid(-z)
        logp = torch.stack([lp[:, 0], lp[:, 1] + ls[:, 0], lp[:, 1] + ln[:, 0],
                            lp[:, 2] + ls[:, 1], lp[:, 2] + ln[:, 1], lp[:, 3]], 1)
        if return_parts:
            return logp, {'pattern': lp.exp(), 'lateralized': torch.sigmoid(z)}
        return logp                                              # log-probabilities, sums to 1 in prob space


# ----------------------------------------------------------------------------------------------- losses
def kl_loss(logp, target, weight=None):
    t = target.clamp_min(1e-7)
    l = (target * (t.log() - logp)).sum(1)
    return (l * weight).mean() if weight is not None else l.mean()


def dm_nll(logp, votes, log_kappa):
    """Dirichlet-multinomial negative log-likelihood of the raw vote counts under concentration kappa * p."""
    kappa = log_kappa.exp()
    a = kappa * logp.exp().clamp_min(1e-6)
    n = votes.sum(1)
    ll = torch.lgamma(kappa) - torch.lgamma(n + kappa) + (torch.lgamma(votes + a) - torch.lgamma(a)).sum(1)
    return -(ll / n.clamp_min(1)).mean()
