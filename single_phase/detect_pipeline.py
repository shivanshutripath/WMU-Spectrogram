"""
Single-phase (1-channel) detection pipeline: STFT, sliding windows, per-window
labels, normalization, balancing, and the event-level detection metrics
(temporal IoU, VOC average precision, precision/recall). Mirrors the three-phase
data_pipeline.py / aug_pipeline.py / run_comparison.py but for one channel.

STFT convention matches the paper (data_pipeline.py): Hamming window L=64, hop
H=16 (75% overlap), keep the lowest F=24 bins, log(1+P) compression.
"""
import numpy as np

# ---- STFT / window configuration (paper convention) ----
L = 64                      # STFT frame length [samples]
H_DEFAULT = 16              # default hop (75% overlap)
F = 24                      # retained low-frequency bins
WIN_FRAMES = 16             # frames per U-Net window (time axis)
HAMMING = np.hamming(L)


def sraw(H):
    """Raw samples spanned by one WIN_FRAMES-frame window at hop H."""
    return (WIN_FRAMES - 1) * H + L


def stft_logspec(x, H=H_DEFAULT):
    """x:(T,) -> Z:(M,F) log-power spectrogram (single channel) at hop H."""
    Tn = x.shape[0]
    M = (Tn - L) // H + 1
    idx = np.arange(L)[None, :] + H * np.arange(M)[:, None]     # (M,L)
    frames = x[idx] * HAMMING[None, :]
    Xf = np.fft.rfft(frames, axis=1)
    P = (np.abs(Xf) ** 2) / L
    return np.log1p(P[:, :F]).astype(np.float32)               # (M,F)


def make_windows(x, Z, s, e, stride_frames, H=H_DEFAULT):
    """Slide WIN_FRAMES-frame windows. Returns dict with:
       U:(n,SRAW,1) raw, P:(n,WIN_FRAMES,F,1) spectrogram, y:(n,), mu:(n,) center sample."""
    SRAW = sraw(H)
    M = Z.shape[0]
    starts = np.arange(0, M - WIN_FRAMES + 1, stride_frames)
    U = np.empty((len(starts), SRAW, 1), np.float32)
    P = np.empty((len(starts), WIN_FRAMES, F, 1), np.float32)
    y = np.empty(len(starts), np.int64)
    mu = np.empty(len(starts), np.float64)
    for i, mk in enumerate(starts):
        r0 = mk * H
        U[i, :, 0] = x[r0:r0 + SRAW]
        P[i, :, :, 0] = Z[mk:mk + WIN_FRAMES, :]
        c = mk * H + ((WIN_FRAMES - 1) * H + L) / 2.0
        mu[i] = c
        y[i] = 1 if (s <= c <= e) else 0
    return dict(U=U, P=P, y=y, mu=mu, starts=starts)


def windows_for_event(ev, stride_frames, H=H_DEFAULT):
    x = ev['x']
    Z = stft_logspec(x, H)
    w = make_windows(x, Z, ev['s'], ev['e'], stride_frames, H)
    w.update(s=ev['s'], e=ev['e'], ftype=ev['ftype'], Z=Z)
    return w


# ---- per-channel normalization (fit on train windows) ----
def fit_norm(train_ws):
    U = np.concatenate([w['U'] for w in train_ws], 0)
    P = np.concatenate([w['P'] for w in train_ws], 0)
    return dict(uU=U.mean((0, 1)), sU=U.std((0, 1)) + 1e-6,
                uP=P.mean((0, 1)), sP=P.std((0, 1)) + 1e-6)

def nU(U, z): return ((U - z['uU']) / z['sU']).astype(np.float32)
def nP(P, z): return ((P - z['uP']) / z['sP']).astype(np.float32)


def balance(ws, neg_per_pos, seed):
    y = np.concatenate([w['y'] for w in ws]); rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    keep = rng.choice(neg, size=min(len(neg), neg_per_pos * len(pos)), replace=False)
    idx = np.concatenate([pos, keep]); rng.shuffle(idx)
    return idx, y


def window_ftypes(ws):
    return np.concatenate([np.full(len(w['y']), w['ftype'], dtype=np.int64) for w in ws])


# ==================== event-level detection metrics ====================
def sustained_interval(mu, p, tau, Lmin=2):
    """Longest run of windows with p>=tau; sub-window edge interpolation."""
    fire = p >= tau; best = None; i = 0; n = len(fire)
    while i < n:
        if fire[i]:
            j = i
            while j + 1 < n and fire[j + 1]:
                j += 1
            if (j - i + 1) >= Lmin and (best is None or (j - i) > (best[1] - best[0])):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    if best is None:
        return False, None, None
    i, j = best
    sh = float(mu[i])
    if i > 0 and p[i] != p[i - 1]:
        f = (tau - p[i - 1]) / (p[i] - p[i - 1])
        sh = float(mu[i - 1] + np.clip(f, 0.0, 1.0) * (mu[i] - mu[i - 1]))
    eh = float(mu[j])
    if j + 1 < n and p[j] != p[j + 1]:
        f = (p[j] - tau) / (p[j] - p[j + 1])
        eh = float(mu[j] + np.clip(f, 0.0, 1.0) * (mu[j + 1] - mu[j]))
    return True, sh, eh


def iou_samp(sh, eh, s, e):
    inter = max(0.0, min(eh, e) - max(sh, s)); union = max(eh, e) - min(sh, s)
    return inter / union if union > 0 else 0.0


def average_precision(conf, iou_arr, iou_thr, n_gt):
    """VOC all-point AP; one GT and one detection per scene (already NMS'd)."""
    if n_gt == 0 or len(conf) == 0:
        return 0.0, np.array([0.0]), np.array([1.0])
    order = np.argsort(-conf)
    tp = (iou_arr[order] >= iou_thr).astype(np.float64)
    fp = 1.0 - tp
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, rec, prec


def pr_at(conf, iou_arr, iou_thr, n_gt, conf_thr):
    keep = conf >= conf_thr
    tp = int(((iou_arr >= iou_thr) & keep).sum())
    fp = int(((iou_arr < iou_thr) & keep).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    return prec, rec


def best_f1(y, p):
    bf, bt = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 37):
        pr = p >= t
        tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = ((~pr) & (y == 1)).sum()
        f = 2 * tp / (2 * tp + fp + fn + 1e-9)
        if f > bf:
            bf, bt = f, t
    return float(bf), float(bt)


def auc(y, p):
    P = int(y.sum()); N = len(y) - P
    if P == 0 or N == 0:
        return float('nan')
    r = np.argsort(np.argsort(p)) + 1
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))
