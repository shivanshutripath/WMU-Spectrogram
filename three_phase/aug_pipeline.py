"""
Data pipeline for the *real augmented* IBR hard-fault dataset
(waveforms_part01..08_events000-199_augmented.npz).

Each .npz part holds 25 events:
    X            (25, 60000, 6)  raw waveform [Va,Vb,Vc, Ia,Ib,Ic]  @ Fs=1000 Hz
    s_sample/e_sample (25,)      ground-truth fault interval [start,end] in samples
    fault_type   (25,)           0..3
    background   (25,)           'A' | 'B' | 'C'
    faulted_mask (25, 3)         which of the 3 phases are involved

This module turns those events into the two windowed representations the
U-Nets consume (U_k raw waveform, P_k spectrogram window) plus a frame-grid
representation + interval "box" the 1-D YOLO detector consumes.  Every model
sees ALL SIX channels (voltage AND current).

The STFT params are inherited from data_pipeline.py (L=64, H=32, F=24); they are
sampling-rate agnostic, so they apply unchanged to the 1 kHz data here.
"""
import glob, os
import numpy as np
import data_pipeline as dp

H, L, F, WIN_FRAMES, SRAW = dp.H, dp.L, dp.F, dp.WIN_FRAMES, dp.SRAW
CH = dp.CH

# YOLO detectors use a coarser temporal grid than the dense U-Nets: the U-Nets
# benefit from the fine STFT hop H for sample-accurate edges, but the YOLO
# single-box objectness is sparse and trains better on the original ~32-sample
# grid. Subsampling the STFT frames by YSTRIDE keeps that grid fixed regardless
# of the (now finer) hop, so YOLO sees the same input as the 50%-overlap setup.
YSTRIDE = max(1, 32 // H)        # =2 when H=16 -> effective YOLO hop = 32 samples
YHOP = H * YSTRIDE               # YOLO frame hop in samples
PARTS = sorted(glob.glob(os.path.join(os.path.dirname(__file__) or '.',
                                      'data', 'waveforms_part*_augmented.npz')))


def load_all_events():
    """Yield one dict per event across all parts (lazy-ish: loads a part at a time)."""
    evs = []
    for path in PARTS:
        d = np.load(path, allow_pickle=True)
        X = d['X']                              # (25,60000,6)
        for i in range(X.shape[0]):
            evs.append(dict(
                x=X[i].astype(np.float64),
                s=int(d['s_sample'][i]), e=int(d['e_sample'][i]),
                ftype=int(d['fault_type'][i]), bg=str(d['background'][i]),
                eid=int(d['event_ids'][i]),
                mask=d['faulted_mask'][i].copy()))
    return evs


def windows_for_event(ev, stride_frames):
    """Produce U/P windows + per-window labels (reuses data_pipeline windowing)."""
    x = ev['x']
    Z = dp.stft_logspec(x)                                   # (M,6,F)
    w = dp.make_windows(x, Z, ev['s'], ev['e'], stride_frames=stride_frames)
    w.update(s=ev['s'], e=ev['e'], bg=ev['bg'], ftype=ev['ftype'], Z=Z)
    return w


def frame_grid_for_event(ev):
    """YOLO input: per-frame spectral features for ALL 6 channels.
       Returns Fseq:(M, 6*F) and the fault interval in *frame* coordinates."""
    Z = dp.stft_logspec(ev['x'])[::YSTRIDE]                  # (M,6,F) on coarse grid
    M = Z.shape[0]
    Fseq = Z.reshape(M, 6 * F).astype(np.float32)            # stack V&I spectra
    # sample -> frame coordinate:  frame f is centred at sample f*YHOP + L/2
    sf = (ev['s'] - L / 2.0) / YHOP
    efr = (ev['e'] - L / 2.0) / YHOP
    return Fseq, float(sf), float(efr), M


def raw_grid_for_event(ev):
    """Raw-waveform YOLO-1D input: per-frame RAW samples for all 6 channels (NO
       STFT), using the same L/H framing as the spectrogram so the temporal grid
       (and hence the box/cell geometry) is identical -- the only difference from
       frame_grid_for_event is raw samples instead of their log-spectrum.
       Returns Fseq:(M, L*6) and the fault interval in *frame* coordinates."""
    x = ev['x']
    T = x.shape[0]
    M = (T - L) // YHOP + 1
    idx = np.arange(L)[None, :] + YHOP * np.arange(M)[:, None]  # (M,L) frame sample indices
    frames = x[idx, :]                                       # (M,L,6)
    Fseq = frames.reshape(M, L * 6).astype(np.float32)       # raw samples per frame
    sf = (ev['s'] - L / 2.0) / YHOP
    efr = (ev['e'] - L / 2.0) / YHOP
    return Fseq, float(sf), float(efr), M


# ---------------- normalization (per-channel, fit on train) ----------------
def fit_norm(train_ws):
    U = np.concatenate([w['U'] for w in train_ws], 0)
    P = np.concatenate([w['P'] for w in train_ws], 0)
    return dict(uU=U.mean((0, 1)), sU=U.std((0, 1)) + 1e-6,
                uP=P.mean((0, 1)), sP=P.std((0, 1)) + 1e-6)

def nU(U, z): return ((U - z['uU']) / z['sU']).astype(np.float32)
def nP(P, z): return ((P - z['uP']) / z['sP']).transpose(0, 1, 3, 2).astype(np.float32)  # ->(n,T,F,6)


def fit_norm_grid(train_grids):
    Fcat = np.concatenate([g[0] for g in train_grids], 0)
    return dict(u=Fcat.mean(0), s=Fcat.std(0) + 1e-6)

def nGrid(Fseq, z): return ((Fseq - z['u']) / z['s']).astype(np.float32)


def frame_grid2d_for_event(ev):
    """YOLO-2D input: full-event spectrogram as (M, F, 6) (time, freq, channels).
       Returns Zg:(M,F,6) and the fault interval in *frame* coordinates."""
    Z = dp.stft_logspec(ev['x'])[::YSTRIDE]                  # (M,6,F) on coarse grid
    Zg = Z.transpose(0, 2, 1).astype(np.float32)             # (M,F,6)
    M = Zg.shape[0]
    sf = (ev['s'] - L / 2.0) / YHOP
    efr = (ev['e'] - L / 2.0) / YHOP
    return Zg, float(sf), float(efr), M


def fit_norm_grid2d(train_grids):
    Z = np.concatenate([g[0] for g in train_grids], 0)       # (n*M, F, 6)
    return dict(u=Z.mean(0), s=Z.std(0) + 1e-6)

def nGrid2d(Zg, z): return ((Zg - z['u']) / z['s']).astype(np.float32)


# ---------------- balancing for window classifier training ----------------
def balance(ws, neg_per_pos, seed):
    y = np.concatenate([w['y'] for w in ws]); rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    keep = rng.choice(neg, size=min(len(neg), neg_per_pos * len(pos)), replace=False)
    idx = np.concatenate([pos, keep]); rng.shuffle(idx)
    return idx, y


def split_events(evs, n_train, n_val, n_test, seed=0):
    """Train/Val on backgrounds A+B; Test on the UNSEEN background C (no leakage)."""
    rng = np.random.default_rng(seed)
    ab = [e for e in evs if e['bg'] in ('A', 'B')]
    c  = [e for e in evs if e['bg'] == 'C']
    rng.shuffle(ab); rng.shuffle(c)
    train = ab[:n_train]
    val   = ab[n_train:n_train + n_val]
    test  = c[:n_test]
    return train, val, test
