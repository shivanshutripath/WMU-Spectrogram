"""
Step 1 - Data pipeline for IBR hard-fault detection.

Implements, exactly as in the paper:
  * Loading 6-channel terminal recordings (va,vb,vc,ia,ib,ic) at Fs.
  * Short-time Fourier transform -> per-channel log power spectrogram S_c (Mj x F)
    with a Hamming taper, log(1+P) compression, and the first F harmonic bins kept.
  * Stacking the 6 channels -> event image Z_j in R^{Mj x 6 x F}.
  * Two windowed representations of the SAME time span:
        U_k  (raw waveform)        in R^{Sraw x 6}     -> 1-D U-Net
        P_k  (spectrogram window)  in R^{W    x 6 x F} -> 2-D U-Net
  * Center-based per-window label  y_k = 1{ s_j <= mu_k <= e_j }.

Because the supplied recordings are unlabeled normal data, this module can also
INJECT physically-motivated hard faults (small magnitude, spectral signature,
little RMS change) into a real normal background to obtain ground-truth [s,e].
Swap inject=False and pass real [s,e] to train on real labels instead.
"""
import os
import numpy as np, pandas as pd

# ---------------- global signal / STFT configuration ----------------
Fs   = 3000.0      # sampling rate [Hz]
F0   = 60.0        # fundamental [Hz]
L    = 64          # STFT frame length [samples]
# STFT hop is configurable (env STFT_HOP) so we can sweep the overlap:
#   H=64 -> 0%/"25%" overlap (M=938), H=32 -> 50% (M=1874), H=16 -> 75% (M=3746).
H    = int(os.environ.get('STFT_HOP', 16))
OVL  = int(round((1 - H / L) * 100))   # percent overlap implied by the hop
KBIN = L // 2 + 1                       # rfft bins = 33
F    = 24          # number of retained harmonic/low-freq bins
WIN_FRAMES = 16    # W : frames per window  (2-D time axis)
SRAW = (WIN_FRAMES - 1) * H + L         # raw samples per window = 544
HAMMING = np.hamming(L)
CH = ['Va','Vb','Vc','Ia','Ib','Ic']

def load_event(path):
    """Return raw waveform array x of shape (T, 6) = (va,vb,vc,ia,ib,ic) for IBR1."""
    df = pd.read_csv(path)
    c = list(df.columns)
    x = df[c[1:7]].to_numpy(dtype=np.float64)   # cols 1..6 = IBR1 V&I
    return x

def stft_logspec(x):
    """x:(T,6) -> Z:(M,6,F) log power spectrogram, per the paper."""
    T = x.shape[0]
    M = (T - L)//H + 1
    idx = np.arange(L)[None,:] + H*np.arange(M)[:,None]      # (M,L) frame indices
    Z = np.empty((M, 6, F), dtype=np.float32)
    for c in range(6):
        frames = x[idx, c] * HAMMING[None, :]               # (M,L) tapered
        Xf = np.fft.rfft(frames, axis=1)                    # (M,KBIN)
        P  = (np.abs(Xf)**2) / L                            # power spectrum
        Z[:, c, :] = np.log1p(P[:, :F]).astype(np.float32)  # keep first F bins, log compress
    return Z                                                # (M,6,F)

# ---------------- fault injection (creates ground-truth labels) ----------------
def inject_fault(x, rng):
    """Add ONE small hard-fault signature to a copy of normal waveform x:(T,6).
    Returns (x_faulted, s, e). RMS barely changes; spectral content shifts."""
    x = x.copy(); T = x.shape[0]
    t = np.arange(T)/Fs
    dur = rng.uniform(0.25, 1.5)                 # fault duration [s]
    s = int(rng.uniform(0.1, 0.85)*T)
    e = min(T-1, s + int(dur*Fs))
    mask = np.zeros(T); mask[s:e] = 1.0
    # smooth onset/offset (transient) -> avoids a pure step
    ramp = np.clip(np.minimum(np.arange(T)-s, e-np.arange(T))/ (0.03*Fs), 0, 1)
    env = mask*ramp
    ftype = rng.integers(0,4)
    phases = rng.choice([0,1,2], size=rng.integers(1,4), replace=False)  # affected phase set
    Iamp = np.median(np.abs(x[:, 3:6])) + 1e-6   # current scale
    Vamp = np.median(np.abs(x[:, 0:3])) + 1e-6   # voltage scale
    for ph in phases:
        ic, vc = 3+ph, ph
        if ftype == 0:   # harmonic injection in current (5th & 7th)
            a = rng.uniform(0.20,0.40)
            x[:,ic] += env*a*Iamp*(np.sin(2*np.pi*5*F0*t+rng.uniform(0,6.28))
                                   +0.7*np.sin(2*np.pi*7*F0*t+rng.uniform(0,6.28)))
        elif ftype == 1: # high-impedance fault: interharmonic + broadband, small bump
            a = rng.uniform(0.10,0.22)
            inter = np.sin(2*np.pi*rng.uniform(150,260)*t+rng.uniform(0,6.28))
            x[:,ic] += env*a*Iamp*(inter + 0.35*rng.standard_normal(T))
        elif ftype == 2: # damped transient oscillation (V & I)
            fo = rng.uniform(250,750); dec = rng.uniform(8,25)
            osc = np.sin(2*np.pi*fo*(t-t[s]))*np.exp(-dec*np.maximum(t-t[s],0))
            x[:,ic] += mask*rng.uniform(0.25,0.5)*Iamp*osc
            x[:,vc] += mask*rng.uniform(0.03,0.07)*Vamp*osc
        else:            # nonlinear flat-topping (adds odd harmonics), tiny RMS change
            seg = slice(s,e); lim = np.percentile(np.abs(x[seg,ic]), 70)
            xc = np.clip(x[seg,ic], -lim, lim)
            x[seg,ic] = xc * (np.std(x[seg,ic])/(np.std(xc)+1e-9))  # renormalize -> RMS preserved
    return x, s, e

# ---------------- windowing + labeling ----------------
def make_windows(x, Z, s, e, stride_frames=8):
    """Slide W-frame windows. Return dict of arrays:
       U:(n,SRAW,6) raw, P:(n,W,6,F) spectrogram, y:(n,), mu:(n,) center sample."""
    M = Z.shape[0]
    starts = np.arange(0, M - WIN_FRAMES + 1, stride_frames)   # window start frame mk
    U = np.empty((len(starts), SRAW, 6), np.float32)
    P = np.empty((len(starts), WIN_FRAMES, 6, F), np.float32)
    y = np.empty(len(starts), np.int64)
    mu = np.empty(len(starts), np.float64)
    for i, mk in enumerate(starts):
        r0 = mk*H
        U[i] = x[r0:r0+SRAW, :]
        P[i] = Z[mk:mk+WIN_FRAMES, :, :]
        c = mk*H + ((WIN_FRAMES-1)*H + L)/2.0        # center sample mu_k
        mu[i] = c
        y[i] = 1 if (s <= c <= e) else 0
    return dict(U=U, P=P, y=y, mu=mu, starts=starts)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = load_event('/mnt/user-data/uploads/DataExport_20241001_000000Z_CSV.xls')
    print("raw event:", x.shape, "duration[s]=", x.shape[0]/Fs)
    xf, s, e = inject_fault(x, rng)
    Z = stft_logspec(xf)
    print("spectrogram Z:", Z.shape, " (M,6,F)")
    w = make_windows(xf, Z, s, e, stride_frames=16)
    print("windows:", w['U'].shape, w['P'].shape, "fault windows=", int(w['y'].sum()),
          "of", len(w['y']), " fault[s,e] samples=",(s,e))
