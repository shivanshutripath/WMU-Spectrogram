"""
Single-phase PQD *event* generator for the detection / localization / typing task.

Unlike generate_days.py (which emits short, fully-labelled 0.2 s clips for the
classification study), this module emits longer single-phase records in which a
single power-quality disturbance occupies a short sub-interval [s, e] of an
otherwise-normal fundamental. This mirrors the three-phase experiment, where each
60 s record carries one short fault to be detected, localized, and typed.

Each record:
    * length T samples (Fs = 3000 Hz, f0 = 60 Hz)  -> a clean fundamental
      background with measurement noise
    * ONE disturbance of type t in {0,1,2,3} injected on [s, e] with smooth
      onset/offset ramps (no pure step edges)
    * ground-truth interval (s, e) and type t

Disturbance types (the "normal" class is the background / no-disturbance state):
    0  harmonic_distortion  odd harmonics (3,5,7,9,11,13) during [s,e]
    1  interharmonic        non-integer-multiple component(s) during [s,e]
    2  voltage_flicker      fundamental amplitude-modulated at 8-25 Hz during [s,e]
    3  oscillation          sustained damped high-frequency ringing during [s,e]

Six "days" are produced (day-specific seed + small nominal drift); days 1-5 are
TRAIN, day 6 is the held-out TEST day (generalization to an unseen day).
"""
import os
import numpy as np

Fs = 3000.0          # sampling rate [Hz]
F0 = 60.0            # fundamental   [Hz]
T = 3000            # samples per record = 1.0 s
N_DAYS = 6
EVENTS_PER_DAY = 40  # records per day

# disturbance interval duration range [s]
DUR_MIN, DUR_MAX = 0.15, 0.45

TYPES = ['harmonic_distortion', 'interharmonic', 'voltage_flicker', 'oscillation']
# "normal" (background / no disturbance) is class id = len(TYPES)
CLASSES = TYPES + ['normal']

_t = np.arange(T) / Fs


def _edge_env(s, e, rng):
    """Smooth on/off envelope on [s,e] with ~30 ms cosine ramps (no step edges)."""
    env = np.zeros(T)
    env[s:e] = 1.0
    ramp = int(0.03 * Fs)
    if ramp > 0:
        r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
        env[s:s + ramp] *= r[:min(ramp, e - s)]
        env[max(s, e - ramp):e] *= r[::-1][:min(ramp, e - s)]
    return env


def _gen_one(label, rng, amp0):
    """Return (v, s, e). One record with a disturbance of `label` on [s,e]."""
    amp = amp0 * rng.uniform(0.97, 1.03)
    ph = rng.uniform(0, 2 * np.pi)
    t = _t
    fund = amp * np.sin(2 * np.pi * F0 * t + ph)

    dur = int(rng.uniform(DUR_MIN, DUR_MAX) * Fs)
    s = int(rng.uniform(0.15 * T, 0.85 * T - dur))
    e = s + dur
    env = _edge_env(s, e, rng)

    v = fund.copy()
    if label == 0:                               # harmonic distortion
        orders = rng.choice([3, 5, 7, 9, 11, 13],
                            size=rng.integers(2, 5), replace=False)
        for h in orders:
            a = rng.uniform(0.06, 0.20)
            v += env * amp * a * np.sin(2 * np.pi * h * F0 * t + rng.uniform(0, 2 * np.pi))

    elif label == 1:                             # interharmonic
        for _ in range(rng.integers(1, 3)):
            m = rng.uniform(1.4, 4.6)
            if abs(m - round(m)) < 0.15:
                m += 0.3
            a = rng.uniform(0.06, 0.18)
            v += env * amp * a * np.sin(2 * np.pi * m * F0 * t + rng.uniform(0, 2 * np.pi))

    elif label == 2:                             # voltage flicker
        f_fl = rng.uniform(8.0, 25.0)
        depth = rng.uniform(0.08, 0.22)
        mod = depth * np.sin(2 * np.pi * f_fl * t + rng.uniform(0, 2 * np.pi))
        v = amp * (1.0 + env * mod) * np.sin(2 * np.pi * F0 * t + ph)

    else:                                        # oscillation (sustained ringing)
        f_os = rng.uniform(300.0, 900.0)
        a = rng.uniform(0.12, 0.35)
        # sustained but amplitude-fluctuating ring across the interval
        fluct = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(20, 60) * t + rng.uniform(0, 6.28))
        v += env * amp * a * fluct * np.sin(2 * np.pi * f_os * t + rng.uniform(0, 6.28))

    # measurement noise (SNR 30-50 dB)
    snr_db = rng.uniform(30.0, 50.0)
    p_sig = np.mean(v ** 2)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    v = v + rng.normal(0.0, np.sqrt(p_noise), size=v.shape)
    return v.astype(np.float32), s, e


def make_day(day_idx):
    rng = np.random.default_rng(2000 + day_idx)
    amp0 = 1.0 + 0.02 * (day_idx - 3)
    X, S, E, Y = [], [], [], []
    for i in range(EVENTS_PER_DAY):
        label = i % len(TYPES)                    # balanced over the 4 disturbance types
        v, s, e = _gen_one(label, rng, amp0)
        X.append(v); S.append(s); E.append(e); Y.append(label)
    perm = rng.permutation(len(Y))
    X = np.stack(X)[perm]
    S = np.array(S, np.int64)[perm]
    E = np.array(E, np.int64)[perm]
    Y = np.array(Y, np.int64)[perm]
    return X, S, E, Y


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, 'data_det')
    os.makedirs(out, exist_ok=True)
    for d in range(1, N_DAYS + 1):
        X, S, E, Y = make_day(d)
        path = os.path.join(out, f'day{d}.npz')
        np.savez_compressed(path, X=X, s=S, e=E, ftype=Y,
                            types=np.array(TYPES), classes=np.array(CLASSES),
                            Fs=Fs, F0=F0, T=T)
        role = 'TEST ' if d == N_DAYS else 'TRAIN'
        counts = np.bincount(Y, minlength=len(TYPES))
        print(f"day{d} [{role}]  {X.shape[0]} records (T={X.shape[1]})  "
              f"per-type {counts.tolist()}  -> {path}")
    print("disturbance types:", TYPES)
