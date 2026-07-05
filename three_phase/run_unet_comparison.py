"""
Three-phase fault detection: U-Net-1D (raw waveform) vs U-Net-2D (spectrogram).

Compares the two matched U-Nets on the REAL augmented IBR dataset, using BOTH
voltage and current (all 6 channels):

    * U-Net-1D  window classifier on the raw waveform window U_k (time domain)
    * U-Net-2D  window classifier on the log-STFT spectrogram window P_k

Both U-Nets share channel widths (8,16,32,64), so any performance gap is due to
the representation (raw vs. spectrogram), not model capacity. Each carries a
parallel 4-way fault-type head trained only on fault windows.

Protocol (no leakage): TRAIN/VAL on backgrounds A+B, TEST on the UNSEEN
background C. Reports window-level metrics (AUC, best-F1) and event-level
detection/localization/typing metrics (AP@IoU, mAP[.5:.95], recall, temporal
IoU, edge errors, fault-type accuracy), writes plots + summary.json to ./plots/.

Run:  python3 run_unet_comparison.py

Epoch counts can be overridden for a quick smoke test, e.g.:
      UNET_EPOCHS=2 UNET2D_EPOCHS=2 python3 run_unet_comparison.py
"""
import os, time, json
import numpy as np
import data_pipeline as dp
import unet_numpy as un
import aug_pipeline as ap

# ---------------- config ----------------
FS = 1000.0                       # sampling rate of the augmented dataset (Hz)
N_CLS = 4                         # fault types {0,1,2,3}
N_TRAIN, N_VAL, N_TEST = 100, 34, 66   # all 134 A+B (train+val), all 66 C (test)
STR_TR, STR_EV = 8, 4
UNET_EPOCHS = int(os.environ.get('UNET_EPOCHS', 20))
UNET2D_EPOCHS = int(os.environ.get('UNET2D_EPOCHS', 30))   # spectrogram benefits from extra epochs
UNET_BS, UNET_LR = 64, 2e-3
LAM_CLS = 1.0                     # weight for fault-type head (1-D)
LAM_CLS_2D = 0.5                  # lighter cls weight when coupled into 2-D trunk
IOU_THR = 0.5                     # detection TP threshold
IOU_SWEEP = np.round(np.arange(0.5, 0.96, 0.05), 2)   # mAP@[.5:.95], COCO-style
AP_TAUS = (0.5, 0.7, 0.8, 0.9)                        # individual AP columns to print
SEED = 0
PLOTDIR = os.path.join(os.path.dirname(__file__) or '.', 'plots')
os.makedirs(PLOTDIR, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PLOTDIR, '.mplcache'))


# ---------------- metrics ----------------
def window_ftypes(ws):
    """Per-window fault type label (= parent event type, one value per window)."""
    return np.concatenate([np.full(len(w['y']), w['ftype'], dtype=np.int64) for w in ws])


def predict_unet(net, X, bs=256):
    """Return p(fault) and argmax fault-type per window (type n/a if n_cls=0)."""
    p1, cls = [], []
    for b in range(0, len(X), bs):
        lg = net.forward(X[b:b + bs])
        p1.append(un.softmax(lg)[:, 1])
        if net.n_cls:
            cls.append(un.softmax(net.cls_logits).argmax(1))
    p1 = np.concatenate(p1)
    return p1, (np.concatenate(cls) if net.n_cls else None)


def softmax_p1(net, X, bs=256):
    return predict_unet(net, X, bs)[0]


def best_f1(y, p):
    bf, bt = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 37):
        pr = p >= t; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = ((~pr) & (y == 1)).sum()
        f = 2 * tp / (2 * tp + fp + fn + 1e-9)
        if f > bf: bf, bt = f, t
    return float(bf), float(bt)


def auc(y, p):
    P = int(y.sum()); N = len(y) - P
    if P == 0 or N == 0: return float('nan')
    r = np.argsort(np.argsort(p)) + 1
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def sustained_interval(mu, p, tau, Lmin=3):
    fire = p >= tau; best = None; i = 0; n = len(fire)
    while i < n:
        if fire[i]:
            j = i
            while j + 1 < n and fire[j + 1]: j += 1
            if (j - i + 1) >= Lmin and (best is None or (j - i) > (best[1] - best[0])): best = (i, j)
            i = j + 1
        else: i += 1
    if best is None: return False, None, None
    i, j = best
    # Sub-window edge refinement: linearly interpolate where p crosses tau between
    # the boundary windows (removes the quantization/inset bias, suits the smooth
    # spectrogram probability ramp).
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
    """VOC all-point AP for a single fault class, one GT per scene, one detection
       per scene (already NMS'd)."""
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
    for i in range(len(mpre) - 2, -1, -1):           # precision envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap_ = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap_, rec, prec


def pr_at(conf, iou_arr, iou_thr, n_gt, conf_thr):
    """Precision & recall at a fixed confidence operating point."""
    keep = conf >= conf_thr
    tp = int(((iou_arr >= iou_thr) & keep).sum())
    fp = int(((iou_arr < iou_thr) & keep).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    return prec, rec


# ---------------- U-Net training ----------------
def train_unet(net, X, y, ftype, Xva, yva, cw, epochs, bs, lr, lam_cls=LAM_CLS, seed=0):
    opt = un.Adam(net, lr=lr); rng = np.random.default_rng(seed); n = len(y)
    best, bestt = -1, 0.5; log = []
    for ep in range(epochs):
        opt.lr = lr * (0.5 ** (ep // 10))
        idx = rng.permutation(n); tot = td = tc = 0.0
        for b in range(0, n, bs):
            bi = idx[b:b + bs]; net.zero_grad()
            lg = net.forward(X[bi])
            ld, dl, _ = un.wce_loss_grad(lg, y[bi], cw)
            if net.n_cls:
                lc, dcls = un.cls_loss_grad(net.cls_logits, ftype[bi], y[bi] == 1)
                net.backward(dl, dcls); loss = ld + lam_cls * lc
                tc += lc * len(bi)
            else:
                net.backward(dl); loss = ld
            opt.step(); tot += loss * len(bi); td += ld * len(bi)
        pv = softmax_p1(net, Xva); f1, th = best_f1(yva, pv)
        if f1 > best: best, bestt = f1, th
        log.append((ep, float(tot / n), float(f1)))
        extra = f" cls={tc/n:.4f}" if net.n_cls else ""
        print(f"   [unet] ep{ep:02d} loss={tot/n:.4f} det={td/n:.4f}{extra} valF1={f1:.3f}")
    return best, bestt, log


def main():
    t0 = time.time()
    print("loading augmented events ...")
    evs = ap.load_all_events()
    train_evs, val_evs, test_evs = ap.split_events(evs, N_TRAIN, N_VAL, N_TEST, seed=SEED)
    print(f"events: train={len(train_evs)} (A/B)  val={len(val_evs)} (A/B)  test={len(test_evs)} (UNSEEN C)")

    # ---- windowed reps for the U-Nets ----
    print("windowing for U-Nets ...")
    tr_w = [ap.windows_for_event(e, STR_TR) for e in train_evs]
    va_w = [ap.windows_for_event(e, STR_TR) for e in val_evs]
    te_w = [ap.windows_for_event(e, STR_EV) for e in test_evs]
    nz = ap.fit_norm(tr_w)

    Uall = np.concatenate([w['U'] for w in tr_w]); Pall = np.concatenate([w['P'] for w in tr_w])
    idx, yall = ap.balance(tr_w, neg_per_pos=6, seed=1)
    ft_all = window_ftypes(tr_w)
    Utr = ap.nU(Uall[idx], nz); Ptr = ap.nP(Pall[idx], nz)
    ytr = yall[idx]; ftr = ft_all[idx]
    Uva = ap.nU(np.concatenate([w['U'] for w in va_w]), nz)
    Pva = ap.nP(np.concatenate([w['P'] for w in va_w]), nz)
    yva = np.concatenate([w['y'] for w in va_w])
    pos = ytr.mean(); cw = np.array([1.0, (1 - pos) / max(pos, 1e-6)], np.float32)
    print(f"train windows={len(ytr)} (pos={int(ytr.sum())}, {pos:.2f})  "
          f"val={len(yva)} (pos={int(yva.sum())})  cw={cw.round(2)}")

    # ---- train the two U-Nets ----
    print("\n== training U-Net-1D (raw waveform) ==")
    net1 = un.UNet('1d', 6, widths=(8, 16, 32, 64), seed=0, n_cls=N_CLS)
    b1, t1, log1 = train_unet(net1, Utr, ytr, ftr, Uva, yva, cw, UNET_EPOCHS, UNET_BS, UNET_LR)
    print("\n== training U-Net-2D (spectrogram) ==")
    net2 = un.UNet('2d', 6, widths=(8, 16, 32, 64), seed=0, n_cls=N_CLS, n_freq=dp.F)
    b2, t2, log2 = train_unet(net2, Ptr, ytr, ftr, Pva, yva, cw, UNET2D_EPOCHS, UNET_BS, UNET_LR,
                              lam_cls=LAM_CLS_2D)
    print(f"\nparams: U-Net-1D={net1.n_params()}  U-Net-2D={net2.n_params()}  [{time.time()-t0:.0f}s]")

    # ---- evaluate on TEST (unseen background C) ----
    print("\n== evaluating on unseen background C ==")
    traces = []
    yte_all = []
    for w in te_w:
        U = ap.nU(w['U'], nz); P = ap.nP(w['P'], nz)
        p1, c1 = predict_unet(net1, U); p2, c2 = predict_unet(net2, P)
        traces.append(dict(mu=w['mu'], p1=p1, p2=p2, c1=c1, c2=c2,
                           s=w['s'], e=w['e'], y=w['y'], ftype=w['ftype']))
        yte_all.append(w['y'])
    yte = np.concatenate(yte_all)
    P1 = np.concatenate([t['p1'] for t in traces]); P2 = np.concatenate([t['p2'] for t in traces])

    n_gt = len(te_w)
    true_cls = np.array([w['ftype'] for w in te_w])

    # ---- per-event detections for each U-Net: (confidence, interval, IoU) ----
    def eval_unet(pkey, ckey, tau):
        E = dict(conf=[], iou=[], inter=[], serr=[], eerr=[], det=[], pcls=[])
        for ti, w in enumerate(te_w):
            p = traces[ti][pkey]; cwin = traces[ti][ckey]
            s, e = w['s'], w['e']
            ok, sh, eh = sustained_interval(w['mu'], p, tau, Lmin=3)
            E['conf'].append(float(p.max()))
            if ok:
                E['iou'].append(iou_samp(sh, eh, s, e)); E['inter'].append((sh, eh))
                E['serr'].append((sh - s) / FS); E['eerr'].append((eh - e) / FS); E['det'].append(True)
                fire = p >= tau
                E['pcls'].append(int(cwin[fire][np.argmax(p[fire])]) if fire.any() else int(cwin[np.argmax(p)]))
            else:
                E['iou'].append(0.0); E['inter'].append(None)
                E['serr'].append(None); E['eerr'].append(None); E['det'].append(False)
                E['pcls'].append(int(cwin[np.argmax(p)]))   # best-guess type even if missed
        E['conf'] = np.array(E['conf']); E['iou'] = np.array(E['iou'])
        E['pcls'] = np.array(E['pcls']); return E

    EV = {'1d': eval_unet('p1', 'c1', t1), '2d': eval_unet('p2', 'c2', t2)}
    OP = {'1d': t1, '2d': t2}
    ORDER = ['1d', '2d']
    NAME = {'1d': 'U-Net-1D', '2d': 'U-Net-2D'}
    NPAR = {'1d': net1.n_params(), '2d': net2.n_params()}

    def metrics(k):
        E = EV[k]; tau = OP[k]
        ap50, _, _ = average_precision(E['conf'], E['iou'], IOU_THR, n_gt)
        mAP = float(np.mean([average_precision(E['conf'], E['iou'], t, n_gt)[0] for t in IOU_SWEEP]))
        aps = {t: average_precision(E['conf'], E['iou'], t, n_gt)[0] for t in AP_TAUS}
        prec, rec = pr_at(E['conf'], E['iou'], IOU_THR, n_gt, tau)
        det = float(np.mean(E['det']))
        mio = float(np.mean([E['iou'][i] for i in range(n_gt) if E['det'][i]] or [0.0]))
        se = [abs(x) for x in E['serr'] if x is not None]
        ee = [abs(x) for x in E['eerr'] if x is not None]
        m = dict(ap50=ap50, mAP=mAP, aps=aps, prec=prec, rec=rec, det=det, iou=mio,
                 serr=(np.mean(se) if se else float('nan')),
                 eerr=(np.mean(ee) if ee else float('nan')), cls_acc=float('nan'))
        tp = (E['conf'] >= OP[k]) & (E['iou'] >= IOU_THR)
        if E['pcls'] is not None:
            m['cls_acc'] = float((E['pcls'][tp] == true_cls[tp]).mean()) if tp.any() else float('nan')
            m['cls_acc_all'] = float((E['pcls'] == true_cls).mean())
            m['n_tp'] = int(tp.sum())
        return m

    M = {k: metrics(k) for k in ORDER}
    a1, a2 = auc(yte, P1), auc(yte, P2)
    f1a, _ = best_f1(yte, P1); f1b, _ = best_f1(yte, P2)
    waucs = {'1d': a1, '2d': a2}; wf1s = {'1d': f1a, '2d': f1b}

    # ---- print table ----
    def row(label, fn):
        print(f"{label:<26}" + "".join(fn(k) for k in ORDER))
    W = 26 + 13 * len(ORDER)
    print("\n" + "=" * W)
    print(f"{'metric (test=unseen bg C)':<26}" + "".join(f"{NAME[k]:>13}" for k in ORDER))
    print("-" * W)
    row('AP@0.5',            lambda k: f"{M[k]['aps'][0.5]:>13.3f}")
    row('AP@0.7',            lambda k: f"{M[k]['aps'][0.7]:>13.3f}")
    row('AP@0.8',            lambda k: f"{M[k]['aps'][0.8]:>13.3f}")
    row('AP@0.9',            lambda k: f"{M[k]['aps'][0.9]:>13.3f}")
    row('mAP@[.5:.95]',      lambda k: f"{M[k]['mAP']:>13.3f}")
    row('precision @op',     lambda k: f"{M[k]['prec']:>13.3f}")
    row('recall @op',        lambda k: f"{M[k]['rec']:>13.3f}")
    row('event detection[%]', lambda k: f"{M[k]['det']*100:>12.0f}%")
    row('interval IoU',      lambda k: f"{M[k]['iou']:>13.3f}")
    row('|start err| [ms]',  lambda k: f"{M[k]['serr']*1000:>13.0f}")
    row('|end err| [ms]',    lambda k: f"{M[k]['eerr']*1000:>13.0f}")
    row('window AUC',        lambda k: f"{waucs[k]:>13.3f}")
    row('window best-F1',    lambda k: f"{wf1s[k]:>13.3f}")
    row('fault-type acc',    lambda k: f"{M[k]['cls_acc']:>13.3f}" if not np.isnan(M[k]['cls_acc']) else f"{M[k]['cls_acc_all']:>13.3f}*")
    row('params',            lambda k: f"{NPAR[k]:>13}")
    print("=" * W)
    for k in ORDER:
        if not np.isnan(M[k]['cls_acc']):
            print(f"{NAME[k]} fault-type accuracy: {M[k]['cls_acc']*100:.0f}% on TP "
                  f"({M[k]['n_tp']}/{n_gt}), {M[k]['cls_acc_all']*100:.0f}% over all events.")

    # ---- U-Net-1D vs U-Net-2D paired comparison ----
    print("\n" + "=" * 60)
    print("U-Net-1D vs U-Net-2D  (positive delta => 2-D better)")
    print("-" * 60)
    print("\nwindow-level:")
    for key, d1, d2 in [('window_auc', waucs['1d'], waucs['2d']),
                        ('window_f1', wf1s['1d'], wf1s['2d'])]:
        print(f"  {key:<14}  1D={d1:.3f}  2D={d2:.3f}  ({d2-d1:+.3f})")
    print("\nevent-level:")
    for key in ['ap50', 'mAP', 'det', 'iou', 'cls_acc']:
        v1, v2 = M['1d'][key], M['2d'][key]
        pct = f" ({(v2-v1)*100:+.0f} pp)" if key in ('det', 'cls_acc') else f" ({v2-v1:+.3f})"
        print(f"  {key:<14}  1D={v1:.3f}  2D={v2:.3f}{pct}")
    print("=" * 60)

    summary = dict(models=ORDER,
                   ap50={k: M[k]['ap50'] for k in ORDER}, mAP={k: M[k]['mAP'] for k in ORDER},
                   ap_by_tau={k: {f"{t:.2f}": M[k]['aps'][t] for t in AP_TAUS} for k in ORDER},
                   precision={k: M[k]['prec'] for k in ORDER}, recall={k: M[k]['rec'] for k in ORDER},
                   detection={k: M[k]['det'] for k in ORDER}, iou={k: M[k]['iou'] for k in ORDER},
                   start_err_ms={k: M[k]['serr']*1000 for k in ORDER},
                   end_err_ms={k: M[k]['eerr']*1000 for k in ORDER},
                   window_auc=waucs, window_f1=wf1s,
                   cls_acc={k: M[k].get('cls_acc') for k in ORDER},
                   cls_acc_all={k: M[k].get('cls_acc_all') for k in ORDER},
                   iou_thr=IOU_THR, op_thr=OP, params=NPAR, n_test=n_gt)
    json.dump(summary, open(os.path.join(PLOTDIR, 'summary.json'), 'w'), indent=2)

    make_plots(log1, log2, traces, te_w, t1, t2, EV, M, n_gt, ORDER, NAME)
    print(f"\nplots + summary written to {PLOTDIR}/   [{time.time()-t0:.0f}s total]")


def make_plots(log1, log2, traces, te_w, t1, t2, EV, M, n_gt, ORDER, NAME):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = {'1d': 'C0', '2d': 'C1'}

    # 1) training curves
    l1, l2 = (np.array(x) for x in (log1, log2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(l1[:, 0], l1[:, 1], label='U-Net-1D'); ax[0].plot(l2[:, 0], l2[:, 1], label='U-Net-2D')
    ax[0].set_title('train loss'); ax[0].set_xlabel('epoch'); ax[0].legend()
    ax[1].plot(l1[:, 0], l1[:, 2], label='U-Net-1D val F1'); ax[1].plot(l2[:, 0], l2[:, 2], label='U-Net-2D val F1')
    ax[1].set_title('validation F1'); ax[1].set_xlabel('epoch'); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, 'training_curves.png'), dpi=120); plt.close(fig)

    # 2) example test events: U-Net prob traces + true fault interval
    n_show = min(4, len(te_w))
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.6 * n_show), squeeze=False)
    for i in range(n_show):
        ax = axes[i, 0]; tr = traces[i]; w = te_w[i]
        ts = tr['mu'] / FS
        ax.plot(ts, tr['p1'], color='C0', lw=1.2, label='U-Net-1D p')
        ax.plot(ts, tr['p2'], color='C1', lw=1.2, label='U-Net-2D p')
        ax.axhline(t1, color='C0', ls=':', lw=.8); ax.axhline(t2, color='C1', ls=':', lw=.8)
        ax.axvspan(w['s'] / FS, w['e'] / FS, color='green', alpha=0.18, label='true fault')
        ax.set_ylim(-0.05, 1.22); ax.set_ylabel('p(fault)')
        ax.set_title(f"test event {i}  true type={w['ftype']}  true=[{w['s']/FS:.2f},{w['e']/FS:.2f}]s")
        if i == 0: ax.legend(loc='upper right', fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel('time [s]')
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, 'test_event_traces.png'), dpi=120); plt.close(fig)

    # 3) precision-recall curves at IoU=0.5
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in ORDER:
        _, rec, prec = average_precision(EV[k]['conf'], EV[k]['iou'], IOU_THR, n_gt)
        ax.plot(rec, prec, '-o', ms=3, color=cols[k], label=f"{NAME[k]} (AP={M[k]['ap50']:.3f})")
    ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title(f'Precision-Recall @ IoU={IOU_THR}'); ax.legend(loc='lower left', fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, 'pr_curves.png'), dpi=120); plt.close(fig)

    # 4) summary bars: AP@0.5, interval IoU, |start err|
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    models = [NAME[k] for k in ORDER]; bar_cols = [cols[k] for k in ORDER]
    ap_vals = [M[k]['ap50'] for k in ORDER]
    iou = [M[k]['iou'] for k in ORDER]
    serr = [M[k]['serr'] * 1000 for k in ORDER]
    for a, vals, ttl in [(ax[0], ap_vals, f'AP @ IoU={IOU_THR}'),
                         (ax[1], iou, 'interval IoU'),
                         (ax[2], serr, '|start err| [ms]')]:
        a.bar(models, vals, color=bar_cols); a.set_title(ttl); a.tick_params(axis='x', labelsize=9)
        for j, v in enumerate(vals):
            a.text(j, v, f'{v:.2f}' if max(vals) < 5 else f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, 'summary_bars.png'), dpi=120); plt.close(fig)

    # 5) AP vs IoU threshold
    taus = np.round(np.arange(0.5, 0.96, 0.05), 2)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k in ORDER:
        aps = [average_precision(EV[k]['conf'], EV[k]['iou'], t, n_gt)[0] for t in taus]
        ax.plot(taus, aps, '-o', ms=3, color=cols[k], label=NAME[k])
    ax.axvline(0.5, color='gray', ls=':', lw=.8)
    ax.set_xlabel('IoU threshold'); ax.set_ylabel('AP'); ax.set_ylim(-0.02, 1.05)
    ax.set_title('AP vs IoU threshold'); ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, 'ap_vs_iou.png'), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
