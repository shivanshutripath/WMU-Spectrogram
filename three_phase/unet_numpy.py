"""
Step 2 - Compact U-Nets in pure NumPy (no DL framework available offline).

Architecture mirrors the paper: convolutional encoder-decoder with skip
connections, channel widths doubling with depth, downsampling along the time
(W) axis, a global pooling layer, and a dense+softmax head -> p(fault).
Conv1D variant reads U_k (raw waveform); Conv2D variant reads P_k (spectrogram).
Both share channel widths so any gap is due to representation, not capacity.
Trained with class-weighted cross-entropy + Adam.

** Multi-task add-on **: pass n_cls>0 to UNet to attach a SECOND dense head that
predicts the fault TYPE (0..n_cls-1) from the same pooled feature. The original
2-way fault/no-fault head is untouched, so detection/localization is unchanged.
The type head is trained only on fault windows (see cls_loss_grad).
"""
import numpy as np

# ---------- layers (each: forward stores cache; backward returns dx, fills grads) ----------
class Conv1D:
    def __init__(s,k,ci,co,seed):
        rng=np.random.default_rng(seed)
        s.W=(rng.standard_normal((k,ci,co))*np.sqrt(2/(k*ci))).astype(np.float32)
        s.b=np.zeros(co,np.float32); s.k=k
    def params(s): return [('W',s.W),('b',s.b)]
    def setgrad(s): s.dW=np.zeros_like(s.W); s.db=np.zeros_like(s.b)
    def forward(s,x):               # x:(B,T,Ci)
        B,T,Ci=x.shape; k=s.k; p=k//2
        xp=np.pad(x,((0,0),(p,p),(0,0)))
        cols=np.stack([xp[:,i:i+T,:] for i in range(k)],axis=2)  # (B,T,k,Ci)
        s.cols=cols.reshape(B,T,k*Ci); s.shape=x.shape
        return s.cols@s.W.reshape(k*Ci,-1)+s.b                   # (B,T,Co)
    def backward(s,dy):
        B,T,Ci=s.shape; k=s.k; p=k//2
        Wm=s.W.reshape(k*Ci,-1)
        s.dW+=(s.cols.reshape(-1,k*Ci).T@dy.reshape(-1,dy.shape[-1])).reshape(s.W.shape)
        s.db+=dy.sum((0,1))
        dcols=(dy@Wm.T).reshape(B,T,k,Ci)
        dxp=np.zeros((B,T+2*p,Ci),np.float32)
        for i in range(k): dxp[:,i:i+T,:]+=dcols[:,:,i,:]
        return dxp[:,p:p+T,:]

class Conv2D:
    def __init__(s,kt,kf,ci,co,seed):
        rng=np.random.default_rng(seed)
        s.W=(rng.standard_normal((kt,kf,ci,co))*np.sqrt(2/(kt*kf*ci))).astype(np.float32)
        s.b=np.zeros(co,np.float32); s.kt=kt; s.kf=kf
    def params(s): return [('W',s.W),('b',s.b)]
    def setgrad(s): s.dW=np.zeros_like(s.W); s.db=np.zeros_like(s.b)
    def forward(s,x):               # x:(B,T,Fq,Ci)
        B,T,Fq,Ci=x.shape; kt,kf=s.kt,s.kf; pt,pf=kt//2,kf//2
        xp=np.pad(x,((0,0),(pt,pt),(pf,pf),(0,0)))
        cols=np.stack([xp[:,i:i+T,j:j+Fq,:] for i in range(kt) for j in range(kf)],axis=3)
        s.cols=cols.reshape(B,T,Fq,kt*kf*Ci); s.shape=x.shape
        return s.cols@s.W.reshape(kt*kf*Ci,-1)+s.b               # (B,T,Fq,Co)
    def backward(s,dy):
        B,T,Fq,Ci=s.shape; kt,kf=s.kt,s.kf; pt,pf=kt//2,kf//2
        Wm=s.W.reshape(kt*kf*Ci,-1)
        s.dW+=(s.cols.reshape(-1,kt*kf*Ci).T@dy.reshape(-1,dy.shape[-1])).reshape(s.W.shape)
        s.db+=dy.sum((0,1,2))
        dcols=(dy@Wm.T).reshape(B,T,Fq,kt*kf,Ci)
        dxp=np.zeros((B,T+2*pt,Fq+2*pf,Ci),np.float32)
        idx=0
        for i in range(kt):
            for j in range(kf):
                dxp[:,i:i+T,j:j+Fq,:]+=dcols[:,:,:,idx,:]; idx+=1
        return dxp[:,pt:pt+T,pf:pf+Fq,:]

class ReLU:
    def forward(s,x): s.m=(x>0); return x*s.m
    def backward(s,dy): return dy*s.m

class AvgPoolT:  # pool factor 2 along time axis=1 (works for 1D & 2D)
    def forward(s,x):
        s.sh=x.shape; T=x.shape[1]
        if x.ndim==3: y=x[:,:T-T%2:1,:].reshape(x.shape[0],T//2,2,x.shape[2]).mean(2)
        else: y=x[:,:T-T%2:1,:,:].reshape(x.shape[0],T//2,2,x.shape[2],x.shape[3]).mean(2)
        return y
    def backward(s,dy):
        dx=np.zeros(s.sh,np.float32)
        if dy.ndim==3: dx[:,:dy.shape[1]*2,:]=np.repeat(dy,2,axis=1)/2.0
        else: dx[:,:dy.shape[1]*2,:,:]=np.repeat(dy,2,axis=1)/2.0
        return dx

class UpT:       # nearest-neighbour upsample factor 2 along time
    def forward(s,x): s.nd=x.ndim; return np.repeat(x,2,axis=1)
    def backward(s,dy):
        if dy.ndim==3: return (dy[:,0::2,:]+dy[:,1::2,:])
        return (dy[:,0::2,:,:]+dy[:,1::2,:,:])

class Dense:
    def __init__(s,ci,co,seed):
        rng=np.random.default_rng(seed)
        s.W=(rng.standard_normal((ci,co))*np.sqrt(2/ci)).astype(np.float32); s.b=np.zeros(co,np.float32)
    def params(s): return [('W',s.W),('b',s.b)]
    def setgrad(s): s.dW=np.zeros_like(s.W); s.db=np.zeros_like(s.b)
    def forward(s,x): s.x=x; return x@s.W+s.b
    def backward(s,dy): s.dW+=s.x.T@dy; s.db+=dy.sum(0); return dy@s.W.T

def crop_time(x,T):  # center/trim along time axis to length T (for skip concat safety)
    return x[:, :T] if x.shape[1]>=T else np.pad(x,[(0,0),(0,T-x.shape[1])]+[(0,0)]*(x.ndim-2))

# ---------------- U-Net assembly (exact power-of-two sizes; no crop needed) ----------------
class UNet:
    """3-down/3-up U-Net. mode '1d' -> Conv1D on U_k; '2d' -> Conv2D on P_k.
       n_cls>0 attaches a parallel fault-TYPE head (multi-task); n_cls=0 -> binary only."""
    def __init__(s, mode, in_ch, widths=(8,16,32,64), seed=0, n_cls=0, n_freq=1):
        s.mode=mode
        s.n_cls=int(n_cls)
        s.n_freq=int(n_freq)               # F bins (2d type head keeps frequency, see forward)
        c0,c1,c2,c3=widths
        C=(lambda ci,co,sd:(Conv1D(5,ci,co,sd) if mode=='1d' else Conv2D(3,3,ci,co,sd)))
        s.e0=C(in_ch,c0,seed+1); s.e1=C(c0,c1,seed+2); s.e2=C(c1,c2,seed+3); s.bn=C(c2,c3,seed+4)
        s.d2=C(c3+c2,c2,seed+5); s.d1=C(c2+c1,c1,seed+6); s.d0=C(c1+c0,c0,seed+7)
        s.head=Dense(c0,2,seed+8)                       # fault / no-fault (unchanged)
        s.relu=[ReLU() for _ in range(7)]
        s.pool=[AvgPoolT() for _ in range(3)]; s.up=[UpT() for _ in range(3)]
        s.layers=[s.e0,s.e1,s.e2,s.bn,s.d2,s.d1,s.d0,s.head]
        if s.n_cls:                                     # NEW: fault-type head
            # 2d: keep frequency (pool time only) -> c0*F features, so the type head
            # can read frequency-specific signatures; 1d has no freq axis (c0).
            cls_in=c0*(s.n_freq if mode=='2d' else 1)
            s.head_cls=Dense(cls_in,s.n_cls,seed+9)
            s.layers.append(s.head_cls)                 # appended last -> keeps existing indices
        s.cls_logits=None
        for l in s.layers: l.setgrad()
    def zero_grad(s):
        for l in s.layers: l.setgrad()
    def n_params(s): return int(sum(p.size for l in s.layers for _,p in l.params()))
    def forward(s,x):
        h0=s.relu[0].forward(s.e0.forward(x)); p0=s.pool[0].forward(h0)
        h1=s.relu[1].forward(s.e1.forward(p0)); p1=s.pool[1].forward(h1)
        h2=s.relu[2].forward(s.e2.forward(p1)); p2=s.pool[2].forward(h2)
        b =s.relu[3].forward(s.bn.forward(p2))
        u2=s.up[0].forward(b)
        d2=s.relu[4].forward(s.d2.forward(np.concatenate([u2,h2],axis=-1)))
        u1=s.up[1].forward(d2)
        d1=s.relu[5].forward(s.d1.forward(np.concatenate([u1,h1],axis=-1)))
        u0=s.up[2].forward(d1)
        d0=s.relu[6].forward(s.d0.forward(np.concatenate([u0,h0],axis=-1)))
        s.gp_ax=tuple(range(1,d0.ndim-1)); s.gp_shape=d0.shape
        s.cw=(h2.shape[-1],h1.shape[-1],h0.shape[-1])
        g=d0.mean(axis=s.gp_ax)                          # (B,c0) pooled feature (binary head)
        if s.n_cls:
            if s.mode=='2d':
                gtype=d0.mean(axis=1).reshape(d0.shape[0],-1)  # pool time only -> (B,F*c0)
            else:
                gtype=g
            s.cls_logits=s.head_cls.forward(gtype)       # (B,n_cls) type logits
        return s.head.forward(g)                          # logits (B,2)  <- same return as before
    def backward(s,dlogits,dcls=None):
        dg=s.head.backward(dlogits)
        dd0_cls = None
        if dcls is not None and s.n_cls:
            dgt = s.head_cls.backward(dcls)
            if s.mode == '2d':                           # cls grad helps 2-D trunk learn spectral type cues
                B, T, Fq, C = s.gp_shape
                dgt = dgt.reshape(B, Fq, C)
                dd0_cls = np.broadcast_to(dgt[:, None, :, :], s.gp_shape).astype(np.float32) / T
            # 1d: decoupled — cls updates head_cls only (avoids hurting sample-level detection)
        n=float(np.prod([s.gp_shape[a] for a in s.gp_ax]))
        dd0=np.broadcast_to(dg.reshape((dg.shape[0],)+(1,)*len(s.gp_ax)+(dg.shape[1],)),
                            s.gp_shape).astype(np.float32)/n
        if dd0_cls is not None:
            dd0 = dd0 + dd0_cls
        dd0=s.relu[6].backward(dd0); dcat0=s.d0.backward(dd0)
        cw2,cw1,cw0=s.cw
        du0=dcat0[...,:-cw0]; dh0=dcat0[...,-cw0:].copy()
        dd1=s.relu[5].backward(s.up[2].backward(du0)); dcat1=s.d1.backward(dd1)
        du1=dcat1[...,:-cw1]; dh1=dcat1[...,-cw1:].copy()
        dd2=s.relu[4].backward(s.up[1].backward(du1)); dcat2=s.d2.backward(dd2)
        du2=dcat2[...,:-cw2]; dh2=dcat2[...,-cw2:].copy()
        db=s.relu[3].backward(s.up[0].backward(du2)); dp2=s.bn.backward(db)
        dh2=dh2+s.pool[2].backward(dp2)
        de2=s.relu[2].backward(dh2); dp1=s.e2.backward(de2)
        dh1=dh1+s.pool[1].backward(dp1)
        de1=s.relu[1].backward(dh1); dp0=s.e1.backward(de1)
        dh0=dh0+s.pool[0].backward(dp0)
        de0=s.relu[0].backward(dh0); s.e0.backward(de0)

# ---------------- loss + Adam ----------------
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def wce_loss_grad(logits,y,w):
    """class-weighted cross-entropy. y:(B,) in {0,1}; w:(2,) class weights.
    returns loss, dlogits."""
    p=softmax(logits); B=len(y)
    wv=w[y]
    loss=-(wv*np.log(p[np.arange(B),y]+1e-9)).sum()/wv.sum()
    onehot=np.zeros_like(p); onehot[np.arange(B),y]=1
    dlog=(p-onehot)*wv[:,None]/wv.sum()
    return loss,dlog.astype(np.float32),p[:,1]

def cls_loss_grad(logits,ftype,mask):
    """Fault-TYPE cross-entropy applied ONLY to fault windows (mask==True).
       logits:(B,n_cls); ftype:(B,) int in [0,n_cls); mask:(B,) bool.
       Returns (loss, dlogits) with dlogits=0 on non-fault rows. Averaged over
       the number of fault windows so it is scale-comparable to wce_loss_grad."""
    p=softmax(logits); B=len(ftype)
    m=mask.astype(bool); nb=max(int(m.sum()),1)
    onehot=np.zeros_like(p); onehot[np.arange(B),ftype]=1.0
    dlog=np.zeros_like(p)
    dlog[m]=(p[m]-onehot[m])/nb
    loss=float(-(np.log(p[np.arange(B),ftype][m]+1e-9)).sum()/nb)
    return loss,dlog.astype(np.float32)

class Adam:
    def __init__(s,net,lr=2e-3,b1=0.9,b2=0.999,eps=1e-8):
        s.net=net; s.lr=lr; s.b1=b1; s.b2=b2; s.eps=eps; s.t=0
        s.m={}; s.v={}
        for li,l in enumerate(net.layers):
            for pn,p in l.params():
                s.m[(li,pn)]=np.zeros_like(p); s.v[(li,pn)]=np.zeros_like(p)
    def step(s):
        s.t+=1
        for li,l in enumerate(s.net.layers):
            grads={'W':getattr(l,'dW',None),'b':getattr(l,'db',None)}
            for pn,p in l.params():
                g=grads[pn]; key=(li,pn)
                s.m[key]=s.b1*s.m[key]+(1-s.b1)*g
                s.v[key]=s.b2*s.v[key]+(1-s.b2)*(g*g)
                mh=s.m[key]/(1-s.b1**s.t); vh=s.v[key]/(1-s.b2**s.t)
                p-=s.lr*mh/(np.sqrt(vh)+s.eps)