"""
features/advanced_features.py
All advanced feature groups: L2 OB, tick imbalance, session clocks,
correlation regime, Hurst, options proxies, COT.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
import warnings; warnings.filterwarnings("ignore")

from config.settings import PATHS

def synthetic_orderbook(bars,n_levels=10):
    n=len(bars); rng=np.random.default_rng(42)
    mid=bars["close"].values.astype(float)
    spd=bars["spread_avg"].values if "spread_avg" in bars.columns else np.full(n,5e-5)
    sp=spd.reshape(-1,1)*np.arange(1,n_levels+1)/2
    bv=bars["volume"].values.reshape(-1,1)/n_levels
    dec=1.0/(np.arange(1,n_levels+1)**0.7)
    return {"bid_levels":mid.reshape(-1,1)-sp,"ask_levels":mid.reshape(-1,1)+sp,
            "bid_vols":bv*dec*rng.uniform(0.8,1.2,(n,n_levels)),
            "ask_vols":bv*dec*rng.uniform(0.8,1.2,(n,n_levels))}

def order_book_features(ob,index,k=5):
    F=pd.DataFrame(index=index)
    bv=ob["bid_vols"][:,:k]; av=ob["ask_vols"][:,:k]
    bl=ob["bid_levels"]; al=ob["ask_levels"]
    F["ob_bid_depth"]=bv.sum(1); F["ob_ask_depth"]=av.sum(1)
    tot=F["ob_bid_depth"]+F["ob_ask_depth"]+1e-9
    F["ob_imbalance"]=(F["ob_bid_depth"]-F["ob_ask_depth"])/tot
    F["ob_spread_pip"]=(al[:,0]-bl[:,0])/0.0001
    F["ob_bid_wall"]=(bv>3*bv.mean(1,keepdims=True)).any(1).astype(float)
    F["ob_ask_wall"]=(av>3*av.mean(1,keepdims=True)).any(1).astype(float)
    bv0=ob["bid_vols"][:,0]; av0=ob["ask_vols"][:,0]
    F["microprice"]=(bl[:,0]*av0+al[:,0]*bv0)/(bv0+av0+1e-9)
    return F

def tick_volume_imbalance(bars,window=20):
    F=pd.DataFrame(index=bars.index)
    tick=np.sign(bars["close"].diff()); vol=bars["volume"].astype(float)
    buy=vol*(tick>0).astype(float)+vol*0.5*(tick==0).astype(float)
    sell=vol*(tick<0).astype(float)+vol*0.5*(tick==0).astype(float)
    F["tvi_buy"]=buy.rolling(window).sum(); F["tvi_sell"]=sell.rolling(window).sum()
    tot=F["tvi_buy"]+F["tvi_sell"]+1e-9
    F["tvi_imbalance"]=(F["tvi_buy"]-F["tvi_sell"])/tot
    F["tvi_ratio"]=(F["tvi_buy"]/(F["tvi_sell"]+1e-9)).clip(0,5)
    F["tvi_impact"]=(tick*vol/(vol.rolling(window).mean()+1e-9)).rolling(window).mean()
    return F

def session_features(index):
    F=pd.DataFrame(index=index); h=index.hour; m=index.minute; dec=h+m/60
    F["sess_tokyo"]=((dec>=0)&(dec<9)).astype(float)
    F["sess_london"]=((dec>=7)&(dec<16)).astype(float)
    F["sess_ny"]=((dec>=12)&(dec<21)).astype(float)
    F["sess_sydney"]=((dec>=21)|(dec<6)).astype(float)
    F["sess_ln_ny"]=((dec>=13)&(dec<16)).astype(float)
    F["sess_open"]=(((dec>=7)&(dec<7.25))|((dec>=12)&(dec<12.25))).astype(float)
    tm=h*60+m
    F["tod_sin"]=np.sin(2*np.pi*tm/1440); F["tod_cos"]=np.cos(2*np.pi*tm/1440)
    F["dow_sin"]=np.sin(2*np.pi*index.dayofweek/5); F["dow_cos"]=np.cos(2*np.pi*index.dayofweek/5)
    return F

def correlation_regime_features(returns_df,window=60):
    F=pd.DataFrame(index=returns_df.index)
    n=len(returns_df); nc=returns_df.shape[1]
    avgs=[]; disps=[]; eigrs=[]
    for i in range(n):
        if i<window: avgs.append(np.nan);disps.append(np.nan);eigrs.append(np.nan);continue
        sub=returns_df.iloc[i-window:i].dropna()
        if len(sub)<window//2 or sub.shape[1]<2: avgs.append(np.nan);disps.append(np.nan);eigrs.append(np.nan);continue
        C=sub.corr().values.copy(); np.fill_diagonal(C,0)
        pairs=C[np.triu_indices(nc,k=1)]
        avgs.append(float(np.mean(pairs))); disps.append(float(np.std(pairs)))
        try:
            eigs=np.sort(np.abs(np.linalg.eigvalsh(sub.corr().values)))[::-1]
            eigrs.append(float(eigs[0]/(eigs[1]+1e-9)) if len(eigs)>1 else 1.0)
        except: eigrs.append(np.nan)
    F["corr_avg"]=pd.Series(avgs,index=returns_df.index)
    F["corr_dispersion"]=pd.Series(disps,index=returns_df.index)
    F["corr_eigenratio"]=pd.Series(eigrs,index=returns_df.index)
    rm=F["corr_avg"].rolling(window*3).mean(); rs=F["corr_avg"].rolling(window*3).std()+1e-9
    F["corr_zscore"]=(F["corr_avg"]-rm)/rs; F["corr_break"]=(F["corr_zscore"].abs()>2.0).astype(float)
    return F.ffill().fillna(0)

def hurst_exponent(arr):
    n=len(arr)
    if n<20: return 0.5
    lags=range(4,min(50,n//2)); tau=[]
    for lag in lags:
        sub=arr[-lag*2:]; chunks=[sub[i:i+lag] for i in range(0,len(sub)-lag+1,lag)]; rs_vals=[]
        for c in chunks:
            if len(c)<2: continue
            c=np.array(c,dtype=float); std=c.std()
            if std<1e-12: continue
            dev=np.cumsum(c-c.mean()); rs=(dev.max()-dev.min())/std
            if rs>0: rs_vals.append(rs)
        if rs_vals: tau.append(np.mean(rs_vals))
    if len(tau)<4: return 0.5
    try:
        H,_=np.polyfit(np.log(list(lags)[:len(tau)]),np.log(tau),1)
        return float(np.clip(H,0.1,0.9))
    except: return 0.5

def rolling_hurst(series,window=120,step=20):
    n=len(series); val=np.full(n,np.nan)
    for i in range(window,n,step):
        h=hurst_exponent(series.values[i-window:i]); val[i-step:i]=h
    return pd.Series(val,index=series.index).ffill().fillna(0.5).rename("hurst")

def fast_trend_score(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Fast range-vs-close volatility ratio (Parkinson-style vs close-to-close vol).
    Ratio > ~1.2 suggests expanding ranges vs Gaussian returns (trending proxy).
    """
    low_s = low.astype(float).clip(lower=1e-12)
    high_s = high.astype(float).clip(lower=1e-12)
    close_s = close.astype(float).clip(lower=1e-12)
    log_hl = np.log(high_s / low_s)
    parkinson = np.sqrt(log_hl.pow(2).rolling(window, min_periods=2).mean() / (4 * np.log(2)))
    prev = close_s.shift(1).clip(lower=1e-12)
    cc = np.log(close_s / prev)
    cc_vol = cc.rolling(window, min_periods=2).std()
    ratio = parkinson / (cc_vol + 1e-8)
    return ratio.clip(lower=0.01).fillna(1.0).rename("parkinson_trend_ratio")

def fractal_dimension(series,window=30):
    arr=series.values.astype(float); n=len(arr); fd=np.full(n,np.nan)
    for i in range(window,n):
        sub=arr[i-window:i]; Lm=[]
        for k in range(1,window//2):
            segs=[abs(sub[j+k]-sub[j]) for j in range(0,window-k,k) if j+k<window]
            if segs: Lm.append(np.mean(segs)*(window-1)/(len(segs)*k*k))
        if len(Lm)>2:
            try:
                s,_=np.polyfit(np.log(np.arange(1,len(Lm)+1)),np.log(np.maximum(Lm,1e-12)),1)
                fd[i]=float(np.clip(-s,1.0,2.0))
            except: fd[i]=1.5
    return pd.Series(fd,index=series.index).ffill().fillna(1.5).rename("fractal_dim")

def regime_label(h):
    return pd.Series(np.where(h>0.6,1.0,np.where(h<0.4,-1.0,0.0)),index=h.index,name="regime_label")

def options_proxy_features(bars,window=20):
    F=pd.DataFrame(index=bars.index)
    r=np.log(bars["close"]/bars["close"].shift(1))
    c=np.log(bars["close"]); h=np.log(bars["high"]); l=np.log(bars["low"])
    F["iv_proxy"]=np.sqrt((0.5*(h-l)**2-(2*np.log(2)-1)*(c-c.shift(1))**2).rolling(window).mean())*np.sqrt(252)
    F["skew_proxy"]=r.rolling(window).skew()
    F["term_proxy"]=(r.rolling(5).std()*np.sqrt(252)/(r.rolling(60).std()*np.sqrt(252)+1e-9)).clip(0.3,3.0)
    up=r[r>0].rolling(window,min_periods=5).std()*np.sqrt(252)
    dn=(-r[r<0]).rolling(window,min_periods=5).std()*np.sqrt(252)
    rr=(up-dn.reindex(bars.index).ffill()).fillna(0)
    F["risk_reversal"]=rr.clip(-0.05,0.05)/0.05
    return F.ffill().fillna(0)

def cot_features(index,cot_data=None):
    F=pd.DataFrame(index=index)
    if cot_data is None:
        rng=np.random.default_rng(11); n=len(index)
        raw=pd.Series(np.cumsum(rng.normal(0,0.02,n)),index=index).ewm(span=500).mean()
        F["cot_net"]=np.tanh(raw); F["cot_noncom_net"]=np.tanh(raw*0.8)
        F["cot_extreme"]=(F["cot_net"].abs()>0.7).astype(float)
        F["cot_change"]=F["cot_net"].diff(7).fillna(0); return F
    cot=cot_data.copy(); cot.index=pd.to_datetime(cot.index,utc=True)
    cot=cot.reindex(index,method="ffill").ffill().bfill()
    ln=cot.get("long_noncom",pd.Series(0,index=index)); sn=cot.get("short_noncom",pd.Series(0,index=index))
    F["cot_net"]=(ln-sn)/(ln+sn+1e-9); F["cot_noncom_net"]=F["cot_net"]
    F["cot_extreme"]=(F["cot_net"].abs()>0.7).astype(float); F["cot_change"]=F["cot_net"].diff(7).fillna(0)
    return F

class AdvancedFeatureEngineer:
    def __init__(self,hurst_window=120,hurst_step=20,corr_window=60,tvi_window=20,options_window=20):
        self.hw=hurst_window;self.hs=hurst_step;self.cw=corr_window;self.tw=tvi_window;self.ow=options_window

    def build(self,bars,base_features,cot_data=None):
        idx=base_features.index; bars_a=bars.reindex(idx).ffill()
        F=pd.DataFrame(index=idx)
        ob=synthetic_orderbook(bars_a); F=pd.concat([F,order_book_features(ob,idx)],axis=1)
        F=pd.concat([F,tick_volume_imbalance(bars_a,self.tw)],axis=1)
        F=pd.concat([F,session_features(idx)],axis=1)
        ret_cols=[c for c in base_features.columns if c.endswith("_ret")]
        if ret_cols: F=pd.concat([F,correlation_regime_features(base_features[ret_cols],self.cw)],axis=1)
        close=bars_a["close"]
        tr = fast_trend_score(bars_a["high"], bars_a["low"], close,
                              window=max(10, min(self.hw, 50)))
        H = (0.5 + 0.25 * np.tanh(tr - 1.0)).rename("hurst")
        fd = fractal_dimension(close, 30)
        rl = regime_label(H)
        F = pd.concat([F, tr.to_frame(), H.to_frame(), fd.to_frame(), rl.to_frame()], axis=1)
        F=pd.concat([F,options_proxy_features(bars_a,self.ow)],axis=1)
        F=pd.concat([F,cot_features(idx,cot_data)],axis=1)
        F=F.replace([np.inf,-np.inf],np.nan).ffill().fillna(0)
        print(f"[AdvFeatures] +{F.shape[1]} cols")
        return F
AdvancedFeatureBuilder = AdvancedFeatureEngineer

# ── Compatibility wrappers for main.py ──────────────────────────────────

class L2OrderBookFeatures:
    def __init__(self, n_levels=10, lookback=20):
        self.n = n_levels
    def from_bars(self, bars):
        ob = synthetic_orderbook(bars, self.n)
        return order_book_features(ob, bars.index, k=min(5, self.n))

def session_clock_features(index):
    return session_features(index)

class CorrelationRegimeDetector:
    def __init__(self, window=60, break_thresh=0.3):
        self.w = window; self.t = break_thresh
    def build(self, returns_df):
        return correlation_regime_features(returns_df, self.w)

def rolling_hurst_fractal(bars, windows=[30, 60, 120]):
    import pandas as pd
    log_ret = (bars['close'] / bars['close'].shift(1)).apply(lambda x: 0 if x<=0 else __import__('math').log(x)).fillna(0) if hasattr(bars['close'], 'apply') else pd.Series(0, index=bars.index)
    log_ret = pd.Series(__import__('numpy').log(bars['close'].values / bars['close'].shift(1).fillna(method='bfill').values), index=bars.index).fillna(0)
    df = pd.DataFrame(index=bars.index)
    for w in windows:
        h = rolling_hurst(log_ret, window=w, step=1)
        fd = fractal_dimension(log_ret, window=w)
        df[f'hurst_{w}']   = h.reindex(bars.index).ffill().fillna(0.5)
        df[f'fractal_{w}'] = fd.reindex(bars.index).ffill().fillna(1.5)
    mid_w = windows[min(1, len(windows)-1)]
    df['trending']       = (df[f'hurst_{mid_w}'] > 0.55).astype(float)
    df['mean_reverting'] = (df[f'hurst_{mid_w}'] < 0.45).astype(float)
    return df

class OptionsSkewFeatures:
    def __init__(self, windows=[5, 20, 60]):
        self.w = windows[-1] if windows else 20
    def build_synthetic(self, bars):
        return options_proxy_features(bars, window=self.w)

class COTFeatures:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or PATHS["data_raw_cot"]
    def build_synthetic(self, index):
        return cot_features(index)

class AdvancedFeatureBuilder:
    """Flexible wrapper — can be called as afb.build(bars) or afb.build(bars, feats)."""
    def __init__(self, hurst_windows=[30,60,120], **kw):
        self._inner = AdvancedFeatureEngineer(hurst_window=hurst_windows[-1] if hurst_windows else 60)
    def build(self, bars, base_features=None, **kw):
        if base_features is None:
            # Build a minimal base_features from bars
            from features.feature_engineering import FeatureEngineer
            fe = FeatureEngineer(atr_window=6, lag_windows=[5,20,60])
            base_features = fe.build(bars)
        return self._inner.build(bars, base_features)
