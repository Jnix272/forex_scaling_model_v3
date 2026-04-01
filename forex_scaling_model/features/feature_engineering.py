"""
features/feature_engineering.py  (v2)
All microstructure + momentum + cross-asset + sentiment features.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

# ── Microstructure ────────────────────────────────────────────────────────────
def order_flow_imbalance(df, window=20):
    d = np.sign(df["close"] - df["open"])
    bv = df["volume"] * d.clip(lower=0)
    sv = df["volume"] * (-d).clip(lower=0)
    return ((bv - sv) / df["volume"].replace(0,np.nan)).rolling(window).mean().rename("ofi")

def order_book_imbalance_proxy(df):
    r = df["high"] - df["low"] + 1e-9
    return ((df["close"] - df["low"]) / r).clip(0,1).rename("obi_proxy")

def trade_arrival_rate(df, window=30):
    if "n_ticks" in df.columns:
        rate = df["n_ticks"].astype(float)
    else:
        vol = df["volume"].astype(float)
        rate = vol / (vol.rolling(window).mean() + 1e-9)
    mu = rate.rolling(window).mean(); s = rate.rolling(window).std() + 1e-9
    return ((rate - mu) / s).rename("tar")

# ── ATR-6 + Volatility ────────────────────────────────────────────────────────
def average_true_range(df, window=6):
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],(df["high"]-prev).abs(),(df["low"]-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(window).mean().rename(f"atr_{window}")

def rolling_volatility(df, window=20):
    return np.log(df["close"]/df["close"].shift(1)).rolling(window).std().rename(f"vol_{window}")

def bollinger_bands(df, window=20, n_std=2.0):
    mid=df["close"].rolling(window).mean(); std=df["close"].rolling(window).std()
    up=mid+n_std*std; lo=mid-n_std*std
    return pd.DataFrame({"bb_mid":mid,"bb_upper":up,"bb_lower":lo,
        "bb_width":(up-lo)/(mid+1e-9),"bb_pct":((df["close"]-lo)/(up-lo+1e-9)).clip(0,1)})

# ── Momentum ─────────────────────────────────────────────────────────────────
def rsi(s, period=14):
    d=s.diff(); g=d.clip(lower=0).rolling(period).mean(); l=(-d.clip(upper=0)).rolling(period).mean()+1e-9
    return (100-100/(1+g/l)).rename(f"rsi_{period}")

def macd(s, fast=12, slow=26, signal=9):
    ef=s.ewm(span=fast,adjust=False).mean(); es=s.ewm(span=slow,adjust=False).mean()
    line=ef-es; sig=line.ewm(span=signal,adjust=False).mean()
    return pd.DataFrame({"macd":line,"macd_sig":sig,"macd_hist":line-sig})

def lag_returns(df, windows=[5,20,60]):
    lp=np.log(df["close"])
    return pd.DataFrame({f"ret_{w}":lp-lp.shift(w) for w in windows})

# ── Cross-Asset ───────────────────────────────────────────────────────────────
class CrossAssetFeatures:
    # Broader intermarket universe used for cross-currency leadership signals.
    ASSETS = {
        "DXY": {"base": 103.0, "corr": -0.55, "noise": 0.004},
        "US10Y": {"base": 4.5, "corr": 0.35, "noise": 0.001},
        "DE10Y": {"base": 2.1, "corr": -0.15, "noise": 0.001},
        "US2Y": {"base": 4.2, "corr": 0.25, "noise": 0.001},
        "WTI": {"base": 72.0, "corr": -0.30, "noise": 0.008},
        "GOLD": {"base": 2200.0, "corr": -0.40, "noise": 0.010},
        "COPPER": {"base": 3.8, "corr": 0.50, "noise": 0.006},
        "IRON_ORE": {"base": 130.0, "corr": 0.45, "noise": 0.007},
        "SPX": {"base": 5000.0, "corr": 0.30, "noise": 0.007},
        "NASDAQ100": {"base": 17500.0, "corr": 0.35, "noise": 0.009},
        "VIX": {"base": 18.0, "corr": -0.65, "noise": 0.015},
        "BTC": {"base": 70000.0, "corr": 0.15, "noise": 0.020},
    }

    def __init__(self, corr_window=60, regime_window=240, lags=(1, 5, 15)):
        self.cw = corr_window
        self.rw = regime_window
        self.lags = tuple(lags)

    def build(self, bars, data=None):
        synthetic = self._synthetic(bars)
        if data is None:
            data = synthetic
        else:
            # Keep feature dimensionality stable: use live series when present,
            # synthetic fallback for any missing assets.
            merged = dict(synthetic)
            merged.update(data)
            data = merged
        F = pd.DataFrame(index=bars.index)
        forex_ret = np.log(bars["close"] / bars["close"].shift(1))
        min_corr_periods = max(5, int(self.cw // 4))
        min_regime_periods = max(5, int(self.rw // 4))
        aligned = {}
        for asset, s in data.items():
            al = s.reindex(bars.index, method="ffill").ffill().bfill()
            aligned[asset] = al
            lr = np.log(al / al.shift(1))
            F[f"{asset}_ret"] = lr
            for lag in self.lags:
                F[f"{asset}_ret_l{lag}"] = lr.shift(lag)
            F[f"{asset}_corr"] = lr.rolling(self.cw, min_periods=min_corr_periods).corr(forex_ret)
            F[f"{asset}_beta"] = (
                lr.rolling(self.cw, min_periods=min_corr_periods).cov(forex_ret)
                / (forex_ret.rolling(self.cw, min_periods=min_corr_periods).var() + 1e-9)
            )

        # Yield spread predictors for USD/EUR pressure and curve regime.
        if "US10Y" in aligned and "DE10Y" in aligned:
            F["yield_spread_us_de_10y"] = aligned["US10Y"] - aligned["DE10Y"]
            F["yield_spread_us_de_10y_chg"] = F["yield_spread_us_de_10y"].diff()
        if "US10Y" in aligned and "US2Y" in aligned:
            F["us_2s10s_spread"] = aligned["US10Y"] - aligned["US2Y"]
            F["us_2s10s_spread_chg"] = F["us_2s10s_spread"].diff()

        # Safe-haven and correlation-breakdown diagnostics.
        if "SPX" in aligned and "VIX" in aligned:
            spx_ret = np.log(aligned["SPX"] / aligned["SPX"].shift(1))
            vix_ret = np.log(aligned["VIX"] / aligned["VIX"].shift(1))
            F["risk_off_signal"] = ((spx_ret < 0).astype(float) + (vix_ret > 0).astype(float)) / 2.0
        if "GOLD_ret" in F.columns and "DXY_ret" in F.columns:
            gd_corr = F["GOLD_ret"].rolling(self.cw, min_periods=min_corr_periods).corr(F["DXY_ret"])
            gd_corr_base = gd_corr.rolling(self.rw, min_periods=min_regime_periods).mean()
            F["gold_dxy_corr"] = gd_corr
            F["gold_dxy_corr_break"] = (gd_corr - gd_corr_base).abs()
        if "WTI_ret" in F.columns and "COPPER_ret" in F.columns:
            F["commodity_fx_lead"] = 0.6 * F["COPPER_ret"].shift(1) + 0.4 * F["WTI_ret"].shift(1)

        # If a column is entirely NaN on short samples, keep pipeline alive.
        all_nan_cols = F.columns[F.isna().all()].tolist()
        if all_nan_cols:
            F[all_nan_cols] = 0.0
        return F.ffill().bfill()

    def _synthetic(self, bars):
        rng = np.random.default_rng(99)
        n = len(bars)
        br = np.log(bars["close"] / bars["close"].shift(1)).fillna(0)
        out = {}
        for a, c in self.ASSETS.items():
            lr = br.values * c["corr"] + rng.normal(0, c["noise"], n)
            out[a] = pd.Series(c["base"] * np.exp(np.cumsum(lr)), index=bars.index)
        return out

# ── Sentiment ─────────────────────────────────────────────────────────────────
def sentiment_decay(s, lam=0.1):
    dec=pd.Series(np.zeros(len(s)),index=s.index,dtype=float)
    lt=None; lv=0.0
    for ts,v in s.items():
        if pd.notna(v) and v!=0: lt=ts; lv=v; dec[ts]=v
        elif lt: dec[ts]=lv*np.exp(-lam*(ts-lt).total_seconds())
    return dec.rename("sentiment_decayed")

def buzz_score(counts, window=5): return counts.rolling(window).sum().fillna(0).rename("buzz")
def eco_surprise(actual, forecast): return (actual-forecast).rename("eco_surprise")

def proj_finbert(emb, dim=8):
    rng=np.random.default_rng(0)
    P=rng.standard_normal((768,dim)).astype(np.float32)
    P/=np.linalg.norm(P,axis=0,keepdims=True)+1e-9
    e=emb.reshape(1,-1) if emb.ndim==1 else emb
    return (e@P).squeeze()

# ── Filters ───────────────────────────────────────────────────────────────────
def vol_filter(F, atr_col="atr_6", mult=3.0, lb=60):
    return (F[atr_col]<=mult*F[atr_col].rolling(lb).mean()).astype(float).rename("vol_ok")

def news_filter(idx, events=None, buf_min=15):
    flags=pd.Series(1.0,index=idx,name="news_ok"); buf=pd.Timedelta(minutes=buf_min)
    if events is None:
        rng=np.random.default_rng(7)
        days=pd.date_range(idx[0].date(),idx[-1].date(),freq="D"); events=[]
        for d in days:
            for off in rng.integers(8*60,18*60,2):
                events.append(pd.Timestamp(d,tz="UTC")+pd.Timedelta(minutes=int(off)))
    for ev in events:
        flags[(idx>=ev-buf)&(idx<=ev+buf)]=0.0
    return flags


# ── Regime Gating ─────────────────────────────────────────────────────────────
class RegimeGateClassifier:
    """
    Lightweight regime detector:
      - target=1 means "traditional macro links are unstable / broken"
      - output is a smooth probability gate usable by downstream models
    """

    def __init__(self, min_samples: int = 80, random_state: int = 42):
        self.min_samples = int(min_samples)
        self.random_state = int(random_state)
        self.model = LogisticRegression(max_iter=500, random_state=self.random_state)

    @staticmethod
    def _zscore(s: pd.Series, lb: int = 60) -> pd.Series:
        mu = s.rolling(lb, min_periods=max(10, lb // 4)).mean()
        sd = s.rolling(lb, min_periods=max(10, lb // 4)).std() + 1e-9
        return (s - mu) / sd

    def fit_predict(self, F: pd.DataFrame) -> pd.Series:
        req = ["gold_dxy_corr_break", "us_2s10s_spread_chg", "yield_spread_us_de_10y_chg", "risk_off_signal"]
        if any(c not in F.columns for c in req):
            return pd.Series(0.0, index=F.index, name="regime_break_prob")

        x1 = self._zscore(F["gold_dxy_corr_break"]).clip(-6, 6)
        x2 = self._zscore(F["us_2s10s_spread_chg"]).clip(-6, 6)
        x3 = self._zscore(F["yield_spread_us_de_10y_chg"]).clip(-6, 6)
        x4 = self._zscore(F["risk_off_signal"], lb=30).clip(-6, 6)
        X = pd.concat([x1, x2, x3, x4], axis=1).replace([np.inf, -np.inf], np.nan)
        X.columns = ["gold_break_z", "curve_chg_z", "yield_chg_z", "risk_off_z"]

        # Pseudo-label "break" regimes when correlation breaks and risk stress co-occur.
        y = ((X["gold_break_z"] > 1.0) & ((X["risk_off_z"] > 0.5) | (X["curve_chg_z"].abs() > 1.0))).astype(int)
        ok = X.notna().all(axis=1)
        if ok.sum() < self.min_samples or y[ok].nunique() < 2:
            score = (0.8 * X["gold_break_z"].fillna(0.0)
                     + 0.5 * X["risk_off_z"].fillna(0.0)
                     + 0.3 * X["curve_chg_z"].abs().fillna(0.0))
            prob = 1.0 / (1.0 + np.exp(-score.clip(-8, 8)))
            return pd.Series(prob, index=F.index, name="regime_break_prob").astype(float)

        self.model.fit(X.loc[ok].values, y.loc[ok].values)
        prob = np.full(len(F), 0.0, dtype=np.float64)
        prob[ok.values] = self.model.predict_proba(X.loc[ok].values)[:, 1]
        return pd.Series(prob, index=F.index, name="regime_break_prob").astype(float)

# ── Master Builder ────────────────────────────────────────────────────────────
class FeatureEngineer:
    def __init__(self,atr_window=6,ofi_window=20,tar_window=30,rsi_period=14,
                 macd_fast=12,macd_slow=26,macd_signal=9,bb_window=20,bb_std=2.0,
                 lag_windows=[5,20,60],vol_mult=3.0,news_buf=15,decay_lam=0.1,fb_dim=8,
                 ca_corr_window=60,ca_regime_window=240,ca_lags=(1,5,15),
                 enable_regime_gate=True):
        self.atr_w=atr_window; self.ofi_w=ofi_window; self.tar_w=tar_window
        self.rsi_p=rsi_period; self.mf=macd_fast; self.ms=macd_slow; self.msig=macd_signal
        self.bb_w=bb_window; self.bb_s=bb_std; self.lags=lag_windows
        self.vm=vol_mult; self.nb=news_buf; self.dl=decay_lam; self.fb=fb_dim
        self.ca=CrossAssetFeatures(
            corr_window=ca_corr_window,
            regime_window=ca_regime_window,
            lags=ca_lags,
        )
        self.enable_regime_gate = bool(enable_regime_gate)
        self.regime_gate = RegimeGateClassifier()

    def build(self,bars,cross_asset=None,sentiment=None,eco_act=None,eco_fc=None,
              art_counts=None,finbert_embs=None,news_events=None):
        F=pd.DataFrame(index=bars.index)
        # microstructure
        F["ofi"]=order_flow_imbalance(bars,self.ofi_w)
        F["obi_proxy"]=order_book_imbalance_proxy(bars)
        F["tar"]=trade_arrival_rate(bars,self.tar_w)
        # atr-6 + vol
        ac=f"atr_{self.atr_w}"
        F[ac]=average_true_range(bars,self.atr_w)
        F["vol_20"]=rolling_volatility(bars,20)
        F=pd.concat([F,bollinger_bands(bars,self.bb_w,self.bb_s)],axis=1)
        # momentum
        F[f"rsi_{self.rsi_p}"]=rsi(bars["close"],self.rsi_p)
        F=pd.concat([F,macd(bars["close"],self.mf,self.ms,self.msig)],axis=1)
        F=pd.concat([F,lag_returns(bars,self.lags)],axis=1)
        # spread
        if "spread_avg" in bars.columns: F["spread_pips"]=bars["spread_avg"]/0.0001
        elif "ask_close" in bars.columns: F["spread_pips"]=(bars["ask_close"]-bars["bid_close"])/0.0001
        else: F["spread_pips"]=0.5
        # cross-asset
        F=pd.concat([F,self.ca.build(bars,cross_asset)],axis=1)
        # filters
        F["vol_ok"]=vol_filter(F,ac,self.vm)
        F["news_ok"]=news_filter(bars.index,news_events,self.nb)
        # regime gating features (gold-vs-yield reliability switch)
        if self.enable_regime_gate:
            rbp = self.regime_gate.fit_predict(F)
            F["regime_break_prob"] = rbp
            # As regime breaks rise, shift weight toward safe-haven (gold/risk) and away from yields.
            F["gate_gold_weight"] = (0.2 + 0.8 * rbp).clip(0.0, 1.0)
            F["gate_yield_weight"] = (1.0 - 0.7 * rbp).clip(0.0, 1.0)
            F["gate_risk_weight"] = (0.3 + 0.7 * rbp).clip(0.0, 1.0)
        # sentiment
        if sentiment is not None:
            al=sentiment.reindex(bars.index,method="ffill").fillna(0)
            F["sentiment_decayed"]=sentiment_decay(al,self.dl)
        else: F["sentiment_decayed"]=0.0
        F["eco_surprise"]=eco_surprise(eco_act,eco_fc).reindex(bars.index).ffill().fillna(0) \
            if eco_act is not None else 0.0
        F["buzz"]=buzz_score(art_counts.reindex(bars.index,method="ffill").fillna(0)) \
            if art_counts is not None else 0.0
        # finbert (batch-create to avoid DataFrame fragmentation)
        fb_cols = [f"fb_{i}" for i in range(self.fb)]
        fb_base = pd.DataFrame(
            np.zeros((len(F), self.fb), dtype=np.float64),
            index=F.index,
            columns=fb_cols,
        )
        F = pd.concat([F, fb_base], axis=1)
        if finbert_embs is not None:
            proj=np.array([proj_finbert(e,self.fb) for e in finbert_embs])
            edf=pd.DataFrame(proj,index=bars.index[:len(proj)],columns=fb_cols)
            F.update(edf)
        # temporal
        h=bars.index.hour; m=bars.index.minute; tm=h*60+m
        temporal = pd.DataFrame({
            "time_sin": np.sin(2*np.pi*tm/1440),
            "time_cos": np.cos(2*np.pi*tm/1440),
            "day_sin": np.sin(2*np.pi*bars.index.dayofweek/5),
            "day_cos": np.cos(2*np.pi*bars.index.dayofweek/5),
            "london_ny": ((h>=13)&(h<=17)).astype(float),
        }, index=F.index)
        F = pd.concat([F, temporal], axis=1)
        n0=len(F); F=F.dropna()
        if n0>len(F): print(f"[Features] Dropped {n0-len(F):,} NaN rows → {len(F):,}×{F.shape[1]}")
        return F
