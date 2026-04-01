"""
features/finbert_sentiment.py
==============================
Three-tier local sentiment pipeline for Forex news headlines.

Tier 1 — Ollama (preferred)
    Calls http://localhost:11434/api/generate with a structured prompt.
    Uses whichever model is pulled (mistral, llama3, phi3, etc.).
    Zero API cost, GPU-accelerated, completely offline.

Tier 2 — FinBERT (fallback)
    Uses transformers.pipeline("text-classification", model="ProsusAI/finbert").
    ~500 MB download on first use. Fine-tuned on financial language.

Tier 3 — VADER (zero-dep fallback)
    Rule-based, no download required, always available.
    Less accurate but fast and robust.

Usage:
    from features.finbert_sentiment import SentimentPipeline
    pipe = SentimentPipeline()                    # auto-detects best tier
    score = pipe.score_headlines(["EUR/USD rises on ECB rate hike"])
    # → float in [-1, +1]
    series = pipe.score_to_series(headlines_df, bars)
    print(pipe.active_backend())                  # "ollama" / "finbert" / "vader"

Caching:
    Results cached to data/embeddings/ keyed by MD5(headline).
    Avoids repeated inference on identical headlines.
"""

import os
import json
import hashlib
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path(os.getenv(
    "SENTIMENT_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "embeddings")
))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "sentiment_cache.pkl"

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

FINBERT_MODEL = "ProsusAI/finbert"
LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ── cache helpers ──────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Tier 1: Ollama ─────────────────────────────────────────────────────────

def _score_ollama(text: str, model: str = OLLAMA_MODEL,
                  timeout: float = 8.0) -> Optional[float]:
    """
    Ask the local Ollama LLM to rate the sentiment of a headline.
    Returns float in [-1, +1], or None if Ollama is unreachable.
    """
    import urllib.request, urllib.error

    prompt = (
        f"You are a financial sentiment analyser for Forex markets. "
        f"Rate the following headline's market sentiment as a single number "
        f"between -1.0 (very bearish) and +1.0 (very bullish). "
        f"Reply with ONLY the number.\n\nHeadline: {text}\n\nSentiment score:"
    )

    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        raw = str(data.get("response", "")).strip()
        # Extract the first float-like substring
        import re
        m = re.search(r"-?\d+\.?\d*", raw)
        if m:
            score = float(m.group())
            return float(np.clip(score, -1.0, 1.0))
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return None   # Ollama not running
    except Exception:
        return None


# ── Tier 2: FinBERT ─────────────────────────────────────────────────────────

_finbert_pipeline = None   # lazily loaded


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            truncation=True,
            max_length=512,
        )
    return _finbert_pipeline


def _score_finbert(text: str) -> Optional[float]:
    try:
        pipe = _get_finbert()
        result = pipe(text[:512])[0]
        label  = result["label"].lower()
        score  = result["score"]       # confidence
        return LABEL_MAP.get(label, 0.0) * score
    except Exception:
        return None


# ── Tier 3: VADER ──────────────────────────────────────────────────────────

_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            # Simple lexicon fallback if vaderSentiment not installed
            class _Simple:
                BULLISH = {"rise", "rises", "gain", "gains", "up", "rally",
                           "bullish", "strong", "beat", "beats", "surges",
                           "high", "positive", "growth"}
                BEARISH = {"fall", "falls", "drop", "drops", "down", "decline",
                           "bearish", "weak", "miss", "misses", "tumbles",
                           "low", "negative", "recession", "crisis"}
                def polarity_scores(self, text):
                    words = set(text.lower().split())
                    bull = len(words & self.BULLISH)
                    bear = len(words & self.BEARISH)
                    total = bull + bear
                    compound = (bull - bear) / total if total else 0.0
                    return {"compound": float(np.clip(compound, -1, 1))}
            _vader_analyzer = _Simple()
    return _vader_analyzer


def _score_vader(text: str) -> float:
    analyzer = _get_vader()
    return float(analyzer.polarity_scores(text)["compound"])


# ── Main pipeline ───────────────────────────────────────────────────────────

class SentimentPipeline:
    """
    Three-tier sentiment pipeline: Ollama → FinBERT → VADER.

    Parameters
    ----------
    prefer_backend : str or None
        Force a specific backend ("ollama", "finbert", "vader").
        If None, auto-detects the best available.
    ollama_model : str
        Ollama model name to use (e.g. "mistral", "llama3", "phi3").
    use_cache : bool
        Whether to cache results by headline MD5.
    """

    def __init__(
        self,
        prefer_backend: Optional[str] = None,
        ollama_model:   str = OLLAMA_MODEL,
        use_cache:      bool = True,
    ):
        self._prefer    = prefer_backend
        self._ollama_m  = ollama_model
        self._use_cache = use_cache
        self._cache: dict = _load_cache() if use_cache else {}
        self._backend: Optional[str] = None   # detected after first call

    def active_backend(self) -> str:
        """Return the name of the currently active backend."""
        return self._backend or "unknown"

    def _score_single(self, text: str) -> float:
        """Score one headline through the tier cascade."""
        key = _cache_key(text)
        if self._use_cache and key in self._cache:
            return self._cache[key]

        score: Optional[float] = None

        if self._prefer in (None, "ollama"):
            score = _score_ollama(text, self._ollama_m)
            if score is not None:
                self._backend = "ollama"

        if score is None and self._prefer in (None, "finbert"):
            try:
                score = _score_finbert(text)
                if score is not None:
                    self._backend = "finbert"
            except ImportError:
                pass

        if score is None:
            score = _score_vader(text)
            self._backend = "vader"

        if self._use_cache:
            self._cache[key] = score
            _save_cache(self._cache)

        return score

    def score_headlines(self, headlines: List[str]) -> float:
        """
        Score a list of headlines and return the weighted mean sentiment.
        Returns float in [-1, +1].
        """
        if not headlines:
            return 0.0
        scores = np.array([self._score_single(h) for h in headlines],
                          dtype=np.float32)
        # Weight by absolute magnitude (stronger signals count more)
        weights = np.abs(scores) + 0.01
        result  = float(np.average(scores, weights=weights))
        return float(np.clip(result, -1.0, 1.0))

    def score_to_series(
        self,
        headlines_df: pd.DataFrame,
        bars:         pd.DataFrame,
        text_col:     str = "headline",
        time_col:     str = "datetime",
        decay_lambda: float = 0.1,
    ) -> pd.Series:
        """
        Align timestamped headlines to bars.index, apply exponential decay.

        Parameters
        ----------
        headlines_df : DataFrame with `text_col` and `time_col` columns.
        bars         : Bar DataFrame with DatetimeTZIndex.
        decay_lambda : Exponential decay rate (higher = faster decay).

        Returns pd.Series aligned to bars.index in [-1, +1].
        """
        if len(headlines_df) == 0:
            return pd.Series(0.0, index=bars.index, name="sentiment")

        # Score and index
        times  = pd.DatetimeIndex(headlines_df[time_col]).tz_localize("UTC") \
                 if headlines_df[time_col].dt.tz is None \
                 else pd.DatetimeIndex(headlines_df[time_col]).tz_convert("UTC")
        scores = [self._score_single(h) for h in headlines_df[text_col]]
        scored = pd.Series(scores, index=times, dtype=float).sort_index()

        # Resample: take mean score per bar
        bar_scores = scored.reindex(scored.index.union(bars.index)).sort_index()
        bar_scores = bar_scores.reindex(bars.index)   # keep only bar timestamps

        # Exponential decay forward-fill
        result = pd.Series(0.0, index=bars.index, dtype=float)
        last_time: Optional[pd.Timestamp] = None
        last_val:  float = 0.0
        for ts in bars.index:
            if pd.notna(bar_scores.get(ts, np.nan)) and bar_scores.get(ts, np.nan) != 0:
                last_time = ts
                last_val  = float(bar_scores[ts])
                result[ts] = last_val
            elif last_time is not None:
                elapsed = (ts - last_time).total_seconds()
                result[ts] = last_val * np.exp(-decay_lambda * elapsed)

        return result.clip(-1.0, 1.0).rename("sentiment")


# ── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SentimentPipeline — smoke test")
    pipe = SentimentPipeline(use_cache=False)

    test_headlines = [
        "EUR/USD rises sharply after ECB surprise rate hike",
        "US Dollar falls amid weak NFP data, recession fears grow",
        "Markets steady as Fed holds rates unchanged",
    ]

    for h in test_headlines:
        s = pipe.score_headlines([h])
        print(f"  [{pipe.active_backend():8s}] {s:+.3f}  {h[:60]}")

    print(f"\n  Active backend: {pipe.active_backend()}")
    print("OK ✓")
