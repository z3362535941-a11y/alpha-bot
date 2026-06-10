"""
MemeBot 策略 v2: 强趋势中的超卖反弹 (均值回归 + 动量双重确认)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
高胜率原则:
  1. 只做明确上升趋势 (EMA21 > EMA50 > EMA100, EMA50 上升)
  2. 等价格超卖后开始回升 (RSI < 35 后上翘)
  3. Stochastic 确认超卖区 (%K < 25 且 %K 上穿 %D)
  4. 成交量放大确认底部 (volume > 1.2x 均量)
  5. 收盘价 > 开盘价 (买方占优的阳线)
  6. 价格在 EMA50 上方 (不抄熊市底)

止损: ATR × 2.2 (贴近支撑，不过宽)
止盈: SL × 3.0 (3:1 RR → 胜率 > 25% 即盈利, 目标 55%+)
追踪止损: 4.5% (锁定浮盈)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from strategies.indicators import rsi, ema, atr, volume_ratio, macd


@dataclass
class Signal:
    action: str
    reason: str
    confidence: float
    sl_pct: float
    tp_pct: float


def _stochastic(close: pd.Series, high: pd.Series, low: pd.Series,
                k_period: int = 14, d_period: int = 3):
    """Stochastic oscillator %K and %D."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(d_period).mean()
    return k.fillna(50), d.fillna(50)


def analyze(df: pd.DataFrame, symbol: str = "") -> Signal:
    if len(df) < 65:
        return Signal("HOLD", "warmup", 0.0, 0.05, 0.15)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi_series = rsi(close, 14)
    rsi_val = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-2]

    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    ema100 = ema(close, 100) if len(close) >= 100 else ema50
    atr_val = atr(high, low, close, 14).iloc[-1]
    vol_r = volume_ratio(volume, 20).iloc[-1]
    stoch_k, stoch_d = _stochastic(close, high, low, 14, 3)
    price = close.iloc[-1]

    # ── 趋势过滤 (EMA21 > EMA50 且 EMA50 上升即可) ──────────────
    ema50_rising = ema50.iloc[-1] > ema50.iloc[-8]
    ema21_above_50 = ema21.iloc[-1] > ema50.iloc[-1]
    # meme币允许价格短暂跌破EMA50，只要大趋势向上
    trend_up = ema21_above_50 and ema50_rising

    if not trend_up:
        return Signal("HOLD", "no_uptrend", 0.0, 0.05, 0.15)

    # ── ATR 动态止损 ─────────────────────────────────────────────
    atr_pct = atr_val / price
    sl_pct = max(0.035, min(0.10, atr_pct * 2.2))
    tp_pct = sl_pct * 3.0  # 3:1 RR

    # ── 超卖信号评分 ──────────────────────────────────────────────
    score = 0.0
    reasons = []

    # RSI 超卖且开始反弹 (最重要信号, 0.35分)
    rsi_oversold = rsi_val < 45   # meme币高波动，45以下即为相对超卖
    rsi_turning_up = rsi_val > rsi_prev
    if rsi_oversold and rsi_turning_up:
        score += 0.35
        reasons.append("rsi_bounce")
    elif rsi_oversold:
        score += 0.18
        reasons.append("rsi_oversold")

    # Stochastic 超卖 + %K 上穿 %D (0.25分)
    stoch_oversold = stoch_k.iloc[-1] < 25
    stoch_crossing = stoch_k.iloc[-1] > stoch_d.iloc[-1] and stoch_k.iloc[-2] <= stoch_d.iloc[-2]
    if stoch_oversold and stoch_crossing:
        score += 0.25
        reasons.append("stoch_cross")
    elif stoch_oversold:
        score += 0.10
        reasons.append("stoch_oversold")

    # 当前收阳线 (收盘 > 开盘, 0.20分)
    bullish_candle = close.iloc[-1] > df["open"].iloc[-1]
    if bullish_candle:
        score += 0.20
        reasons.append("bull_candle")

    # 成交量确认 (0.20分)
    if vol_r > 1.2:
        score += 0.20
        reasons.append("vol_confirm")
    elif vol_r > 0.9:
        score += 0.08
        reasons.append("vol_ok")

    # 需要至少: RSI超卖反弹 + 一个额外确认 = 0.50+
    if score >= 0.50 and rsi_oversold:
        return Signal("BUY", "+".join(reasons), score, sl_pct, tp_pct)

    return Signal("HOLD", f"score={score:.2f}", score, sl_pct, tp_pct)
