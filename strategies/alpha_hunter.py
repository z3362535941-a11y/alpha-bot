"""
AlphaBot 策略 v2: 强势突破 + 三线共振 + 动量加速
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alpha 原则 (目标胜率 45%+, RR 4:1):
  1. 真实突破: 收盘 > 20日高点 (非盘中触碰)
  2. 量能爆发: 突破日成交量 > 20日均量 × 1.8 (不要求2x, 容错)
  3. 三线共振: EMA9 > EMA21 > EMA50 且三线均上升
  4. MACD 柱状图 > 0 且递增 (动量加速中)
  5. RSI 在 50-72 (强势区但未超买)
  6. 突破后价格保持在突破点上方 (确认有效)
  7. ATR 扩张 (波动率增加 = 动量真实)

止损: ATR × 2.0 (突破策略需要呼吸空间)
止盈: SL × 4.0 (4:1 RR → 胜率 > 20% 即盈利)
追踪止损: 5.0% (给利润奔跑空间)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from strategies.indicators import rsi, ema, atr, macd, volume_ratio


@dataclass
class Signal:
    action: str
    reason: str
    confidence: float
    sl_pct: float
    tp_pct: float


def _confirmed_breakout(close: pd.Series, high: pd.Series,
                        volume: pd.Series, lookback: int = 20) -> tuple[bool, float]:
    """收盘价突破 N 日高点 + 成交量确认. Returns (is_breakout, vol_ratio)."""
    if len(close) < lookback + 2:
        return False, 0.0
    prev_high = high.iloc[-lookback-1:-1].max()
    vol_r = volume_ratio(volume, lookback).iloc[-1]
    broke_out = close.iloc[-1] > prev_high
    return broke_out and vol_r > 1.8, vol_r


def _three_ema_aligned(ema9: pd.Series, ema21: pd.Series, ema50: pd.Series) -> bool:
    """EMA9 > EMA21 > EMA50 且三线均在上升."""
    return (ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] and
            ema9.iloc[-1] > ema9.iloc[-3] and
            ema21.iloc[-1] > ema21.iloc[-5] and
            ema50.iloc[-1] > ema50.iloc[-8])


def _atr_expanding(high: pd.Series, low: pd.Series, close: pd.Series) -> bool:
    """当前 ATR > 过去 10 根均值 (波动率扩张 = 动量真实)."""
    atr_now = atr(high, low, close, 5).iloc[-1]
    atr_avg = atr(high, low, close, 14).iloc[-10:-1].mean()
    return atr_now > atr_avg * 1.05 if atr_avg > 0 else False


def analyze(df: pd.DataFrame, symbol: str = "") -> Signal:
    if len(df) < 60:
        return Signal("HOLD", "warmup", 0.0, 0.05, 0.20)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi_series = rsi(close, 14)
    rsi_val = rsi_series.iloc[-1]
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    atr_val = atr(high, low, close, 14).iloc[-1]
    macd_line, signal_line, hist = macd(close, 8, 21, 5)
    price = close.iloc[-1]

    atr_pct = atr_val / price
    sl_pct = max(0.04, min(0.10, atr_pct * 2.0))
    tp_pct = sl_pct * 4.0  # 4:1 RR

    # ── 核心信号评分 ───────────────────────────────────────────
    score = 0.0
    reasons = []

    # 突破 + 量能 (0.35分, 核心条件)
    breakout, vol_r = _confirmed_breakout(close, high, volume, 20)
    if breakout:
        score += 0.35
        reasons.append("vol_breakout")
    elif vol_r > 1.5:
        # 接近突破也给部分分
        score += 0.10

    # 三线共振 (0.30分)
    three_ema = _three_ema_aligned(ema9, ema21, ema50)
    if three_ema:
        score += 0.30
        reasons.append("3ema_aligned")
    elif ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
        score += 0.15
        reasons.append("ema_ordered")

    # MACD 柱状图正向且递增 (0.20分)
    macd_bull = hist.iloc[-1] > 0
    macd_accel = hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]
    if macd_bull and macd_accel:
        score += 0.20
        reasons.append("macd_accel")
    elif macd_bull:
        score += 0.08
        reasons.append("macd_bull")

    # RSI 强势区 (0.15分)
    rsi_strong = 50 < rsi_val < 72
    if rsi_strong:
        score += 0.15
        reasons.append("rsi_strong")
    elif 45 < rsi_val <= 50:
        score += 0.05

    # ATR 扩张确认动量 (0.10分)
    if _atr_expanding(high, low, close):
        score += 0.10
        reasons.append("atr_expand")

    # 需要突破 + 三线 + 其他至少一个确认
    has_breakout = "vol_breakout" in reasons
    has_3ema = "3ema_aligned" in reasons or "ema_ordered" in reasons

    if score >= 0.70 and has_breakout and has_3ema:
        return Signal("BUY", "+".join(reasons), score, sl_pct, tp_pct)

    return Signal("HOLD", f"score={score:.2f}", score, sl_pct, tp_pct)
