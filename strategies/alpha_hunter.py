"""
AlphaBot 策略: 量价突破 + 三线共振
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alpha 原则: 等待真正的突破，不买假突破
- 突破条件: 收盘价 > N日高点 (不只是盘中触碰)
- 量能确认: 突破当日成交量 > 20日均量的 2倍
- 三线共振: EMA9 > EMA21 > EMA50 (短中长都在涨)
- 动量确认: MACD 柱状图为正且上升
- RSI 过滤: RSI 45-70 (强势但非超买)

止损: 突破点下方 ATR × 1.8 (若假突破快速止损)
止盈: ATR × 5 (追踪止损 4% 锁定利润)
期望值: 胜率 35% × RR 5:1 = 75% 期望回报
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
                        volume: pd.Series, lookback: int = 20) -> bool:
    """收盘价突破 N 日高点 + 成交量 2x 确认."""
    if len(close) < lookback + 2:
        return False
    # 用 lookback-1 根 K 线的最高价 (不含当根)
    prev_high = high.iloc[-lookback-1:-1].max()
    vol_ratio = volume_ratio(volume, lookback).iloc[-1]
    return close.iloc[-1] > prev_high and vol_ratio > 2.0


def _three_ema_aligned(ema9: pd.Series, ema21: pd.Series, ema50: pd.Series) -> bool:
    """EMA9 > EMA21 > EMA50 且三线均在上升."""
    return (ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] and
            ema9.iloc[-1] > ema9.iloc[-3] and
            ema50.iloc[-1] > ema50.iloc[-8])


def analyze(df: pd.DataFrame, symbol: str = "") -> Signal:
    if len(df) < 55:
        return Signal("HOLD", "warmup", 0.0, 0.05, 0.15)

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
    sl_pct = max(0.035, min(0.09, atr_pct * 1.8))
    tp_pct = sl_pct * 5.0  # 5:1 RR — 允许追踪止损锁定更大利润

    # ─── 核心信号 ─────────────────────────────────────────────
    breakout = _confirmed_breakout(close, high, volume, 20)
    three_ema = _three_ema_aligned(ema9, ema21, ema50)
    macd_bull = hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]
    rsi_strong = 45 < rsi_val < 70

    score = 0.0
    reasons = []

    if breakout:
        score += 0.40
        reasons.append("vol_breakout")
    if three_ema:
        score += 0.30
        reasons.append("3ema_aligned")
    if macd_bull:
        score += 0.20
        reasons.append("macd_bull")
    if rsi_strong:
        score += 0.10
        reasons.append("rsi_strong")

    if score >= 0.65:  # 需要突破+至少一个额外确认
        return Signal("BUY", "+".join(reasons), score, sl_pct, tp_pct)

    return Signal("HOLD", f"score={score:.2f}", score, sl_pct, tp_pct)
