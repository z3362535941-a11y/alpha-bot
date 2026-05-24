"""
MemeBot 策略: RSI 超卖反弹 + EMA 趋势过滤
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
逻辑: 只在明确上升趋势中买入超卖回调（均值回归）
高胜率原则: 不追涨，等价格跌回支撑再买
- 趋势: EMA21 > EMA50 且 EMA50 本身在上升
- 超卖: RSI < 35
- 量能: 成交量 > 均量
- 止损: ATR × 2.5 (给足波动空间)
- 止盈: ATR × 4.0 (2.5:1 最低 RR，期望值正)
"""
import pandas as pd
from dataclasses import dataclass
from strategies.indicators import rsi, ema, atr, volume_ratio, macd


@dataclass
class Signal:
    action: str
    reason: str
    confidence: float
    sl_pct: float
    tp_pct: float


def analyze(df: pd.DataFrame, symbol: str = "") -> Signal:
    if len(df) < 55:
        return Signal("HOLD", "warmup", 0.0, 0.05, 0.12)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi_series = rsi(close, 14)
    rsi_val = rsi_series.iloc[-1]
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    atr_val = atr(high, low, close, 14).iloc[-1]
    vol_ratio = volume_ratio(volume, 20).iloc[-1]
    price = close.iloc[-1]
    _, _, hist = macd(close, 8, 21, 5)

    # ─── 趋势过滤: EMA21 > EMA50 且 EMA50 在上升 ──────────────
    ema50_rising = ema50.iloc[-1] > ema50.iloc[-10]
    trend_up = ema21.iloc[-1] > ema50.iloc[-1] and ema50_rising

    if not trend_up:
        return Signal("HOLD", "no_uptrend", 0.0, 0.05, 0.12)

    atr_pct = atr_val / price
    sl_pct = max(0.04, min(0.12, atr_pct * 2.5))
    tp_pct = sl_pct * 2.5  # 2.5:1 RR → 期望值正 (需胜率 > 29%)

    # ─── 高置信买入: RSI 超卖 + 量能 ──────────────────────────
    rsi_oversold = rsi_val < 38
    rsi_recovering = rsi_series.iloc[-1] > rsi_series.iloc[-2]  # RSI 开始回升
    vol_ok = vol_ratio > 0.8
    price_above_ema50 = price > ema50.iloc[-1]
    macd_hist_ok = hist.iloc[-1] > hist.iloc[-3]  # MACD 动量改善

    score = 0.0
    reasons = []

    if rsi_oversold:
        score += 0.40
        reasons.append("rsi_oversold")
    if rsi_recovering:
        score += 0.25
        reasons.append("rsi_turning")
    if price_above_ema50:
        score += 0.15
        reasons.append("above_ema50")
    if macd_hist_ok:
        score += 0.10
        reasons.append("macd_improve")
    if vol_ok:
        score += 0.10
        reasons.append("vol_ok")

    if score >= 0.65:
        return Signal("BUY", "+".join(reasons), score, sl_pct, tp_pct)

    return Signal("HOLD", f"score={score:.2f}", score, sl_pct, tp_pct)
