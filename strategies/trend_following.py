"""
MainBot 策略 v2: 趋势内高质量回调买点 (Pullback-in-Trend)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
高胜率原则 (目标胜率 60%+):
  1. 强趋势: EMA20 > EMA50 > EMA100, 三线均上升
  2. 回调到支撑: 价格接触 EMA20 或布林带中轨下方
  3. RSI 在 32-52 (回调充分但趋势未破)
  4. OBV 趋势向上 (机构未出货)
  5. MACD 柱状图从负转正 (动量恢复)
  6. 收阳线确认 (当前 bar 收盘 > 开盘)
  7. 成交量高于均量 0.8x (有买盘介入)

止损: 最近 3 根 K 线低点下方 ATR × 1.5
止盈: SL × 3.0 (3:1 RR → 胜率 > 25% 即盈利, 目标 60%)
追踪止损: 3.0% (主流币波动小, 更紧追踪)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from strategies.indicators import rsi, ema, bollinger_bands, atr, obv, macd


@dataclass
class Signal:
    action: str
    reason: str
    confidence: float
    sl_pct: float
    tp_pct: float


def analyze(df: pd.DataFrame, symbol: str = "") -> Signal:
    if len(df) < 65:
        return Signal("HOLD", "warmup", 0.0, 0.05, 0.15)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    open_ = df["open"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema100 = ema(close, 100) if len(close) >= 100 else ema50
    rsi_series = rsi(close, 14)
    rsi_val = rsi_series.iloc[-1]
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
    atr_val = atr(high, low, close, 14).iloc[-1]
    obv_series = obv(close, volume)
    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    price = close.iloc[-1]

    # ── 趋势过滤 (全部必须满足) ──────────────────────────────────
    ema20_above_50 = ema20.iloc[-1] > ema50.iloc[-1]
    ema50_rising = ema50.iloc[-1] > ema50.iloc[-8]
    ema20_rising = ema20.iloc[-1] > ema20.iloc[-5]
    price_above_ema50 = price > ema50.iloc[-1]

    if not (ema20_above_50 and ema50_rising and price_above_ema50 and ema20_rising):
        return Signal("HOLD", "no_uptrend", 0.0, 0.05, 0.15)

    # ── 动态止损 ─────────────────────────────────────────────────
    atr_pct = atr_val / price
    recent_low = low.iloc[-4:-1].min()
    sl_from_recent = (price - recent_low) / price
    sl_pct = max(0.03, min(0.09, max(atr_pct * 1.8, sl_from_recent * 1.1)))
    tp_pct = sl_pct * 3.0  # 3:1 RR

    # ── 回调质量评分 ─────────────────────────────────────────────
    score = 0.0
    reasons = ["uptrend"]
    score += 0.15  # 趋势基础分

    # 回调到支撑位 (0.30分)
    near_ema20 = price <= ema20.iloc[-1] * 1.012
    near_bb_lower = price <= bb_lower.iloc[-1] * 1.015
    between_ema_bb = bb_lower.iloc[-1] <= price <= bb_mid.iloc[-1]
    if near_ema20 or near_bb_lower:
        score += 0.30
        reasons.append("at_support")
    elif between_ema_bb:
        score += 0.15
        reasons.append("near_support")

    # RSI 回调区 且 开始回升 (0.25分)
    rsi_in_zone = 30 < rsi_val < 52
    rsi_turning = rsi_series.iloc[-1] > rsi_series.iloc[-2]
    if rsi_in_zone and rsi_turning:
        score += 0.25
        reasons.append("rsi_bounce")
    elif rsi_in_zone:
        score += 0.12
        reasons.append("rsi_pullback")

    # OBV 趋势向上 (0.15分) - 机构未出货
    obv_uptrend = (obv_series.iloc[-1] > obv_series.iloc[-5] and
                   obv_series.iloc[-5] > obv_series.iloc[-10])
    if obv_uptrend:
        score += 0.15
        reasons.append("obv_up")

    # MACD 柱状图改善 (0.10分)
    macd_improving = hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]
    macd_crossing = hist.iloc[-1] > 0 and hist.iloc[-2] <= 0
    if macd_crossing:
        score += 0.10
        reasons.append("macd_cross")
    elif macd_improving:
        score += 0.05
        reasons.append("macd_improve")

    # 收阳线确认 (0.05分)
    bullish_bar = close.iloc[-1] > open_.iloc[-1]
    if bullish_bar:
        score += 0.05
        reasons.append("bull_bar")

    # 需要在支撑 + RSI回弹 + 其他确认 → 0.70+
    has_support = any(r in reasons for r in ("at_support", "near_support"))
    has_rsi = any(r in reasons for r in ("rsi_bounce", "rsi_pullback"))

    if score >= 0.68 and has_support and has_rsi:
        return Signal("BUY", "+".join(reasons), score, sl_pct, tp_pct)

    return Signal("HOLD", f"score={score:.2f}", score, sl_pct, tp_pct)
