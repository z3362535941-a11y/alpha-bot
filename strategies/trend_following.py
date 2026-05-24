"""
MainBot 策略: 趋势内回调做多 (Pullback-in-Trend)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
高胜率原则:
  1. 先确认趋势 (EMA20>EMA50, EMA50 上升)
  2. 等价格回调到 EMA20 附近或布林带下轨
  3. RSI 35-50 (回调但未崩)
  4. OBV 依然上升 (机构仍在买)
  5. MACD 柱状图从负转正

止损: ATR × 2.5 (宽止损，减少噪音止损)
止盈: ATR × 5.0 (2:1 RR，胜率 > 34% 即盈利)
"""
import pandas as pd
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
    if len(df) < 60:
        return Signal("HOLD", "warmup", 0.0, 0.05, 0.15)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi_series = rsi(close, 14)
    rsi_val = rsi_series.iloc[-1]
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
    atr_val = atr(high, low, close, 14).iloc[-1]
    obv_series = obv(close, volume)
    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    price = close.iloc[-1]

    # ─── 趋势过滤 (必须满足) ─────────────────────────────────
    ema50_rising = ema50.iloc[-1] > ema50.iloc[-8]
    trend_up = ema20.iloc[-1] > ema50.iloc[-1] and ema50_rising

    if not trend_up:
        return Signal("HOLD", "no_uptrend", 0.0, 0.05, 0.15)

    atr_pct = atr_val / price
    sl_pct = max(0.04, min(0.10, atr_pct * 2.5))
    tp_pct = sl_pct * 2.5  # 2:1 RR → 需胜率 > 29% 即盈利

    # ─── 回调确认 ────────────────────────────────────────────
    near_ema20 = price <= ema20.iloc[-1] * 1.015     # 价格在 EMA20 附近（含以下 1.5%）
    near_bb_lower = price <= bb_lower.iloc[-1] * 1.02
    pullback = near_ema20 or near_bb_lower

    rsi_pullback = 30 < rsi_val < 52              # RSI 回调但未崩
    rsi_turning = rsi_series.iloc[-1] > rsi_series.iloc[-2]  # RSI 开始回升
    obv_uptrend = obv_series.iloc[-1] > obv_series.iloc[-5]  # OBV 依然上升
    macd_recovering = hist.iloc[-1] > hist.iloc[-3]           # MACD 柱状改善

    score = 0.0
    reasons = ["uptrend"]
    score += 0.20

    if pullback:
        score += 0.30
        reasons.append("pullback_to_support")
    if rsi_pullback and rsi_turning:
        score += 0.25
        reasons.append("rsi_bounce")
    elif rsi_pullback:
        score += 0.15
        reasons.append("rsi_ok")
    if obv_uptrend:
        score += 0.15
        reasons.append("obv_confirm")
    if macd_recovering:
        score += 0.10
        reasons.append("macd_recover")

    if score >= 0.70:
        return Signal("BUY", "+".join(reasons), score, sl_pct, tp_pct)

    return Signal("HOLD", f"score={score:.2f}", score, sl_pct, tp_pct)
