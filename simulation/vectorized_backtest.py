"""
Vectorized Backtesting Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
O(n) per symbol: all indicators are precomputed once on the full
dataset; the bar loop only reads scalar values by integer index.

Supported bot types:
  memo  — RSI Exhaustion Reversal  (MemeBot logic)
  main  — Pullback to EMA Bounce   (MainstreamBot logic)
  alpha — Post-Breakout Pullback   (AlphaBot logic)

Returns
-------
trades  : list[dict]  — one dict per closed trade
summary : dict        — aggregate performance metrics

Usage
-----
    from simulation.vectorized_backtest import run_vectorized
    trades, summary = run_vectorized(df, "BTCUSDT", "main", 10_000, {})
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Default strategy parameters
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, dict] = {
    "memo": dict(
        rsi_oversold=38,
        rsi_confirm=40,
        sl_atr_mult=2.5,
        tp_rr=2.5,
        ema_slow=50,
    ),
    "main": dict(
        rsi_pullback_max=52,
        sl_atr_mult=2.5,
        tp_rr=2.5,
        ema_fast=20,
        ema_slow=50,
    ),
    "alpha": dict(
        breakout_lookback=20,
        vol_mult=2.0,
        sl_atr_mult=1.8,
        tp_rr=5.0,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers — all operate on full Series, return full Series (O(n))
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, pandas ewm (O(n))."""
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI using EWM (O(n))."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Average True Range using EWM smoothing (O(n))."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (O(n))."""
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def _vol_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume relative to rolling mean (O(n))."""
    return volume / volume.rolling(window=period).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized rolling-window helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_any_below(arr: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """
    Returns a boolean array where result[i] is True iff any value in
    arr[i-window : i] (exclusive of i) is below `threshold`.
    O(n) via a running counter.
    """
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    # count of values below threshold in the sliding window
    below = (arr < threshold).astype(np.int8)
    # prefix sum for O(1) range queries
    prefix = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        prefix[k + 1] = prefix[k] + below[k]
    for i in range(window, n):
        # window covers [i-window, i)
        out[i] = (prefix[i] - prefix[i - window]) > 0
    return out


def _rolling_any_at_or_below(arr: np.ndarray, ref: np.ndarray,
                               factor: float, window: int) -> np.ndarray:
    """
    Returns True at i if any arr[j] <= ref[j]*factor for j in [i-window, i).
    O(n*window) in the worst case but window is small (3), so effectively O(n).
    For window <= 5 a simple loop is fine.
    """
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    cond = arr <= ref * factor          # element-wise boolean
    # prefix sum for O(1) range queries
    prefix = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        prefix[k + 1] = prefix[k] + int(cond[k])
    for i in range(window, n):
        out[i] = (prefix[i] - prefix[i - window]) > 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

def _size_position(cash: float, price: float, sl_pct: float,
                   risk_pct: float = 0.02) -> float:
    """
    Fixed-fractional: risk `risk_pct` of current cash per trade.
    qty = risk_amount / (price * sl_pct), capped to available cash.
    """
    if price <= 0 or sl_pct <= 0:
        return 0.0
    risk_amount = cash * risk_pct
    qty = risk_amount / (price * sl_pct)
    max_qty = cash / price          # never spend more than available cash
    return min(qty, max_qty)


# ─────────────────────────────────────────────────────────────────────────────
# Equity-curve statistics  — O(n), fully vectorized
# ─────────────────────────────────────────────────────────────────────────────

def _compute_summary(
    trades: list[dict],
    capital: float,
    close: np.ndarray,
    n_bars: int,
) -> dict:
    """
    Build a bar-by-bar equity curve from the trade list, then compute
    risk/performance statistics.

    Equity curve construction is O(n) using numpy scatter + cumsum:
      1. Start with a flat cash array.
      2. For each trade, record delta-cash events at entry and exit bars.
      3. Cumsum the delta array to get the running-cash component.
      4. Overlay mark-to-market P&L for the open portion of each trade
         using vectorised index-array slicing.
    """
    total = len(trades)
    if total == 0:
        return dict(
            total_return_pct=0.0,
            win_rate=0.0,
            max_drawdown_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            profit_factor=0.0,
            max_consecutive_losses=0,
            total_trades=0,
        )

    # ── Step 1: build a cash-delta array (O(n)) ──────────────────────────────
    # cash_delta[i] captures the change in cash at bar i.
    # We start with `capital` at bar 0.
    cash_events = np.zeros(n_bars, dtype=float)
    cash_events[0] = capital

    # For each trade we store the qty used (computed from cash *at entry*).
    # We rebuild running cash in the same forward pass used for equity.
    # Trade list is already in chronological order (one position at a time).
    trade_qtys: list[float] = []

    running_cash = capital
    for t in trades:
        ei = t["entry_idx"]
        xi = min(t["exit_idx"], n_bars - 1)
        entry_p = t["entry_price"]
        exit_p = t["exit_price"]
        sl_pct = (entry_p - t["sl"]) / entry_p if entry_p > 0 else 0.02
        qty = _size_position(running_cash, entry_p, sl_pct)
        trade_qtys.append(qty)

        # Cash decreases at entry, increases at exit
        cost = entry_p * qty
        running_cash -= cost
        running_cash += exit_p * qty   # update running_cash for next trade sizing

    # ── Step 2: build equity curve bar-by-bar (O(n)) ─────────────────────────
    # equity[i] = cash-held + mark-to-market value of open position at bar i
    equity = np.empty(n_bars, dtype=float)
    cash = capital

    # Build a sorted list of trade events for a single-pass sweep.
    # entry_events[i] = (entry_price, qty, exit_idx)
    # We rely on positions being non-overlapping (one position at a time).

    # Map: bar_index -> (event_type, trade_index)
    # event_type: 'entry' or 'exit'
    entry_map: dict[int, int] = {}   # bar -> trade index
    exit_map: dict[int, int] = {}    # bar -> trade index
    for idx, t in enumerate(trades):
        entry_map[t["entry_idx"]] = idx
        exit_map[min(t["exit_idx"], n_bars - 1)] = idx

    open_trade_idx: int = -1
    open_qty: float = 0.0
    open_entry_p: float = 0.0
    cash = capital

    for b in range(n_bars):
        # Handle exit first (so we free cash before potentially entering same bar)
        if b in exit_map:
            t_idx = exit_map[b]
            t = trades[t_idx]
            exit_p = t["exit_price"]
            qty_t = trade_qtys[t_idx]
            # Return proceeds to cash
            cash += exit_p * qty_t
            open_trade_idx = -1
            open_qty = 0.0
            open_entry_p = 0.0

        # Handle entry
        if b in entry_map:
            t_idx = entry_map[b]
            t = trades[t_idx]
            entry_p = t["entry_price"]
            qty_t = trade_qtys[t_idx]
            cash -= entry_p * qty_t
            open_trade_idx = t_idx
            open_qty = qty_t
            open_entry_p = entry_p

        # Mark-to-market equity
        if open_trade_idx >= 0:
            equity[b] = cash + open_qty * close[b]
        else:
            equity[b] = cash

    # ── Drawdown ─────────────────────────────────────────────────────────────
    peak = np.maximum.accumulate(equity)
    # Guard against zero peak (edge case)
    safe_peak = np.where(peak > 0, peak, 1.0)
    drawdown = (peak - equity) / safe_peak * 100.0
    max_drawdown_pct = float(drawdown.max())

    # ── Daily returns (treat each bar as one period) ─────────────────────────
    # Use log returns for Sharpe/Sortino to avoid compounding bias.
    # Guard against non-positive equity values.
    safe_equity = np.where(equity > 0, equity, np.nan)
    log_returns = np.diff(np.log(safe_equity))
    # Drop NaNs (can appear if equity was 0 / negative due to data issues)
    log_returns = log_returns[np.isfinite(log_returns)]

    rf_daily = 0.0  # risk-free rate ~0 in crypto context
    mean_ret = float(log_returns.mean()) if len(log_returns) > 0 else 0.0
    std_ret = float(log_returns.std(ddof=1)) if len(log_returns) > 1 else 0.0

    sharpe = (mean_ret - rf_daily) / std_ret * math.sqrt(252) if std_ret > 0 else 0.0

    downside = log_returns[log_returns < rf_daily]
    std_down = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_ret - rf_daily) / std_down * math.sqrt(252) if std_down > 0 else 0.0

    # ── Trade statistics ─────────────────────────────────────────────────────
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / total * 100.0

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max consecutive losses
    max_consec = 0
    cur_consec = 0
    for t in trades:
        if t["pnl"] <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    final_equity = float(equity[-1])
    total_return_pct = (final_equity - capital) / capital * 100.0

    return dict(
        total_return_pct=round(total_return_pct, 4),
        win_rate=round(win_rate, 2),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        profit_factor=round(profit_factor, 4),
        max_consecutive_losses=max_consec,
        total_trades=total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal generators — fully vectorized, return bool numpy arrays length n
# ─────────────────────────────────────────────────────────────────────────────

def _signals_memo(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    params: dict,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MemoBot — RSI Exhaustion Reversal ("Buy the bounce, not the dip")

    BUY when ALL of:
      - RSI was < rsi_oversold in the last 5 bars  (dip was there)
      - RSI is now > rsi_confirm                   (confirmed recovery)
      - EMA21 > EMA_slow                           (uptrend filter)
      - EMA_slow is rising vs 10 bars ago

    All conditions are built as boolean arrays; no Python loop over bars.
    Returns (buy_signal, sl_arr, tp_arr) — 1-D numpy arrays of length n.
    """
    rsi_oversold: float = params["rsi_oversold"]
    rsi_confirm: float  = params["rsi_confirm"]
    sl_mult: float      = params["sl_atr_mult"]
    tp_rr: float        = params["tp_rr"]
    ema_slow_p: int     = int(params["ema_slow"])

    s   = pd.Series(close)
    h   = pd.Series(high)
    lo  = pd.Series(low)

    # ── Precompute all indicators on full dataset — O(n) each ────────────────
    rsi_s    = _rsi(s, 14).values                      # shape (n,)
    ema21    = _ema(s, 21).values
    ema_slow = _ema(s, ema_slow_p).values
    atr_s    = _atr(h, lo, s, 14).values

    # ── Vectorized conditions ─────────────────────────────────────────────────

    # cond1: RSI dipped below rsi_oversold in any of the previous 5 bars.
    # _rolling_any_below(arr, threshold, window)[i] checks arr[i-5 : i].
    rsi_dipped = _rolling_any_below(rsi_s, rsi_oversold, window=5)

    # cond2: RSI now (at bar i) is above rsi_confirm
    rsi_now_ok = rsi_s > rsi_confirm

    # cond3: EMA21 > EMA_slow (uptrend)
    trend_up = ema21 > ema_slow

    # cond4: EMA_slow is rising — compare to 10 bars ago (shift by 10)
    ema_slow_10ago = np.roll(ema_slow, 10)
    ema_slow_10ago[:10] = np.nan
    ema_rising = ema_slow > ema_slow_10ago

    # ── ATR-based SL/TP arrays (only computed where buy fires, but built
    #    for all bars so indexing is O(1) in the main loop) ─────────────────
    atr_pct  = np.where(close > 0, atr_s / close, 0.0)
    sl_raw   = atr_pct * sl_mult
    sl_arr   = np.clip(sl_raw, 0.04, 0.12)
    tp_arr   = sl_arr * tp_rr

    # ── Combined buy signal ───────────────────────────────────────────────────
    buy = rsi_dipped & rsi_now_ok & trend_up & ema_rising

    return buy, sl_arr, tp_arr


def _signals_main(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    params: dict,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MainBot — Pullback to EMA Bounce ("Buy after the bounce off EMA")

    BUY when ALL of:
      - Price touched EMA_fast in last 3 bars: close[j] <= EMA_fast[j]*1.01
      - Now price > EMA_fast                    (bounced above)
      - RSI < rsi_pullback_max                  (not overbought)
      - EMA_fast > EMA_slow                     (uptrend)
      - OBV rising vs 5 bars ago

    Fully vectorized using prefix sums; O(n) total.
    """
    rsi_max: float   = params["rsi_pullback_max"]
    sl_mult: float   = params["sl_atr_mult"]
    tp_rr: float     = params["tp_rr"]
    ema_fast_p: int  = int(params["ema_fast"])
    ema_slow_p: int  = int(params["ema_slow"])

    s   = pd.Series(close)
    h   = pd.Series(high)
    lo  = pd.Series(low)
    v   = pd.Series(volume)

    # ── Precompute all indicators — O(n) each ────────────────────────────────
    rsi_s    = _rsi(s, 14).values
    ema_fast = _ema(s, ema_fast_p).values
    ema_slow = _ema(s, ema_slow_p).values
    atr_s    = _atr(h, lo, s, 14).values
    obv_s    = _obv(s, v).values

    # ── Vectorized conditions ─────────────────────────────────────────────────

    # cond1: price touched EMA_fast in last 3 bars (i-3 .. i-1)
    # touched means close[j] <= ema_fast[j] * 1.01
    touched = _rolling_any_at_or_below(close, ema_fast, factor=1.01, window=3)

    # cond2: current bar price is above EMA_fast
    price_above = close > ema_fast

    # cond3: RSI not overbought
    rsi_ok = rsi_s < rsi_max

    # cond4: uptrend
    trend_up = ema_fast > ema_slow

    # cond5: OBV rising vs 5 bars ago
    obv_5ago = np.roll(obv_s, 5)
    obv_5ago[:5] = np.nan
    obv_rising = obv_s > obv_5ago

    # ── ATR-based SL/TP ───────────────────────────────────────────────────────
    atr_pct = np.where(close > 0, atr_s / close, 0.0)
    sl_arr  = np.clip(atr_pct * sl_mult, 0.04, 0.10)
    tp_arr  = sl_arr * tp_rr

    # ── Combined buy signal ───────────────────────────────────────────────────
    buy = touched & price_above & rsi_ok & trend_up & obv_rising

    return buy, sl_arr, tp_arr


def _signals_alpha(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    params: dict,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    AlphaBot — Post-Breakout Pullback ("Buy the retest, not the breakout")

    BUY when ALL of:
      - Price broke above N-day high 3-10 bars ago AND that breakout bar
        had volume > vol_mult × 20-bar average volume        (historical)
      - Current price pulled back to within 2% of that breakout level
        i.e. 0.98 * bo_level <= close[i] <= 1.02 * bo_level
      - EMA9 > EMA21 > EMA50                                 (3-EMA alignment)

    Breakout detection is fully vectorized with numpy rolling-max.
    The "within 2% of the most recent qualifying breakout" check uses
    a forward-propagated breakout level array (O(n) scan).
    """
    lookback: int  = int(params["breakout_lookback"])
    vol_mult: float = params["vol_mult"]
    sl_mult: float  = params["sl_atr_mult"]
    tp_rr: float    = params["tp_rr"]

    s  = pd.Series(close)
    h  = pd.Series(high)
    lo = pd.Series(low)
    v  = pd.Series(volume)

    # ── Precompute all indicators — O(n) each ────────────────────────────────
    ema9  = _ema(s, 9).values
    ema21 = _ema(s, 21).values
    ema50 = _ema(s, 50).values
    atr_s = _atr(h, lo, s, 14).values

    # Rolling max of high over the previous `lookback` bars (exclusive of i)
    # shift(1) excludes bar i itself; .rolling(lookback) looks back lookback bars.
    prev_high = pd.Series(high).shift(1).rolling(lookback).max().values

    # Volume moving average (20-bar)
    vol_avg = v.rolling(20).mean().values

    # ── Step 1: Identify breakout bars vectorially ────────────────────────────
    # A breakout at bar i: close[i] > prev_high[i] AND volume[i] > vol_mult * avg
    with np.errstate(invalid="ignore"):
        is_breakout = (
            (close > prev_high) &
            (volume > vol_mult * vol_avg) &
            np.isfinite(prev_high) &
            np.isfinite(vol_avg)
        )                                              # shape (n,) bool

    # breakout_level[i] = close[i] if is_breakout[i], else NaN
    breakout_level = np.where(is_breakout, close, np.nan)

    # ── Step 2: For each bar i, find the most recent breakout in [i-10, i-3]
    #   Then check whether close[i] is within 2% of that level.
    #
    #   We need: "is there a breakout at any lag in [3,10]?"
    #   We handle this vectorially:
    #     - Compute a "best breakout level for lag k" array for k=3..10
    #     - Take the first (closest) valid one across lags.
    #
    #   This is O(n * 8) = O(n) since 8 lags is a constant.
    # ─────────────────────────────────────────────────────────────────────────
    # recent_bo_level[i] = breakout level of the most recent qualifying
    # breakout in the window [i-10 .. i-3], or NaN if none.
    recent_bo_level = np.full(n, np.nan, dtype=float)
    for lag in range(3, 11):           # 3, 4, ..., 10
        # shifted_level[i] = breakout_level[i - lag]
        shifted = np.roll(breakout_level, lag)
        shifted[:lag] = np.nan
        # Fill only where we haven't found a closer breakout yet
        not_filled = np.isnan(recent_bo_level)
        valid = np.isfinite(shifted)
        recent_bo_level = np.where(not_filled & valid, shifted, recent_bo_level)

    # ── Step 3: Pullback condition (vectorised) ───────────────────────────────
    with np.errstate(invalid="ignore"):
        pullback_ok = (
            np.isfinite(recent_bo_level) &
            (close >= recent_bo_level * 0.98) &
            (close <= recent_bo_level * 1.02)
        )

    # ── Step 4: 3-EMA alignment ───────────────────────────────────────────────
    three_ema = (ema9 > ema21) & (ema21 > ema50)

    # ── ATR-based SL/TP ───────────────────────────────────────────────────────
    atr_pct = np.where(close > 0, atr_s / close, 0.0)
    # Alpha uses tighter 1.5x ATR stop (per spec), capped to [0.035, 0.09]
    sl_arr  = np.clip(atr_pct * 1.5, 0.035, 0.09)
    tp_arr  = sl_arr * tp_rr

    # ── Combined buy signal ───────────────────────────────────────────────────
    buy = pullback_ok & three_ema

    return buy, sl_arr, tp_arr


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

def run_vectorized(
    df: pd.DataFrame,
    symbol: str,
    bot_type: str,               # "memo" | "main" | "alpha"
    capital: float,
    params: dict,
    warmup: int = 55,
) -> tuple[list[dict], dict]:
    """
    Fast O(n) vectorized backtesting engine.

    Parameters
    ----------
    df        : OHLCV DataFrame with columns open/high/low/close/volume
    symbol    : ticker label (stored in each trade dict)
    bot_type  : "memo" | "main" | "alpha"
    capital   : starting capital in USD
    params    : strategy parameters (merged with defaults for missing keys)
    warmup    : number of leading bars to skip before trading

    Design
    ------
    Phase 1 — Precompute: All indicator series are built once on the full df.
              This is O(n) per indicator, O(n) total.

    Phase 2 — Signal arrays: Boolean buy/SL/TP arrays are built entirely with
              numpy/pandas vectorised ops (no Python loop over bars).

    Phase 3 — Bar loop: Iterates once over [warmup, n).  At each bar it only
              reads precomputed scalar values by integer index — O(1) per bar,
              O(n) total.  No `iloc[:i+1]` slicing, no per-bar indicator
              recomputation.

    Phase 4 — Equity / statistics: O(n) vectorised pass over the trade list
              using numpy scatter operations (no nested loops).

    Returns
    -------
    (trades, summary)
    trades  : list of dicts, one per closed trade, keys:
                symbol, entry_idx, entry_price, exit_idx, exit_price,
                pnl, reason, sl, tp
    summary : performance metrics dict with keys:
                total_return_pct, win_rate, max_drawdown_pct, sharpe,
                sortino, profit_factor, max_consecutive_losses, total_trades
    """
    if bot_type not in _DEFAULTS:
        raise ValueError(
            f"Unknown bot_type '{bot_type}'. Choose from: {list(_DEFAULTS)}"
        )

    # ── Merge caller params over defaults ────────────────────────────────────
    p = {**_DEFAULTS[bot_type], **params}

    # ── Extract raw numpy arrays once — avoids repeated Series overhead ───────
    # Phase 1: O(n) extraction
    close  = df["close"].to_numpy(dtype=float)
    high   = df["high"].to_numpy(dtype=float)
    low    = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n      = len(close)

    if n < warmup + 5:
        return [], dict(
            total_return_pct=0.0, win_rate=0.0, max_drawdown_pct=0.0,
            sharpe=0.0, sortino=0.0, profit_factor=0.0,
            max_consecutive_losses=0, total_trades=0,
        )

    # ── Phase 2: Precompute ALL indicator series upfront — O(n) total ─────────
    # Signal functions build all indicator arrays internally on the full dataset.
    # Each indicator (EMA, RSI, ATR, OBV) is computed exactly once via pandas
    # ewm/rolling — O(n) per indicator.
    if bot_type == "memo":
        buy_signal, sl_arr, tp_arr = _signals_memo(
            close, high, low, volume, p, n
        )
    elif bot_type == "main":
        buy_signal, sl_arr, tp_arr = _signals_main(
            close, high, low, volume, p, n
        )
    else:  # alpha
        buy_signal, sl_arr, tp_arr = _signals_alpha(
            close, high, low, volume, p, n
        )

    # ── Phase 3: Bar loop — O(n), reads precomputed scalars by index ──────────
    # No iloc[:i+1], no per-bar recomputation.
    # Max 2 concurrent positions globally is managed at the caller level;
    # this engine handles one symbol → at most 1 position for this symbol.
    trades: list[dict] = []
    cash = float(capital)

    in_position = False
    entry_idx   = 0
    entry_price = 0.0
    qty         = 0.0
    sl_price    = 0.0
    tp_price    = 0.0

    for i in range(warmup, n):
        price    = close[i]
        bar_low  = low[i]
        bar_high = high[i]

        if in_position:
            # ── Exit logic: check SL/TP against this bar's low/high ─────────
            hit_sl = bar_low  <= sl_price
            hit_tp = bar_high >= tp_price

            if hit_sl or hit_tp:
                # Conservative: if both trigger on the same bar, SL wins
                if hit_sl:
                    exit_price = sl_price
                    reason     = "stop_loss"
                else:
                    exit_price = tp_price
                    reason     = "take_profit"

                pnl   = (exit_price - entry_price) * qty
                cash += exit_price * qty          # return proceeds to cash

                trades.append(dict(
                    symbol      = symbol,
                    entry_idx   = entry_idx,
                    entry_price = entry_price,
                    exit_idx    = i,
                    exit_price  = exit_price,
                    pnl         = pnl,
                    reason      = reason,
                    sl          = sl_price,
                    tp          = tp_price,
                ))
                in_position = False
            # Skip new-entry check this bar (whether we closed or not; if we
            # just closed we could in theory re-enter, but for safety we skip)
            continue

        # ── Entry logic: only if a buy signal is precomputed at this bar ────
        if buy_signal[i]:
            sl_p     = float(sl_arr[i])
            tp_p     = float(tp_arr[i])
            qty_new  = _size_position(cash, price, sl_p, risk_pct=0.02)
            cost     = price * qty_new
            if qty_new > 0 and cost <= cash:
                cash       -= cost
                in_position = True
                entry_idx   = i
                entry_price = price
                qty         = qty_new
                sl_price    = price * (1.0 - sl_p)
                tp_price    = price * (1.0 + tp_p)

    # ── Force-close any position still open at the last bar ──────────────────
    if in_position:
        exit_price = close[-1]
        pnl        = (exit_price - entry_price) * qty
        cash      += exit_price * qty
        trades.append(dict(
            symbol      = symbol,
            entry_idx   = entry_idx,
            entry_price = entry_price,
            exit_idx    = n - 1,
            exit_price  = exit_price,
            pnl         = pnl,
            reason      = "end_of_data",
            sl          = sl_price,
            tp          = tp_price,
        ))

    # ── Phase 4: Summary statistics — O(n), fully vectorised ─────────────────
    summary = _compute_summary(trades, capital, close, n)

    return trades, summary
