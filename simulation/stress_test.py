"""
多市场、多周期压力测试框架
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试场景:
  bull   — 牛市 (强劲上涨，回调买机)
  bear   — 熊市 (持续下跌，考验止损)
  chop   — 震荡 (横盘整理，假信号多)
  crash  — 暴跌 (突发性崩盘，测试风控)
  pump   — 暴涨 (急速拉升后回调)
  mixed  — 混合行情 (真实市场最接近)

输出:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - 最大连续亏损次数
  - 平均盈亏比
  - 胜率
  - 最大回撤
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from simulation.market_sim import MarketProfile, generate_ohlcv


REGIME_CONFIGS = {
    "bull": dict(drift=1.5, annual_vol=0.80, regime_change_prob=0.01,
                 jump_prob=0.02, jump_magnitude=0.04, mean_reversion=0.05),
    "bear": dict(drift=-0.8, annual_vol=0.90, regime_change_prob=0.01,
                 jump_prob=0.03, jump_magnitude=0.06, mean_reversion=0.03),
    "chop": dict(drift=0.05, annual_vol=0.60, regime_change_prob=0.10,
                 jump_prob=0.01, jump_magnitude=0.02, mean_reversion=0.30),
    "crash": dict(drift=-2.0, annual_vol=2.00, regime_change_prob=0.05,
                  jump_prob=0.08, jump_magnitude=0.12, mean_reversion=0.05),
    "pump":  dict(drift=3.0, annual_vol=1.50, regime_change_prob=0.04,
                  jump_prob=0.10, jump_magnitude=0.15, mean_reversion=0.02),
    "mixed": dict(drift=0.50, annual_vol=0.80, regime_change_prob=0.04,
                  jump_prob=0.04, jump_magnitude=0.07, mean_reversion=0.08),
}


def make_regime_data(regime: str, profile_type: str, symbols: List[str],
                     n_bars: int, seed: int) -> Dict[str, pd.DataFrame]:
    """生成特定行情场景的市场数据."""
    np.random.seed(seed)
    base_prices = {"memecoin": 0.05, "mainstream": 10000.0, "alpha": 15.0}
    vol_mult = {"memecoin": 2.5, "mainstream": 1.0, "alpha": 1.5}

    cfg = REGIME_CONFIGS[regime].copy()
    cfg["annual_vol"] *= vol_mult[profile_type]

    result = {}
    for i, sym in enumerate(symbols):
        p = MarketProfile(
            name=profile_type,
            base_price=base_prices[profile_type] * np.random.uniform(0.5, 3.0),
            annual_vol=cfg["annual_vol"] * np.random.uniform(0.8, 1.2),
            drift=cfg["drift"],
            jump_prob=cfg["jump_prob"],
            jump_magnitude=cfg["jump_magnitude"],
            mean_reversion=cfg["mean_reversion"],
            regime_change_prob=cfg["regime_change_prob"],
        )
        result[sym] = generate_ohlcv(p, n_bars=n_bars, seed=seed + i * 7)
    return result


@dataclass
class BotMetrics:
    bot_name: str
    regime: str
    seed: int
    return_pct: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    max_consecutive_losses: int
    avg_win: float
    avg_loss: float
    profit_factor: float


def compute_metrics(history: list, starting_capital: float, risk_free: float = 0.04) -> dict:
    """Compute Sharpe, Sortino, Calmar, consecutive losses from trade history."""
    if not history:
        return {"sharpe": 0, "sortino": 0, "calmar": 0,
                "max_consec_loss": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0}

    pnls = [t.pnl for t in history]
    returns = [p / starting_capital for p in pnls]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (999 if wins else 0)

    # Sharpe (annualized per-trade)
    if len(returns) > 1:
        std = np.std(returns, ddof=1)
        mean_r = np.mean(returns)
        rf_per_trade = risk_free / 252
        sharpe = (mean_r - rf_per_trade) / std * np.sqrt(252) if std > 0 else 0
    else:
        sharpe = 0

    # Sortino (only downside deviation)
    neg_returns = [r for r in returns if r < 0]
    if len(neg_returns) > 1:
        downside_std = np.std(neg_returns, ddof=1)
        sortino = (np.mean(returns) - risk_free/252) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    else:
        sortino = 0

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for p in pnls:
        if p < 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": 0,  # computed after knowing max_drawdown
        "max_consec_loss": max_consec,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
    }
