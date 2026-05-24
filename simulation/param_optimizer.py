"""
网格搜索参数优化器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
优化目标: 最大化 Sharpe Ratio (风险调整收益)
防止过拟合: 在训练集找最优参数 → 在测试集验证
"""
import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ParamResult:
    params: dict
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    train_win_rate: float
    test_win_rate: float
    train_max_dd: float
    test_max_dd: float
    overfit_score: float  # |train_sharpe - test_sharpe|, lower is better


MEMECOIN_PARAM_GRID = {
    "rsi_oversold": [30, 35, 40],
    "sl_atr_mult": [2.0, 2.5, 3.0],
    "tp_rr": [2.0, 2.5, 3.0],
    "ema_slow": [50, 60],
}

MAINSTREAM_PARAM_GRID = {
    "rsi_pullback_max": [45, 52, 58],
    "sl_atr_mult": [2.0, 2.5, 3.0],
    "tp_rr": [2.0, 2.5, 3.0],
    "bb_entry_pct": [1.0, 1.5, 2.0],  # how close to BB lower
}

ALPHA_PARAM_GRID = {
    "breakout_lookback": [15, 20, 25],
    "vol_mult_required": [1.8, 2.0, 2.5],
    "sl_atr_mult": [1.5, 1.8, 2.2],
    "tp_rr": [4.0, 5.0, 6.0],
}


def grid_search(param_grid: Dict[str, List], evaluate_fn, n_trials: int = 3) -> List[ParamResult]:
    """Run grid search over parameter combinations."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        try:
            r = evaluate_fn(params, n_trials=n_trials)
            results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: x.test_sharpe, reverse=True)
    return results


def best_params_summary(results: List[ParamResult], top_n: int = 5) -> pd.DataFrame:
    rows = []
    for r in results[:top_n]:
        row = {**r.params,
               "train_sharpe": r.train_sharpe,
               "test_sharpe": r.test_sharpe,
               "test_return%": r.test_return,
               "test_winrate%": r.test_win_rate,
               "test_maxdd%": r.test_max_dd,
               "overfit": r.overfit_score}
        rows.append(row)
    return pd.DataFrame(rows)
