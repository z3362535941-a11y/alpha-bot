"""
Alpha Bot — 寻找超额收益机会
策略：突破+背离+随机指标交叉
风险：单笔 2%，止损 ATR×1.8，止盈 3.5:1
"""
import pandas as pd
from bots.base_bot import BaseBot
from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from strategies import alpha_hunter


class AlphaBot(BaseBot):
    trailing_stop_pct: float = 0.04  # 4% trailing stop to lock in profits

    def __init__(self, capital: float, risk_manager: RiskManager, portfolio: Portfolio):
        super().__init__("AlphaBot", capital, risk_manager, portfolio)
        self.logger.info(f"AlphaBot initialized | capital=${capital:,.0f}")

    def get_signal(self, df: pd.DataFrame, symbol: str) -> alpha_hunter.Signal:
        return alpha_hunter.analyze(df, symbol)

    def run_backtest(self, data: dict[str, pd.DataFrame], warmup: int = 30):
        symbols = list(data.keys())
        n_bars = min(len(v) for v in data.values())

        self.logger.info(f"AlphaBot backtest | {len(symbols)} symbols | {n_bars} bars")

        for i in range(warmup, n_bars):
            for sym in symbols:
                df_slice = data[sym].iloc[:i + 1]
                self.on_bar(sym, df_slice)

        for sym in list(self.open_positions.keys()):
            price = data[sym]["close"].iloc[-1]
            self._close(sym, price, "end_of_simulation")

        return self.portfolio.summary()
