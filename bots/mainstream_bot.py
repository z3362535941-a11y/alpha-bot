"""
Mainstream Coin Bot — 趋势跟随，高胜率
策略：EMA多周期对齐 + RSI超卖反弹 + 布林带支撑
风险：单笔 2%，止损 ATR×2，止盈 3:1+
"""
import pandas as pd
from bots.base_bot import BaseBot
from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from strategies import trend_following


class MainstreamBot(BaseBot):
    max_positions: int = 2       # max 2 mainstream positions at once
    trailing_stop_pct: float = 0.035  # 3.5% trailing stop

    def __init__(self, capital: float, risk_manager: RiskManager, portfolio: Portfolio):
        super().__init__("MainBot", capital, risk_manager, portfolio)
        self.logger.info(f"MainstreamBot initialized | capital=${capital:,.0f}")

    def get_signal(self, df: pd.DataFrame, symbol: str) -> trend_following.Signal:
        return trend_following.analyze(df, symbol)

    def run_backtest(self, data: dict[str, pd.DataFrame], warmup: int = 55):
        symbols = list(data.keys())
        n_bars = min(len(v) for v in data.values())

        self.logger.info(f"MainstreamBot backtest | {len(symbols)} symbols | {n_bars} bars")

        for i in range(warmup, n_bars):
            for sym in symbols:
                df_slice = data[sym].iloc[:i + 1]
                self.on_bar(sym, df_slice)

        for sym in list(self.open_positions.keys()):
            price = data[sym]["close"].iloc[-1]
            self._close(sym, price, "end_of_simulation")

        return self.portfolio.summary()
