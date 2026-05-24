"""
Memecoin Bot — 专注短期动量爆发
策略：动量+成交量爆增+MACD交叉
风险：单笔 2%，止损 ATR自适应，止盈 3:1
"""
import pandas as pd
from bots.base_bot import BaseBot
from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from strategies import momentum


class MemeBot(BaseBot):
    max_positions: int = 2       # max 2 meme positions at once
    trailing_stop_pct: float = 0.05  # 5% trailing stop for memecoins

    def __init__(self, capital: float, risk_manager: RiskManager, portfolio: Portfolio):
        super().__init__("MemeBot", capital, risk_manager, portfolio)
        self.logger.info(f"MemeBot initialized | capital=${capital:,.0f}")

    def get_signal(self, df: pd.DataFrame, symbol: str) -> momentum.Signal:
        return momentum.analyze(df, symbol)

    def run_backtest(self, data: dict[str, pd.DataFrame], warmup: int = 30):
        symbols = list(data.keys())
        n_bars = min(len(v) for v in data.values())

        self.logger.info(f"MemeBot backtest | {len(symbols)} symbols | {n_bars} bars")

        for i in range(warmup, n_bars):
            for sym in symbols:
                df_slice = data[sym].iloc[:i + 1]
                self.on_bar(sym, df_slice)

        # Force close all open positions at end
        for sym in list(self.open_positions.keys()):
            price = data[sym]["close"].iloc[-1]
            self._close(sym, price, "end_of_simulation")

        return self.portfolio.summary()
