from abc import ABC, abstractmethod
from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from core.logger import get_logger, log_trade
import pandas as pd


class BaseBot(ABC):
    max_positions: int = 3
    trailing_stop_pct: float = 0.0

    def __init__(self, name: str, capital: float, risk_manager: RiskManager, portfolio: Portfolio):
        self.name = name
        self.capital = capital
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        self.logger = get_logger(name)
        self.open_positions: dict = {}
        self._dd_warned: bool = False

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, symbol: str):
        ...

    def _total_equity(self, current_prices: dict) -> float:
        """Cash + mark-to-market value of all open positions."""
        pos_value = sum(
            pos.qty * current_prices.get(sym, pos.entry_price)
            for sym, pos in self.open_positions.items()
        )
        return self.portfolio.cash + pos_value

    def on_bar(self, symbol: str, df: pd.DataFrame):
        price = df["close"].iloc[-1]

        # Check exits first
        if symbol in self.open_positions:
            exit_reason = self.risk_manager.check_exits(symbol, price)
            if exit_reason:
                self._close(symbol, price, exit_reason)
                return

        if symbol in self.open_positions:
            return

        if len(self.open_positions) >= self.max_positions:
            return

        # Use total equity (cash + mark-to-market open positions) for drawdown gate
        equity = self._total_equity({symbol: price})
        self.risk_manager.update_peak(equity)
        self.portfolio.update_equity(equity)

        if not self.risk_manager.is_in_drawdown_limit(equity):
            if not self._dd_warned:
                self.logger.warning("Drawdown limit hit — pausing new entries")
                self._dd_warned = True
            return
        self._dd_warned = False

        sig = self.get_signal(df, symbol)
        if sig.action == "BUY":
            self._open(symbol, price, sig)

    def _open(self, symbol: str, price: float, sig):
        qty = self.risk_manager.position_size(
            self.portfolio.cash, price,
            volatility_factor=sig.sl_pct / 0.05,
        )
        if qty <= 0 or price * qty > self.portfolio.cash * 0.95:
            return
        pos = self.risk_manager.open_position(
            symbol, price, qty, self.name, sig.sl_pct, sig.tp_pct,
            trailing_pct=self.trailing_stop_pct,
        )
        if pos:
            self.portfolio.cash -= price * qty
            self.open_positions[symbol] = pos
            log_trade(self.name, "BUY", symbol, price, qty, reason=sig.reason)

    def _close(self, symbol: str, price: float, reason: str):
        pos = self.risk_manager.close_position(symbol)
        if not pos:
            return
        pnl = pos.current_pnl(price)
        # Return sale proceeds to cash (cost was deducted at open)
        self.portfolio.cash += price * pos.qty
        self.portfolio.record_trade(
            self.name, symbol, "SELL", price, pos.qty, pnl, reason
        )
        self.open_positions.pop(symbol, None)
        log_trade(self.name, "SELL", symbol, price, pos.qty, pnl, reason)
