from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime


@dataclass
class TradeRecord:
    timestamp: str
    bot: str
    symbol: str
    action: str
    price: float
    qty: float
    pnl: float
    reason: str
    capital_after: float


@dataclass
class Portfolio:
    starting_capital: float
    cash: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 0.0
    history: List[TradeRecord] = field(default_factory=list)
    bot_pnl: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.cash = self.starting_capital
        self.peak_value = self.starting_capital

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100

    @property
    def total_return_pct(self) -> float:
        return (self.cash - self.starting_capital) / self.starting_capital * 100

    def update_equity(self, equity: float):
        """Call mid-bar to track drawdown even with open positions."""
        if equity > self.peak_value:
            self.peak_value = equity
        drawdown = (self.peak_value - equity) / self.peak_value * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def record_trade(self, bot: str, symbol: str, action: str,
                     price: float, qty: float, pnl: float, reason: str):
        # Cash is already managed by _open/_close in BaseBot — do NOT touch cash here.
        self.total_pnl += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1

        if self.cash > self.peak_value:
            self.peak_value = self.cash

        drawdown = (self.peak_value - self.cash) / self.peak_value * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.bot_pnl[bot] = self.bot_pnl.get(bot, 0.0) + pnl

        self.history.append(TradeRecord(
            timestamp=datetime.now().isoformat(),
            bot=bot,
            symbol=symbol,
            action=action,
            price=price,
            qty=qty,
            pnl=pnl,
            reason=reason,
            capital_after=self.cash,
        ))

    def summary(self) -> dict:
        return {
            "starting_capital": self.starting_capital,
            "current_capital": round(self.cash, 2),
            "total_pnl": round(self.total_pnl, 2),
            "return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(self.win_rate, 1),
            "max_drawdown_pct": round(self.max_drawdown, 2),
            "bot_pnl": {k: round(v, 2) for k, v in self.bot_pnl.items()},
        }
