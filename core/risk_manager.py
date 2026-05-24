from dataclasses import dataclass, field
from typing import Optional
from config.settings import RiskConfig


@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    bot_name: str
    cost: float = 0.0
    highest_price: float = 0.0  # for trailing stop
    trailing_pct: float = 0.0   # trailing stop distance

    def __post_init__(self):
        self.cost = self.entry_price * self.qty
        self.highest_price = self.entry_price

    def update_trailing(self, current_price: float):
        if current_price > self.highest_price:
            self.highest_price = current_price
            if self.trailing_pct > 0:
                new_sl = current_price * (1 - self.trailing_pct)
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl

    def current_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.qty

    def pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price

    def should_stop_loss(self, price: float) -> bool:
        return price <= self.stop_loss

    def should_take_profit(self, price: float) -> bool:
        return price >= self.take_profit


class RiskManager:
    def __init__(self, config: RiskConfig, starting_capital: float):
        self.config = config
        self.peak_capital = starting_capital
        self.positions: dict[str, Position] = {}

    def update_peak(self, current_capital: float):
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

    def is_in_drawdown_limit(self, current_capital: float) -> bool:
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        return drawdown < self.config.max_portfolio_drawdown

    def position_size(self, capital: float, price: float,
                      volatility_factor: float = 1.0) -> float:
        """Kelly-inspired position sizing with volatility adjustment."""
        risk_amount = capital * self.config.max_risk_per_trade
        sl_distance = self.config.stop_loss_pct * volatility_factor
        max_loss_per_unit = price * sl_distance
        if max_loss_per_unit <= 0:
            return 0.0
        qty = risk_amount / max_loss_per_unit
        max_cost = capital * self.config.max_position_size
        max_qty_by_size = max_cost / price
        return min(qty, max_qty_by_size)

    def open_position(self, symbol: str, price: float, qty: float,
                      bot_name: str, sl_pct: float, tp_pct: float,
                      trailing_pct: float = 0.0) -> Optional[Position]:
        if symbol in self.positions:
            return None
        pos = Position(
            symbol=symbol,
            entry_price=price,
            qty=qty,
            stop_loss=price * (1 - sl_pct),
            take_profit=price * (1 + tp_pct),
            bot_name=bot_name,
            trailing_pct=trailing_pct,
        )
        self.positions[symbol] = pos
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        return self.positions.pop(symbol, None)

    def check_exits(self, symbol: str, current_price: float) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        pos.update_trailing(current_price)
        if pos.should_stop_loss(current_price):
            return "STOP_LOSS"
        if pos.should_take_profit(current_price):
            return "TAKE_PROFIT"
        return None
