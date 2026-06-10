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
    highest_price: float = 0.0
    trailing_pct: float = 0.0
    # partial TP: first target at 1.5x SL to bank profits early
    partial_tp: float = 0.0
    partial_done: bool = False

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

    def should_partial_tp(self, price: float) -> bool:
        return not self.partial_done and self.partial_tp > 0 and price >= self.partial_tp


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
        """Risk-based position sizing: risk exactly max_risk_per_trade of capital per trade."""
        risk_amount = capital * self.config.max_risk_per_trade
        sl_distance = self.config.stop_loss_pct * max(0.5, volatility_factor)
        max_loss_per_unit = price * sl_distance
        if max_loss_per_unit <= 0:
            return 0.0
        qty = risk_amount / max_loss_per_unit
        # Cap by max position size
        max_cost = capital * self.config.max_position_size
        max_qty_by_size = max_cost / price
        return min(qty, max_qty_by_size)

    def open_position(self, symbol: str, price: float, qty: float,
                      bot_name: str, sl_pct: float, tp_pct: float,
                      trailing_pct: float = 0.0) -> Optional[Position]:
        if symbol in self.positions:
            return None
        # Partial TP at 1.5× SL distance (locks in half profit early)
        partial_tp = price * (1 + sl_pct * 1.5)
        pos = Position(
            symbol=symbol,
            entry_price=price,
            qty=qty,
            stop_loss=price * (1 - sl_pct),
            take_profit=price * (1 + tp_pct),
            bot_name=bot_name,
            trailing_pct=trailing_pct,
            partial_tp=partial_tp,
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
        if pos.should_partial_tp(current_price):
            # Mark partial done and move SL to breakeven
            pos.partial_done = True
            pos.stop_loss = max(pos.stop_loss, pos.entry_price * 1.001)
            return None  # don't fully exit, just tighten SL
        return None
