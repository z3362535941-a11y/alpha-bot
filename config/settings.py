import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.015   # 1.5% per trade (was 2%)
    max_portfolio_drawdown: float = 0.12 # 12% max drawdown gate
    max_position_size: float = 0.15     # 15% max per position (was 20%)
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.125


@dataclass
class BotAllocation:
    memecoin: float = 0.20
    mainstream: float = 0.50
    alpha: float = 0.30


@dataclass
class Settings:
    trading_mode: str = "paper"
    starting_capital: float = 10000.0
    exchange: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    risk: RiskConfig = field(default_factory=RiskConfig)
    allocation: BotAllocation = field(default_factory=BotAllocation)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        risk = RiskConfig(
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", 0.015)),
            max_portfolio_drawdown=float(os.getenv("MAX_PORTFOLIO_RISK", 0.12)),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", 0.15)),
        )
        allocation = BotAllocation(
            memecoin=float(os.getenv("MEMECOIN_ALLOCATION", 0.20)),
            mainstream=float(os.getenv("MAINSTREAM_ALLOCATION", 0.50)),
            alpha=float(os.getenv("ALPHA_ALLOCATION", 0.30)),
        )
        return cls(
            trading_mode=os.getenv("TRADING_MODE", "paper"),
            starting_capital=float(os.getenv("STARTING_CAPITAL", 10000.0)),
            exchange=os.getenv("EXCHANGE", "binance"),
            api_key=os.getenv("API_KEY", ""),
            api_secret=os.getenv("API_SECRET", ""),
            risk=risk,
            allocation=allocation,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Watchlists
MEMECOIN_WATCHLIST = [
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK",
    "WIF", "MEME", "NEIRO", "MOG", "BRETT",
]

MAINSTREAM_WATCHLIST = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "AVAX", "DOT", "MATIC", "LINK",
]

ALPHA_WATCHLIST = [
    "SUI", "APT", "ARB", "OP", "INJ",
    "TIA", "SEI", "PYTH", "JTO", "W",
]
