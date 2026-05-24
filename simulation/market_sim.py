"""
Realistic market simulator using Geometric Brownian Motion with:
- Regime switching (bull/bear/sideways)
- Volume correlation to price moves
- Fat tails for memecoin volatility
- Mean reversion component for mainstream
"""
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass


@dataclass
class MarketProfile:
    name: str
    base_price: float
    annual_vol: float      # annualized volatility
    drift: float           # annual drift
    jump_prob: float       # probability of volume spike
    jump_magnitude: float  # jump size multiplier
    mean_reversion: float  # 0=none, 1=strong
    regime_change_prob: float


PROFILES = {
    "memecoin": MarketProfile(
        name="memecoin",
        base_price=0.01,
        annual_vol=2.0,       # 200% annual vol
        drift=1.20,           # strong upward bias (memecoins in bull market)
        jump_prob=0.08,
        jump_magnitude=0.15,
        mean_reversion=0.03,
        regime_change_prob=0.04,
    ),
    "mainstream": MarketProfile(
        name="mainstream",
        base_price=50000.0,
        annual_vol=0.65,      # 65% annual vol
        drift=0.60,           # moderately bullish (crypto bull cycle)
        jump_prob=0.03,
        jump_magnitude=0.05,
        mean_reversion=0.10,
        regime_change_prob=0.02,
    ),
    "alpha": MarketProfile(
        name="alpha",
        base_price=10.0,
        annual_vol=1.10,      # 110% annual vol
        drift=0.90,           # bullish altcoin environment
        jump_prob=0.05,
        jump_magnitude=0.10,
        mean_reversion=0.05,
        regime_change_prob=0.03,
    ),
}


def generate_ohlcv(
    profile: MarketProfile,
    n_bars: int = 500,
    interval_hours: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    dt = interval_hours / (365 * 24)
    sqrt_dt = np.sqrt(dt)

    prices = [profile.base_price]
    volumes = []
    regime = 1  # 1=bull, -1=bear, 0=sideways

    for i in range(n_bars):
        # Regime switching: skewed toward bull (realistic bull cycle)
        if np.random.random() < profile.regime_change_prob:
            regime = np.random.choice([1, -1, 0], p=[0.60, 0.25, 0.15])

        drift_adj = profile.drift * regime * dt
        diffusion = profile.annual_vol * sqrt_dt * np.random.randn()

        # Fat tails via t-distribution for memecoins
        if profile.name == "memecoin":
            diffusion = profile.annual_vol * sqrt_dt * np.random.standard_t(df=3) * 0.7

        # Jump component
        jump = 0.0
        if np.random.random() < profile.jump_prob:
            jump = profile.jump_magnitude * np.random.choice([1, -1], p=[0.6, 0.4])
            jump *= np.random.exponential(1.0)

        # Mean reversion
        long_mean = profile.base_price * (1 + 0.5 * regime)
        mr = profile.mean_reversion * (long_mean - prices[-1]) / prices[-1] * dt

        ret = drift_adj + diffusion + jump + mr
        new_price = max(prices[-1] * (1 + ret), prices[-1] * 0.01)
        prices.append(new_price)

        # Volume: higher on big moves
        base_vol = abs(new_price - prices[-2]) / prices[-2]
        vol = (1_000_000 * (1 + base_vol * 20) *
               np.random.lognormal(0, 0.5) * (profile.base_price / 100 + 0.001))
        volumes.append(vol)

    prices = prices[1:]
    rows = []
    for i, (close, vol) in enumerate(zip(prices, volumes)):
        spread = close * 0.005 * (1 + np.random.exponential(0.5))
        high = close + abs(np.random.normal(0, spread))
        low = close - abs(np.random.normal(0, spread))
        open_ = prices[i - 1] if i > 0 else close
        rows.append({
            "open": open_,
            "high": max(high, open_, close),
            "low": min(low, open_, close),
            "close": close,
            "volume": vol,
        })

    df = pd.DataFrame(rows)
    df.index = pd.date_range(
        start="2024-01-01",
        periods=n_bars,
        freq=f"{int(interval_hours)}h",
    )
    return df


def generate_multi_asset(
    symbols: list[str],
    profile_type: str,
    n_bars: int = 500,
    base_price_range: Tuple[float, float] = (1.0, 100.0),
) -> dict[str, pd.DataFrame]:
    profile = PROFILES[profile_type]
    result = {}
    for i, sym in enumerate(symbols):
        p = MarketProfile(**vars(profile))
        p.base_price = np.random.uniform(*base_price_range)
        p.drift += np.random.normal(0, 0.1)
        p.annual_vol *= np.random.uniform(0.7, 1.4)
        result[sym] = generate_ohlcv(p, n_bars=n_bars, seed=42 + i)
    return result
