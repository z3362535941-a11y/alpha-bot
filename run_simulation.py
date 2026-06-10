#!/usr/bin/env python3
"""
Alpha-Bot 三机器人联合模拟试跑
总资金: $10,000 USDT (paper trading)
- MemeBot   20% = $2,000
- MainBot   50% = $5,000
- AlphaBot  30% = $3,000

模式: 多场景蒙特卡洛 (5个随机种子) → 平均结果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from colorama import Fore, Style, init
from tabulate import tabulate

from config.settings import (
    Settings, MEMECOIN_WATCHLIST, MAINSTREAM_WATCHLIST, ALPHA_WATCHLIST
)
from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from simulation.market_sim import generate_multi_asset
from bots.memecoin_bot import MemeBot
from bots.mainstream_bot import MainstreamBot
from bots.alpha_bot import AlphaBot

init(autoreset=True)


def banner():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║       ALPHA-BOT  三机器人自动交易系统  v1.0                  ║
║       模式: PAPER TRADING — 5场景蒙特卡洛模拟               ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")


def print_section(title: str):
    print(f"\n{Fore.YELLOW}{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}{Style.RESET_ALL}")


def run_one_scenario(seed: int, settings: Settings, n_bars: int, verbose: bool = False) -> dict:
    """Run one complete simulation scenario with the given seed."""
    np.random.seed(seed)

    meme_data = generate_multi_asset(
        MEMECOIN_WATCHLIST[:5], "memecoin", n_bars,
        base_price_range=(0.001, 1.0),
    )
    main_data = generate_multi_asset(
        MAINSTREAM_WATCHLIST[:5], "mainstream", n_bars,
        base_price_range=(500, 60000),
    )
    alpha_data = generate_multi_asset(
        ALPHA_WATCHLIST[:5], "alpha", n_bars,
        base_price_range=(1, 80),
    )

    cap_memo = settings.starting_capital * settings.allocation.memecoin
    cap_main = settings.starting_capital * settings.allocation.mainstream
    cap_alpha = settings.starting_capital * settings.allocation.alpha

    port_memo = Portfolio(starting_capital=cap_memo)
    port_main = Portfolio(starting_capital=cap_main)
    port_alpha = Portfolio(starting_capital=cap_alpha)

    risk_cfg = settings.risk
    rm_memo = RiskManager(risk_cfg, cap_memo)
    rm_main = RiskManager(risk_cfg, cap_main)
    rm_alpha = RiskManager(risk_cfg, cap_alpha)

    # Suppress logging for non-verbose runs
    import logging
    if not verbose:
        logging.getLogger("MemeBot").setLevel(logging.CRITICAL)
        logging.getLogger("MainBot").setLevel(logging.CRITICAL)
        logging.getLogger("AlphaBot").setLevel(logging.CRITICAL)

    memo_bot = MemeBot(cap_memo, rm_memo, port_memo)
    main_bot = MainstreamBot(cap_main, rm_main, port_main)
    alpha_bot = AlphaBot(cap_alpha, rm_alpha, port_alpha)

    import contextlib, io
    suppress = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

    with suppress:
        memo_r = memo_bot.run_backtest(meme_data)
        main_r = main_bot.run_backtest(main_data)
        alpha_r = alpha_bot.run_backtest(alpha_data)

    total_start = cap_memo + cap_main + cap_alpha
    total_end = memo_r["current_capital"] + main_r["current_capital"] + alpha_r["current_capital"]
    total_trades = memo_r["total_trades"] + main_r["total_trades"] + alpha_r["total_trades"]
    total_wins = memo_r["winning_trades"] + main_r["winning_trades"] + alpha_r["winning_trades"]

    return {
        "seed": seed,
        "memo": memo_r,
        "main": main_r,
        "alpha": alpha_r,
        "total_start": total_start,
        "total_end": total_end,
        "total_pnl": total_end - total_start,
        "total_return_pct": (total_end - total_start) / total_start * 100,
        "total_trades": total_trades,
        "win_rate": total_wins / total_trades * 100 if total_trades > 0 else 0,
        "max_dd": max(memo_r["max_drawdown_pct"], main_r["max_drawdown_pct"], alpha_r["max_drawdown_pct"]),
    }


def main():
    banner()
    settings = Settings()
    N_BARS = 2160  # 2160 小时 K 线 = 90天 (更真实的统计样本)
    SEEDS = [42, 137, 256, 512, 999]

    print(f"  资金分配: MemeBot=${settings.starting_capital * settings.allocation.memecoin:,.0f}"
          f" | MainBot=${settings.starting_capital * settings.allocation.mainstream:,.0f}"
          f" | AlphaBot=${settings.starting_capital * settings.allocation.alpha:,.0f}")
    print(f"  模拟长度: {N_BARS} 小时 K 线 (≈{N_BARS//24}天) × {len(SEEDS)} 场景")

    # ─── 多场景运行 ────────────────────────────────────────────
    print_section(f"运行 {len(SEEDS)} 个市场场景...")
    t0 = time.time()
    results = []
    for i, seed in enumerate(SEEDS):
        print(f"  场景 {i+1}/{len(SEEDS)} (seed={seed})...", end=" ", flush=True)
        r = run_one_scenario(seed, settings, N_BARS, verbose=False)
        results.append(r)
        pnl_color = Fore.GREEN if r["total_pnl"] >= 0 else Fore.RED
        print(f"{pnl_color}{r['total_return_pct']:+.1f}%{Style.RESET_ALL} "
              f"胜率={r['win_rate']:.0f}% 交易={r['total_trades']}")

    elapsed = time.time() - t0

    # ─── 详细展示最后一个场景 ───────────────────────────────────
    last = results[-1]
    print_section(f"场景详情 (seed={SEEDS[-1]})")

    def fmt_bot(name, r, color):
        pnl_c = Fore.GREEN if r["total_pnl"] >= 0 else Fore.RED
        print(f"  {color}【{name}】{Style.RESET_ALL}")
        print(f"  起始资金:  ${r['starting_capital']:>10,.2f}")
        print(f"  当前资金:  ${r['current_capital']:>10,.2f}")
        print(f"  总盈亏:    {pnl_c}${r['total_pnl']:>+10,.2f}  ({r['return_pct']:+.1f}%){Style.RESET_ALL}")
        print(f"  胜率:      {Fore.CYAN}{r['win_rate_pct']:>6.1f}%{Style.RESET_ALL}  "
              f"({r['winning_trades']}W / {r['losing_trades']}L / {r['total_trades']}T)")
        print(f"  最大回撤:  {Fore.RED}{r['max_drawdown_pct']:>6.2f}%{Style.RESET_ALL}")

    fmt_bot("Memecoin Bot", last["memo"], Fore.MAGENTA)
    fmt_bot("Mainstream Bot", last["main"], Fore.BLUE)
    fmt_bot("Alpha Bot", last["alpha"], Fore.GREEN)

    # ─── 多场景汇总表 ──────────────────────────────────────────
    print_section("多场景汇总 (5场景平均)")
    table_rows = []
    for r in results:
        pnl_sign = "+" if r["total_pnl"] >= 0 else ""
        table_rows.append([
            f"seed={r['seed']}",
            f"${r['total_end']:,.2f}",
            f"{pnl_sign}{r['total_return_pct']:.1f}%",
            f"{r['win_rate']:.0f}%",
            r["total_trades"],
            f"{r['max_dd']:.1f}%",
        ])

    avg_ret = np.mean([r["total_return_pct"] for r in results])
    avg_wr = np.mean([r["win_rate"] for r in results])
    avg_dd = np.mean([r["max_dd"] for r in results])
    avg_end = np.mean([r["total_end"] for r in results])
    avg_trades = int(np.mean([r["total_trades"] for r in results]))
    best = max(results, key=lambda x: x["total_return_pct"])
    worst = min(results, key=lambda x: x["total_return_pct"])

    table_rows.append(["─"*10, "─"*12, "─"*9, "─"*6, "─"*7, "─"*7])
    avg_sign = "+" if avg_ret >= 0 else ""
    table_rows.append([
        "AVERAGE",
        f"${avg_end:,.2f}",
        f"{avg_sign}{avg_ret:.1f}%",
        f"{avg_wr:.0f}%",
        avg_trades,
        f"{avg_dd:.1f}%",
    ])

    headers = ["场景", "最终资金", "收益率", "胜率", "交易数", "最大回撤"]
    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))

    avg_pnl = avg_end - 10000
    avg_color = Fore.GREEN if avg_pnl >= 0 else Fore.RED

    print(f"""
  平均总盈亏:    {avg_color}${avg_pnl:>+,.2f}  ({avg_ret:+.1f}%){Style.RESET_ALL}
  平均胜率:      {Fore.CYAN}{avg_wr:.1f}%{Style.RESET_ALL}
  最佳场景:      {Fore.GREEN}+{best['total_return_pct']:.1f}%{Style.RESET_ALL} (seed={best['seed']})
  最差场景:      {Fore.RED}{worst['total_return_pct']:+.1f}%{Style.RESET_ALL} (seed={worst['seed']})
  平均最大回撤:  {Fore.RED}{avg_dd:.1f}%{Style.RESET_ALL}
  模拟总耗时:    {elapsed:.2f}s
""")

    grade_map = [
        (15, "A+", Fore.GREEN), (8, "A", Fore.GREEN),
        (3, "B", Fore.YELLOW), (0, "C", Fore.YELLOW), (-999, "D", Fore.RED)
    ]
    grade, grade_color = next((g, c) for t, g, c in grade_map if avg_ret >= t)
    print(f"  综合评分: {grade_color}{grade}{Style.RESET_ALL}\n")

    # ─── 机器人分项平均 ─────────────────────────────────────────
    print_section("各机器人跨场景平均表现")
    bot_rows = [
        ["MemeBot",  f"+{np.mean([r['memo']['return_pct'] for r in results]):.1f}%",
         f"{np.mean([r['memo']['win_rate_pct'] for r in results]):.0f}%",
         f"{int(np.mean([r['memo']['total_trades'] for r in results]))}"],
        ["MainBot",  f"+{np.mean([r['main']['return_pct'] for r in results]):.1f}%",
         f"{np.mean([r['main']['win_rate_pct'] for r in results]):.0f}%",
         f"{int(np.mean([r['main']['total_trades'] for r in results]))}"],
        ["AlphaBot", f"+{np.mean([r['alpha']['return_pct'] for r in results]):.1f}%",
         f"{np.mean([r['alpha']['win_rate_pct'] for r in results]):.0f}%",
         f"{int(np.mean([r['alpha']['total_trades'] for r in results]))}"],
    ]
    print(tabulate(bot_rows, headers=["Bot", "平均收益率", "平均胜率", "平均交易数"],
                   tablefmt="rounded_outline"))

    # ─── 保存 ────────────────────────────────────────────────────
    import json
    from core.logger import TRADE_LOG
    os.makedirs("logs", exist_ok=True)
    with open("logs/simulation_summary.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("memo", "main", "alpha")}
                   for r in results], f, indent=2)
    print(f"\n  模拟汇总已保存: logs/simulation_summary.json\n")


if __name__ == "__main__":
    main()
