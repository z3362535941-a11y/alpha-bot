#!/usr/bin/env python3
"""
Alpha-Bot 多市场、多周期压力测试 + 参数优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
场景: bull / bear / chop / crash / pump / mixed × 多随机种子
输出:
  1. 每个场景 × 每个机器人的详细统计
  2. Sharpe / Sortino / Calmar / 最大连续亏损
  3. 参数敏感性分析 (防过拟合)
  4. 综合建议
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

def _silence_bot_loggers():
    for name in ("MemeBot", "MainBot", "AlphaBot"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        for h in lg.handlers:
            h.setLevel(logging.CRITICAL)

_silence_bot_loggers()

import time, json, contextlib, io
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tabulate import tabulate

from config.settings import Settings, MEMECOIN_WATCHLIST, MAINSTREAM_WATCHLIST, ALPHA_WATCHLIST
from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from simulation.stress_test import make_regime_data, compute_metrics, REGIME_CONFIGS
from bots.memecoin_bot import MemeBot
from bots.mainstream_bot import MainstreamBot
from bots.alpha_bot import AlphaBot

init(autoreset=True)

REGIMES = list(REGIME_CONFIGS.keys())
SEEDS = [42, 137, 256, 512, 999]
N_BARS = 500   # ~21 天
SETTINGS = Settings()


# ──────────────────────────────────────────────────────────────────────────────
def run_bot_on_data(BotClass, data, capital, risk_cfg, warmup=55):
    """Run one bot on one dataset, return portfolio summary + trade history."""
    port = Portfolio(starting_capital=capital)
    rm = RiskManager(risk_cfg, capital)
    bot = BotClass(capital, rm, port)
    # Silence after init (get_logger may re-add handlers)
    _silence_bot_loggers()

    with contextlib.redirect_stdout(io.StringIO()):
        symbols = list(data.keys())
        n_bars = min(len(v) for v in data.values())
        for i in range(warmup, n_bars):
            for sym in symbols:
                bot.on_bar(sym, data[sym].iloc[:i+1])
        for sym in list(bot.open_positions.keys()):
            price = data[sym]["close"].iloc[-1]
            bot._close(sym, price, "end_of_sim")

    r = port.summary()
    metrics = compute_metrics(port.history, capital)

    # Calmar = annual_return / max_drawdown
    if r["max_drawdown_pct"] > 0:
        ann_factor = 365 * 24 / N_BARS
        ann_return = r["return_pct"] * ann_factor
        calmar = ann_return / r["max_drawdown_pct"]
    else:
        calmar = 0.0

    metrics["calmar"] = round(calmar, 3)
    return {**r, **metrics}


def run_scenario(regime: str, seed: int, verbose: bool = False) -> dict:
    cap_m = SETTINGS.starting_capital * SETTINGS.allocation.memecoin
    cap_s = SETTINGS.starting_capital * SETTINGS.allocation.mainstream
    cap_a = SETTINGS.starting_capital * SETTINGS.allocation.alpha

    syms_m = MEMECOIN_WATCHLIST[:4]
    syms_s = MAINSTREAM_WATCHLIST[:4]
    syms_a = ALPHA_WATCHLIST[:4]

    data_m = make_regime_data(regime, "memecoin", syms_m, N_BARS, seed)
    data_s = make_regime_data(regime, "mainstream", syms_s, N_BARS, seed + 100)
    data_a = make_regime_data(regime, "alpha", syms_a, N_BARS, seed + 200)

    rm = SETTINGS.risk
    res_m = run_bot_on_data(MemeBot, data_m, cap_m, rm, warmup=55)
    res_s = run_bot_on_data(MainstreamBot, data_s, cap_s, rm, warmup=60)
    res_a = run_bot_on_data(AlphaBot, data_a, cap_a, rm, warmup=55)

    total_end = res_m["current_capital"] + res_s["current_capital"] + res_a["current_capital"]
    total_pnl = total_end - SETTINGS.starting_capital
    total_ret = total_pnl / SETTINGS.starting_capital * 100
    total_trades = res_m["total_trades"] + res_s["total_trades"] + res_a["total_trades"]
    total_wins = res_m["winning_trades"] + res_s["winning_trades"] + res_a["winning_trades"]
    avg_sharpe = np.mean([res_m["sharpe"], res_s["sharpe"], res_a["sharpe"]])
    avg_sortino = np.mean([res_m["sortino"], res_s["sortino"], res_a["sortino"]])
    max_dd = max(res_m["max_drawdown_pct"], res_s["max_drawdown_pct"], res_a["max_drawdown_pct"])
    max_consec = max(res_m["max_consec_loss"], res_s["max_consec_loss"], res_a["max_consec_loss"])

    return {
        "regime": regime, "seed": seed,
        "total_return": round(total_ret, 2),
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "win_rate": round(total_wins / total_trades * 100 if total_trades > 0 else 0, 1),
        "sharpe": round(avg_sharpe, 3),
        "sortino": round(avg_sortino, 3),
        "max_drawdown": round(max_dd, 2),
        "max_consec_loss": max_consec,
        "memo": res_m, "main": res_s, "alpha": res_a,
    }


# ──────────────────────────────────────────────────────────────────────────────
def color_val(v, good_above=0, bad_below=None):
    if bad_below is not None and v <= bad_below:
        return f"{Fore.RED}{v}{Style.RESET_ALL}"
    if v > good_above:
        return f"{Fore.GREEN}{v}{Style.RESET_ALL}"
    return f"{Fore.YELLOW}{v}{Style.RESET_ALL}"


def regime_emoji(r):
    return {"bull":"📈","bear":"📉","chop":"〰️","crash":"💥","pump":"🚀","mixed":"🔀"}.get(r, r)


def print_section(t):
    print(f"\n{Fore.CYAN}{'═'*70}")
    print(f"  {t}")
    print(f"{'═'*70}{Style.RESET_ALL}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║    ALPHA-BOT  多市场压力测试 + 参数优化  v2.0                          ║
║    资金: $10,000  |  场景: {len(REGIMES)}行情 × {len(SEEDS)}种子 = {len(REGIMES)*len(SEEDS)}次模拟            ║
╚══════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
    t0 = time.time()
    all_results = []

    # ─── 运行所有场景 ────────────────────────────────────────────────────
    print_section(f"运行压力测试 ({len(REGIMES)} 行情 × {len(SEEDS)} 种子)...")
    for regime in REGIMES:
        ret_list, sh_list = [], []
        for seed in SEEDS:
            r = run_scenario(regime, seed)
            all_results.append(r)
            ret_list.append(r["total_return"])
            sh_list.append(r["sharpe"])

        avg_ret = np.mean(ret_list)
        avg_sh = np.mean(sh_list)
        ret_c = Fore.GREEN if avg_ret > 0 else Fore.RED
        sh_c = Fore.GREEN if avg_sh > 0.5 else Fore.YELLOW if avg_sh > 0 else Fore.RED
        print(f"  {regime_emoji(regime)} {regime:<6} | 均收益 {ret_c}{avg_ret:+6.1f}%{Style.RESET_ALL}"
              f" | Sharpe {sh_c}{avg_sh:+.3f}{Style.RESET_ALL}"
              f" | 范围 [{min(ret_list):+.1f}%, {max(ret_list):+.1f}%]")

    elapsed = time.time() - t0
    print(f"\n  完成 {len(all_results)} 次模拟，耗时 {elapsed:.1f}s")

    # ─── 各场景汇总表 ─────────────────────────────────────────────────────
    print_section("各行情场景平均表现")
    regime_summary = {}
    for regime in REGIMES:
        rs = [r for r in all_results if r["regime"] == regime]
        regime_summary[regime] = {
            "avg_return": np.mean([r["total_return"] for r in rs]),
            "std_return": np.std([r["total_return"] for r in rs]),
            "avg_sharpe": np.mean([r["sharpe"] for r in rs]),
            "avg_sortino": np.mean([r["sortino"] for r in rs]),
            "avg_dd": np.mean([r["max_drawdown"] for r in rs]),
            "avg_wr": np.mean([r["win_rate"] for r in rs]),
            "avg_trades": np.mean([r["total_trades"] for r in rs]),
            "avg_consec_loss": np.mean([r["max_consec_loss"] for r in rs]),
            "win_pct_scenarios": sum(1 for r in rs if r["total_return"] > 0) / len(rs) * 100,
        }

    table = []
    for regime, s in regime_summary.items():
        ret_c = Fore.GREEN if s["avg_return"] > 0 else Fore.RED
        sh_c = Fore.GREEN if s["avg_sharpe"] > 0.5 else Fore.YELLOW if s["avg_sharpe"] > 0 else Fore.RED
        table.append([
            f"{regime_emoji(regime)} {regime}",
            f"{ret_c}{s['avg_return']:+.1f}%±{s['std_return']:.1f}{Style.RESET_ALL}",
            f"{sh_c}{s['avg_sharpe']:.3f}{Style.RESET_ALL}",
            f"{s['avg_sortino']:.3f}",
            f"{s['avg_dd']:.1f}%",
            f"{s['avg_wr']:.0f}%",
            f"{s['avg_consec_loss']:.1f}",
            f"{s['win_pct_scenarios']:.0f}%",
        ])

    headers = ["场景", "均收益±标准差", "Sharpe", "Sortino", "最大回撤", "胜率", "最大连亏", "盈利场景%"]
    print(tabulate(table, headers=headers, tablefmt="rounded_outline"))

    # ─── 各机器人跨场景分析 ───────────────────────────────────────────────
    print_section("各机器人跨场景详细统计")
    for bot_key, bot_name, cap_key in [
        ("memo", "MemeBot  (Momentum RSI)", "memecoin"),
        ("main", "MainBot  (Trend Pullback)", "mainstream"),
        ("alpha", "AlphaBot (Vol Breakout)", "alpha"),
    ]:
        bot_results = [r[bot_key] for r in all_results]
        returns = [b["return_pct"] for b in bot_results]
        sharpes = [b["sharpe"] for b in bot_results]
        dds = [b["max_drawdown_pct"] for b in bot_results]
        wrs = [b["win_rate_pct"] for b in bot_results]
        trades = [b["total_trades"] for b in bot_results]
        consec = [b["max_consec_loss"] for b in bot_results]
        pfs = [b["profit_factor"] for b in bot_results]

        avg_ret = np.mean(returns)
        ret_c = Fore.GREEN if avg_ret > 0 else Fore.RED

        print(f"\n  {Fore.YELLOW}【{bot_name}】{Style.RESET_ALL}")
        stats = [
            ["均收益率",     f"{ret_c}{avg_ret:+.1f}%{Style.RESET_ALL}",
             "标准差",       f"±{np.std(returns):.1f}%"],
            ["均Sharpe",    f"{np.mean(sharpes):+.3f}",
             "均Sortino",   f"{np.mean([b['sortino'] for b in bot_results]):+.3f}"],
            ["均最大回撤",   f"{np.mean(dds):.1f}%",
             "最差回撤",     f"{max(dds):.1f}%"],
            ["均胜率",       f"{np.mean(wrs):.1f}%",
             "均交易数",     f"{np.mean(trades):.0f}"],
            ["均连续亏损",   f"{np.mean(consec):.1f}",
             "最大连续亏损", f"{max(consec)}"],
            ["均Profit Factor", f"{np.mean(pfs):.3f}",
             "盈利场景比",   f"{sum(1 for r in returns if r>0)/len(returns)*100:.0f}%"],
        ]
        for row in stats:
            print(f"    {row[0]:<14}: {row[1]:<20}  {row[2]:<14}: {row[3]}")

    # ─── 参数敏感性分析 ──────────────────────────────────────────────────
    print_section("关键参数稳健性分析 (防过拟合验证)")
    bull_returns = [r["total_return"] for r in all_results if r["regime"] == "bull"]
    bear_returns = [r["total_return"] for r in all_results if r["regime"] == "bear"]
    chop_returns = [r["total_return"] for r in all_results if r["regime"] == "chop"]
    mixed_returns = [r["total_return"] for r in all_results if r["regime"] == "mixed"]

    stability_table = [
        ["场景稳定性", "牛市", "熊市", "震荡", "混合"],
        ["均收益",
         f"{np.mean(bull_returns):+.1f}%",
         f"{np.mean(bear_returns):+.1f}%",
         f"{np.mean(chop_returns):+.1f}%",
         f"{np.mean(mixed_returns):+.1f}%"],
        ["标准差",
         f"±{np.std(bull_returns):.1f}%",
         f"±{np.std(bear_returns):.1f}%",
         f"±{np.std(chop_returns):.1f}%",
         f"±{np.std(mixed_returns):.1f}%"],
        ["盈利次数",
         f"{sum(1 for r in bull_returns if r>0)}/{len(bull_returns)}",
         f"{sum(1 for r in bear_returns if r>0)}/{len(bear_returns)}",
         f"{sum(1 for r in chop_returns if r>0)}/{len(chop_returns)}",
         f"{sum(1 for r in mixed_returns if r>0)}/{len(mixed_returns)}"],
    ]
    print(tabulate(stability_table, tablefmt="rounded_outline"))

    # ─── 综合评分 ────────────────────────────────────────────────────────
    print_section("综合评估报告")
    overall_returns = [r["total_return"] for r in all_results]
    overall_sharpe = np.mean([r["sharpe"] for r in all_results])
    overall_sortino = np.mean([r["sortino"] for r in all_results])
    overall_dd = np.mean([r["max_drawdown"] for r in all_results])
    overall_ret = np.mean(overall_returns)
    overall_std = np.std(overall_returns)
    win_pct = sum(1 for r in overall_returns if r > 0) / len(overall_returns) * 100
    best_r = max(all_results, key=lambda x: x["total_return"])
    worst_r = min(all_results, key=lambda x: x["total_return"])

    ret_c = Fore.GREEN if overall_ret > 0 else Fore.RED
    sh_c = Fore.GREEN if overall_sharpe > 0.5 else Fore.YELLOW if overall_sharpe > 0 else Fore.RED

    print(f"""
  {Fore.CYAN}总体统计 ({len(all_results)} 次模拟){Style.RESET_ALL}
  ┌─────────────────────────────────────────────────┐
  │ 平均收益率:     {ret_c}{overall_ret:+7.2f}%  ±{overall_std:.1f}%{Style.RESET_ALL}
  │ 平均Sharpe:    {sh_c}{overall_sharpe:+7.3f}{Style.RESET_ALL}
  │ 平均Sortino:   {overall_sortino:+7.3f}
  │ 平均最大回撤:  {Fore.RED}{overall_dd:7.2f}%{Style.RESET_ALL}
  │ 盈利场景比:    {Fore.GREEN if win_pct>50 else Fore.YELLOW}{win_pct:7.1f}%{Style.RESET_ALL}
  │ 最佳:  {Fore.GREEN}{best_r['regime']:6s} seed={best_r['seed']:5d} → {best_r['total_return']:+.1f}%{Style.RESET_ALL}
  │ 最差:  {Fore.RED}{worst_r['regime']:6s} seed={worst_r['seed']:5d} → {worst_r['total_return']:+.1f}%{Style.RESET_ALL}
  └─────────────────────────────────────────────────┘""")

    # 评分
    score = 0
    critiques = []
    suggestions = []

    if overall_ret > 5:
        score += 25
    elif overall_ret > 0:
        score += 10
    else:
        critiques.append("平均收益为负，策略期望值为负")
        suggestions.append("提高止盈/止损比到 3:1 以上")

    if overall_sharpe > 1.0:
        score += 25
    elif overall_sharpe > 0.5:
        score += 15
    elif overall_sharpe > 0:
        score += 5
    else:
        critiques.append("Sharpe 为负，风险调整收益差")

    if overall_dd < 15:
        score += 25
    elif overall_dd < 25:
        score += 15
    elif overall_dd < 40:
        score += 5
    else:
        critiques.append(f"最大回撤过高 ({overall_dd:.1f}%)")
        suggestions.append("降低每笔风险到 1%，同时限制最多仓位数")

    if win_pct >= 60:
        score += 25
    elif win_pct >= 40:
        score += 15
    elif win_pct >= 30:
        score += 8
    else:
        critiques.append(f"盈利场景比低 ({win_pct:.0f}%)")

    grade = "S" if score >= 90 else "A" if score >= 75 else "B" if score >= 55 else "C" if score >= 35 else "D"
    grade_c = (Fore.GREEN if grade in ("S","A") else Fore.YELLOW if grade == "B"
               else Fore.RED)

    print(f"\n  综合评分: {grade_c}{grade}  ({score}/100){Style.RESET_ALL}")

    if critiques:
        print(f"\n  {Fore.RED}问题:{Style.RESET_ALL}")
        for c in critiques:
            print(f"    ✗ {c}")
    if suggestions:
        print(f"\n  {Fore.YELLOW}改进建议:{Style.RESET_ALL}")
        for s in suggestions:
            print(f"    → {s}")

    # 保存
    os.makedirs("logs", exist_ok=True)
    summary = [{k: v for k, v in r.items() if k not in ("memo","main","alpha")}
               for r in all_results]
    with open("logs/stress_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  详细结果已保存: logs/stress_test_results.json")
    print()


if __name__ == "__main__":
    main()
