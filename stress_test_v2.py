#!/usr/bin/env python3
"""
Alpha-Bot 压力测试 v2.0 — 向量化回测 + 参数优化 + 防过拟合
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6 行情场景 × 5 随机种子 = 30 次完整模拟
参数优化: 训练集(3种子)最大化Sharpe → 测试集(2种子)验证防过拟合
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time, json, itertools
import numpy as np
from colorama import Fore, Style, init
from tabulate import tabulate

from simulation.stress_test import make_regime_data, REGIME_CONFIGS
from simulation.vectorized_backtest import run_vectorized
from config.settings import (MEMECOIN_WATCHLIST, MAINSTREAM_WATCHLIST, ALPHA_WATCHLIST)

init(autoreset=True)

# ── 全局配置 ───────────────────────────────────────────────────────────────────
REGIMES      = list(REGIME_CONFIGS.keys())          # 6 行情场景
TRAIN_SEEDS  = [42, 137, 256]
TEST_SEEDS   = [512, 999]
ALL_SEEDS    = TRAIN_SEEDS + TEST_SEEDS
N_BARS       = 480      # ~20 天
CAPITAL      = 10_000.0

SYMS   = {"memo": MEMECOIN_WATCHLIST[:3], "main": MAINSTREAM_WATCHLIST[:3], "alpha": ALPHA_WATCHLIST[:3]}
ALLOC  = {"memo": 0.20, "main": 0.50, "alpha": 0.30}
WARMUP = {"memo": 55,   "main": 60,   "alpha": 65}

# ── 参数搜索空间 ───────────────────────────────────────────────────────────────
GRIDS = {
    "memo":  [{"rsi_oversold": r, "sl_atr_mult": s, "tp_rr": t}
              for r in [35, 38, 42] for s in [2.0, 2.5, 3.0] for t in [2.0, 2.5, 3.0]],
    "main":  [{"rsi_pullback_max": r, "sl_atr_mult": s, "tp_rr": t}
              for r in [45, 52, 58] for s in [2.0, 2.5, 3.0] for t in [2.0, 2.5, 3.0]],
    "alpha": [{"breakout_lookback": b, "vol_mult": v, "sl_atr_mult": s, "tp_rr": t}
              for b in [15, 20] for v in [1.8, 2.0, 2.5] for s in [1.5, 2.0] for t in [4.0, 5.0]],
}


# ── 核心: 运行单一机器人在多个symbol上 ─────────────────────────────────────────
def run_bot(bot_key: str, data_dict: dict, params: dict) -> dict:
    """Run one bot on all its symbols; use per-symbol equity-curve Sharpe."""
    cap = CAPITAL * ALLOC[bot_key]
    cap_per_sym = cap / len(SYMS[bot_key])
    wm = WARMUP[bot_key]

    all_trades = []
    sym_sharpes, sym_sortinos = [], []

    for sym in SYMS[bot_key]:
        if sym not in data_dict:
            continue
        trades, summary = run_vectorized(
            data_dict[sym], sym, bot_key, cap_per_sym, params, warmup=wm
        )
        all_trades.extend(trades)
        # Use the engine's own equity-curve Sharpe (reliable even with few trades)
        if summary["total_trades"] > 0:
            sym_sharpes.append(summary["sharpe"])
            sym_sortinos.append(summary["sortino"])

    # Aggregate trade-level stats
    total_pnl = sum(t["pnl"] for t in all_trades)
    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    ret_pct = total_pnl / cap * 100

    # Drawdown from combined equity
    equity = cap; peak = cap; max_dd = 0.0
    for t in sorted(all_trades, key=lambda x: x["exit_idx"]):
        equity += t["pnl"]
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    # Max consecutive losses
    mc = cc = 0
    for t in sorted(all_trades, key=lambda x: x["exit_idx"]):
        if t["pnl"] <= 0: cc += 1; mc = max(mc, cc)
        else: cc = 0

    pf = (sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
          if losses else (999.0 if wins else 0.0))

    # Use mean of per-symbol Sharpes (equity-curve based, stable)
    sharpe  = float(np.mean(sym_sharpes))  if sym_sharpes  else 0.0
    sortino = float(np.mean(sym_sortinos)) if sym_sortinos else 0.0

    return {
        "return_pct":      round(ret_pct, 3),
        "win_rate_pct":    round(len(wins)/len(all_trades)*100 if all_trades else 0, 1),
        "sharpe":          round(sharpe, 3),
        "sortino":         round(sortino, 3),
        "max_drawdown":    round(max_dd, 2),
        "max_consec_loss": mc,
        "profit_factor":   round(pf, 3),
        "total_trades":    len(all_trades),
    }


def run_scenario(regime: str, seed: int, params: dict) -> dict:
    results = {}
    for bk in ("memo", "main", "alpha"):
        ptype = {"memo":"memecoin","main":"mainstream","alpha":"alpha"}[bk]
        data = make_regime_data(regime, ptype, SYMS[bk], N_BARS,
                                seed + {"memo":0,"main":100,"alpha":200}[bk])
        results[bk] = run_bot(bk, data, params.get(bk, {}))

    total_ret = sum(results[k]["return_pct"] * ALLOC[k] for k in results)
    avg_sharpe = np.mean([results[k]["sharpe"] for k in results])
    max_dd = max(results[k]["max_drawdown"] for k in results)
    max_mc = max(results[k]["max_consec_loss"] for k in results)
    total_trades = sum(results[k]["total_trades"] for k in results)
    win_rates = [results[k]["win_rate_pct"] for k in results if results[k]["total_trades"] > 0]
    avg_wr = np.mean(win_rates) if win_rates else 0

    return {
        "regime": regime, "seed": seed,
        "total_return": round(total_ret, 3),
        "sharpe": round(avg_sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "max_consec_loss": max_mc,
        "total_trades": total_trades,
        "avg_win_rate": round(avg_wr, 1),
        **{f"{k}_ret": results[k]["return_pct"] for k in results},
        **{f"{k}_sh":  results[k]["sharpe"] for k in results},
        **{f"{k}_wr":  results[k]["win_rate_pct"] for k in results},
        **{f"{k}_dd":  results[k]["max_drawdown"] for k in results},
        **{f"{k}_mc":  results[k]["max_consec_loss"] for k in results},
        **{f"{k}_pf":  results[k]["profit_factor"] for k in results},
    }


# ── 参数优化 ──────────────────────────────────────────────────────────────────
def optimize(regime: str = "mixed") -> dict:
    best_by_bot = {}
    for bk in ("memo", "main", "alpha"):
        ptype = {"memo":"memecoin","main":"mainstream","alpha":"alpha"}[bk]
        best_sh = -999.0
        best_p  = GRIDS[bk][0]
        for p in GRIDS[bk]:
            shs = []
            for seed in TRAIN_SEEDS:
                data = make_regime_data(regime, ptype, SYMS[bk], N_BARS,
                                        seed + {"memo":0,"main":100,"alpha":200}[bk])
                r = run_bot(bk, data, p)
                shs.append(r["sharpe"])
            avg = np.mean(shs)
            if avg > best_sh:
                best_sh = avg
                best_p  = p

        # Validate on test seeds
        test_shs = []
        for seed in TEST_SEEDS:
            data = make_regime_data(regime, ptype, SYMS[bk], N_BARS,
                                    seed + {"memo":0,"main":100,"alpha":200}[bk])
            r = run_bot(bk, data, best_p)
            test_shs.append(r["sharpe"])
        test_sh = np.mean(test_shs)
        best_by_bot[bk] = {"params": best_p, "train_sh": round(best_sh, 3),
                           "test_sh": round(test_sh, 3), "overfit": round(abs(best_sh-test_sh), 3)}
    return best_by_bot


# ── 输出辅助 ──────────────────────────────────────────────────────────────────
def section(t):
    print(f"\n{Fore.CYAN}{'═'*72}\n  {t}\n{'═'*72}{Style.RESET_ALL}")

REGIME_ICON = {"bull":"📈","bear":"📉","chop":"〰 ","crash":"💥","pump":"🚀","mixed":"🔀"}


def main():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║  ALPHA-BOT  压力测试 v2.0  |  均值回归 + 确认性入场 + 参数优化          ║
║  资金 $10,000  |  6行情×{len(ALL_SEEDS)}种子 = {len(REGIMES)*len(ALL_SEEDS)} 次模拟 + 参数网格搜索           ║
╚══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
    t0 = time.time()

    # ─── 1. 参数优化 ─────────────────────────────────────────────────────────
    section("参数优化 (mixed行情 | 训练3种子 → 测试2种子 防过拟合)")
    print("  搜索中...", end="", flush=True)
    opt = optimize("mixed")
    print(" 完成")

    opt_rows = []
    for bk, info in opt.items():
        overfit_c = Fore.GREEN if info["overfit"] < 0.5 else Fore.YELLOW if info["overfit"] < 1.0 else Fore.RED
        opt_rows.append([bk, str(info["params"]),
                         f"{info['train_sh']:+.3f}", f"{info['test_sh']:+.3f}",
                         f"{overfit_c}{info['overfit']:.3f}{Style.RESET_ALL}"])
    print(tabulate(opt_rows, headers=["Bot","最优参数","训练Sharpe","测试Sharpe","过拟合"],
                   tablefmt="rounded_outline"))

    best_params = {k: v["params"] for k, v in opt.items()}

    # ─── 2. 全场景压力测试 ──────────────────────────────────────────────────
    section(f"全场景压力测试 ({len(REGIMES)} × {len(ALL_SEEDS)} = {len(REGIMES)*len(ALL_SEEDS)} 次)")
    all_results = []
    for regime in REGIMES:
        rets = []
        for seed in ALL_SEEDS:
            r = run_scenario(regime, seed, best_params)
            all_results.append(r)
            rets.append(r["total_return"])
        avg = np.mean(rets)
        ret_c = Fore.GREEN if avg > 0 else Fore.RED
        wp = sum(1 for x in rets if x > 0) / len(rets) * 100
        print(f"  {REGIME_ICON[regime]} {regime:<6} | 均收益 {ret_c}{avg:+.1f}%{Style.RESET_ALL}"
              f" | 范围 [{min(rets):+.1f}%..{max(rets):+.1f}%] | 盈利场景 {wp:.0f}%")

    elapsed = time.time() - t0

    # ─── 3. 各场景汇总 ───────────────────────────────────────────────────────
    section("各行情场景平均表现")
    rows = []
    for regime in REGIMES:
        rs = [r for r in all_results if r["regime"] == regime]
        avg_ret = np.mean([r["total_return"] for r in rs])
        avg_sh  = np.mean([r["sharpe"] for r in rs])
        avg_wr  = np.mean([r["avg_win_rate"] for r in rs])
        avg_dd  = np.mean([r["max_drawdown"] for r in rs])
        avg_mc  = np.mean([r["max_consec_loss"] for r in rs])
        std_ret = np.std([r["total_return"] for r in rs])
        win_sc  = sum(1 for r in rs if r["total_return"] > 0) / len(rs) * 100
        ret_c = Fore.GREEN if avg_ret > 0 else Fore.RED
        sh_c  = (Fore.GREEN if avg_sh > 1.0 else Fore.YELLOW if avg_sh > 0.3
                 else Fore.YELLOW if avg_sh > 0 else Fore.RED)
        rows.append([
            f"{REGIME_ICON[regime]}{regime}",
            f"{ret_c}{avg_ret:+.1f}%±{std_ret:.1f}{Style.RESET_ALL}",
            f"{sh_c}{avg_sh:.3f}{Style.RESET_ALL}",
            f"{avg_wr:.0f}%",
            f"{avg_dd:.1f}%",
            f"{avg_mc:.0f}",
            f"{win_sc:.0f}%",
        ])
    print(tabulate(rows, headers=["场景","均收益±σ","Sharpe","均胜率","均回撤","均连亏","盈利%"],
                   tablefmt="rounded_outline"))

    # ─── 4. 各机器人统计 ─────────────────────────────────────────────────────
    section("各机器人跨行情统计")
    bot_rows = []
    for bk, label in [("memo","MemeBot  RSI超卖反弹"),("main","MainBot  EMA回调反弹"),
                      ("alpha","AlphaBot 突破回踩")]:
        rets = [r[f"{bk}_ret"] for r in all_results]
        shs  = [r[f"{bk}_sh"]  for r in all_results]
        wrs  = [r[f"{bk}_wr"]  for r in all_results]
        dds  = [r[f"{bk}_dd"]  for r in all_results]
        mcs  = [r[f"{bk}_mc"]  for r in all_results]
        pfs  = [r[f"{bk}_pf"]  for r in all_results]
        avg_ret = np.mean(rets)
        ret_c = Fore.GREEN if avg_ret > 0 else Fore.RED
        win_sc = sum(1 for x in rets if x > 0) / len(rets) * 100
        bot_rows.append([
            label,
            f"{ret_c}{avg_ret:+.1f}%±{np.std(rets):.1f}{Style.RESET_ALL}",
            f"{np.mean(shs):+.3f}",
            f"{np.mean(wrs):.0f}%",
            f"{np.mean(dds):.1f}%",
            f"{np.mean(mcs):.0f}",
            f"{np.mean(pfs):.2f}",
            f"{win_sc:.0f}%",
        ])
    print(tabulate(bot_rows,
                   headers=["Bot","均收益±σ","Sharpe","胜率","最大回撤","连亏","PF","盈利%"],
                   tablefmt="rounded_outline"))

    # ─── 5. 综合评分 ─────────────────────────────────────────────────────────
    section("综合评分与建议")
    all_rets = [r["total_return"] for r in all_results]
    all_sh   = [r["sharpe"] for r in all_results]
    all_dd   = [r["max_drawdown"] for r in all_results]
    all_mc   = [r["max_consec_loss"] for r in all_results]
    avg_ret  = np.mean(all_rets)
    avg_sh   = np.mean(all_sh)
    avg_dd   = np.mean(all_dd)
    avg_mc   = np.mean(all_mc)
    win_pct  = sum(1 for r in all_rets if r > 0) / len(all_rets) * 100
    best  = max(all_results, key=lambda x: x["total_return"])
    worst = min(all_results, key=lambda x: x["total_return"])

    score, notes, tips = 0, [], []
    if avg_ret > 10:   score += 30
    elif avg_ret > 5:  score += 22
    elif avg_ret > 0:  score += 12
    else: notes.append("均收益为负"); tips.append("检查均值回归参数，RSI阈值可放宽到40-45")

    if avg_sh > 1.5:   score += 25
    elif avg_sh > 1.0: score += 20
    elif avg_sh > 0.5: score += 13
    elif avg_sh > 0:   score += 6
    else: notes.append(f"Sharpe不足 ({avg_sh:.3f})"); tips.append("降低止损ATR倍数，提高止盈RR到3:1以上")

    if avg_dd < 8:     score += 25
    elif avg_dd < 15:  score += 20
    elif avg_dd < 25:  score += 12
    else: notes.append(f"回撤过大({avg_dd:.1f}%)"); tips.append("单笔风险降至1%，最大仓位2个")

    if win_pct >= 70:  score += 20
    elif win_pct >= 55:score += 16
    elif win_pct >= 40:score += 9
    else: notes.append(f"盈利场景比低({win_pct:.0f}%)"); tips.append("增加趋势过滤，只在EMA50上升时交易")

    grade = ("S" if score >= 92 else "A+" if score >= 85 else "A" if score >= 75
             else "B" if score >= 55 else "C" if score >= 35 else "D")
    gc = Fore.GREEN if grade in ("S","A+","A") else Fore.YELLOW if grade == "B" else Fore.RED

    ret_c = Fore.GREEN if avg_ret > 0 else Fore.RED
    sh_c  = Fore.GREEN if avg_sh > 1.0 else Fore.YELLOW if avg_sh > 0 else Fore.RED

    print(f"""
  {Fore.WHITE}{'─'*58}{Style.RESET_ALL}
  总模拟次数:    {len(all_results)} 次 ({len(REGIMES)} 场景 × {len(ALL_SEEDS)} 种子)
  平均收益率:    {ret_c}{avg_ret:+.2f}%{Style.RESET_ALL}  (std=±{np.std(all_rets):.1f}%)
  平均 Sharpe:  {sh_c}{avg_sh:+.3f}{Style.RESET_ALL}
  平均最大回撤:  {Fore.RED}{avg_dd:.2f}%{Style.RESET_ALL}
  平均最大连亏:  {avg_mc:.0f} 次
  盈利场景比:    {Fore.GREEN if win_pct>=50 else Fore.YELLOW}{win_pct:.0f}%{Style.RESET_ALL}
  最佳:          {Fore.GREEN}{best['regime']} seed={best['seed']} → {best['total_return']:+.1f}%{Style.RESET_ALL}
  最差:          {Fore.RED}{worst['regime']} seed={worst['seed']} → {worst['total_return']:+.1f}%{Style.RESET_ALL}
  运行耗时:      {elapsed:.1f}s
  {Fore.WHITE}{'─'*58}{Style.RESET_ALL}
  综合评分: {gc}【{grade}】  {score}/100{Style.RESET_ALL}""")

    if notes:
        print(f"\n  {Fore.RED}⚠ 问题:{Style.RESET_ALL}")
        for n in notes: print(f"    · {n}")
    if tips:
        print(f"\n  {Fore.YELLOW}→ 建议:{Style.RESET_ALL}")
        for t in tips: print(f"    · {t}")

    # 保存结果
    os.makedirs("logs", exist_ok=True)
    with open("logs/stress_v2_results.json", "w") as f:
        json.dump({"best_params": best_params, "summary": {
            "avg_return": avg_ret, "avg_sharpe": avg_sh, "avg_max_dd": avg_dd,
            "win_pct_scenarios": win_pct, "grade": grade, "score": score,
        }, "results": [{k:v for k,v in r.items()} for r in all_results]}, f, indent=2)
    print(f"\n  结果已保存: logs/stress_v2_results.json\n")


if __name__ == "__main__":
    main()
