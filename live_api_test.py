#!/usr/bin/env python3
"""
公开 API 多周期回测 — Multi-Timeframe Live API Backtest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
自动对比 1h / 4h / 1d 三个周期，找出每个币种最赚钱的组合。

用法:
  python live_api_test.py                  # 自动跑全部周期
  python live_api_test.py --interval 4h    # 只跑 4h
  python live_api_test.py --interval 1d    # 只跑日线
  python live_api_test.py --interval 1h    # 只跑 1h
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tabulate import tabulate

from simulation.vectorized_backtest import run_vectorized

init(autoreset=True)

# ── 周期配置 ───────────────────────────────────────────────────────────────────
INTERVALS = {
    "1h": {
        "binance": "1h", "okx": "1H", "bybit": "60",
        "n_bars": 480, "desc": "1小时K线 ≈ 20天",
        "params": {   # 专为1h真实数据放宽RSI阈值
            "memo":  {"rsi_oversold": 38, "rsi_confirm": 44, "sl_atr_mult": 1.5, "tp_rr": 2.0, "ema_slow": 50},
            "main":  {"rsi_low": 42, "rsi_high": 58, "sl_atr_mult": 1.5, "tp_rr": 2.5, "ema_fast": 20, "ema_slow": 50},
            "alpha": {"breakout_lookback": 15, "vol_mult": 1.5, "sl_atr_mult": 1.5, "tp_rr": 3.0},
        },
    },
    "4h": {
        "binance": "4h", "okx": "4H", "bybit": "240",
        "n_bars": 500, "desc": "4小时K线 ≈ 83天",
        "params": {   # 4h数据趋势更清晰，使用压力测试优化参数
            "memo":  {"rsi_oversold": 40, "rsi_confirm": 46, "sl_atr_mult": 1.5, "tp_rr": 2.5, "ema_slow": 50},
            "main":  {"rsi_low": 45, "rsi_high": 62, "sl_atr_mult": 1.5, "tp_rr": 3.0, "ema_fast": 20, "ema_slow": 50},
            "alpha": {"breakout_lookback": 15, "vol_mult": 1.8, "sl_atr_mult": 2.0, "tp_rr": 4.0},
        },
    },
    "1d": {
        "binance": "1d", "okx": "1Dutc", "bybit": "D",
        "n_bars": 500, "desc": "日线K线 ≈ 500天",
        "params": {   # 日线趋势最稳定，使用压力测试最优参数
            "memo":  {"rsi_oversold": 42, "rsi_confirm": 46, "sl_atr_mult": 1.5, "tp_rr": 2.5, "ema_slow": 50},
            "main":  {"rsi_low": 50, "rsi_high": 65, "sl_atr_mult": 1.5, "tp_rr": 3.5, "ema_fast": 20, "ema_slow": 50},
            "alpha": {"breakout_lookback": 15, "vol_mult": 1.8, "sl_atr_mult": 2.0, "tp_rr": 4.0},
        },
    },
}

CAPITAL = 10_000.0
ALLOC   = {"memo": 0.20, "main": 0.50, "alpha": 0.30}
WARMUP  = {"memo": 55,   "main": 60,   "alpha": 65}

BOT_SYMBOLS = {
    "memo":  ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT"],
    "main":  ["BTCUSDT",  "ETHUSDT",  "BNBUSDT",  "SOLUSDT",   "XRPUSDT"],
    "alpha": ["SUIUSDT",  "APTUSDT",  "ARBUSDT",  "OPUSDT",    "INJUSDT"],
}

BOT_LABEL = {
    "memo":  "MemeBot  (RSI超卖反弹)",
    "main":  "MainBot  (RSI动量回升)",
    "alpha": "AlphaBot (突破回踩)",
}

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control":   "no-cache",
}


# ── 数据获取 ───────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 12) -> dict | list:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _to_df(rows: list, ts_col: int = 0) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df.drop(columns=["ts"])


def fetch_binance(symbol: str, interval_key: str, n: int) -> pd.DataFrame | None:
    interval = INTERVALS[interval_key]["binance"]
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={symbol}&interval={interval}&limit={min(n, 1000)}")
    try:
        raw = _get(url)
        return _to_df(raw).tail(n)
    except Exception:
        return None


def fetch_okx(symbol: str, interval_key: str, n: int) -> pd.DataFrame | None:
    interval = INTERVALS[interval_key]["okx"]
    inst = symbol.replace("USDT", "-USDT")
    candles, after = [], ""
    pages = (n // 100) + 1
    try:
        for _ in range(pages):
            url = (f"https://www.okx.com/api/v5/market/candles"
                   f"?instId={inst}&bar={interval}&limit=100"
                   + (f"&after={after}" if after else ""))
            data = _get(url).get("data", [])
            if not data:
                break
            candles.extend(data)
            after = data[-1][0]
            time.sleep(0.25)
        if not candles:
            return None
        rows = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in candles[::-1]]
        return _to_df(rows).tail(n)
    except Exception:
        return None


def fetch_bybit(symbol: str, interval_key: str, n: int) -> pd.DataFrame | None:
    interval = INTERVALS[interval_key]["bybit"]
    candles, end_ms = [], ""
    pages = (n // 200) + 1
    try:
        for _ in range(pages):
            url = (f"https://api.bybit.com/v5/market/kline"
                   f"?category=linear&symbol={symbol}&interval={interval}&limit=200"
                   + (f"&end={end_ms}" if end_ms else ""))
            result = _get(url).get("result", {})
            data = result.get("list", [])
            if not data:
                break
            candles.extend(data)
            end_ms = data[-1][0]
            time.sleep(0.25)
        if not candles:
            return None
        rows = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in candles[::-1]]
        return _to_df(rows).tail(n)
    except Exception:
        return None


FETCHERS = {
    "Binance": fetch_binance,
    "OKX":     fetch_okx,
    "Bybit":   fetch_bybit,
}


# ── 运行单个机器人 ─────────────────────────────────────────────────────────────

def run_bot_on_df(bot_key: str, symbol: str, df: pd.DataFrame,
                  params: dict) -> dict:
    cap = CAPITAL * ALLOC[bot_key]
    wm  = WARMUP[bot_key]
    try:
        trades, summary = run_vectorized(df, symbol, bot_key, cap, params, warmup=wm)
    except Exception:
        summary = {"total_return_pct": 0, "win_rate": 0, "sharpe": 0,
                   "max_drawdown_pct": 0, "total_trades": 0, "profit_factor": 0}
    return {
        "symbol":        symbol,
        "bot":           bot_key,
        "return_pct":    round(summary["total_return_pct"], 3),
        "win_rate":      round(summary["win_rate"], 1),
        "sharpe":        round(summary["sharpe"], 3),
        "max_drawdown":  round(summary["max_drawdown_pct"], 2),
        "total_trades":  summary["total_trades"],
        "profit_factor": round(summary.get("profit_factor", 0), 2),
    }


# ── 输出工具 ──────────────────────────────────────────────────────────────────

def section(t: str):
    print(f"\n{Fore.CYAN}{'═'*70}\n  {t}\n{'═'*70}{Style.RESET_ALL}")


def cr(v: float) -> str:
    c = Fore.GREEN if v > 0 else (Fore.YELLOW if v == 0 else Fore.RED)
    return f"{c}{v:+.2f}%{Style.RESET_ALL}"


# ── 单周期扫描 ────────────────────────────────────────────────────────────────

def run_interval(interval_key: str, exchange: str,
                 fetcher) -> list[dict]:
    cfg    = INTERVALS[interval_key]
    params = cfg["params"]
    n      = cfg["n_bars"]
    results = []

    for bot_key, symbols in BOT_SYMBOLS.items():
        label = BOT_LABEL[bot_key]
        print(f"\n  {Fore.YELLOW}▶ {label}{Style.RESET_ALL}")
        for sym in symbols:
            print(f"    {sym:<14}", end="", flush=True)
            df = fetcher(sym, interval_key, n)
            if df is None or len(df) < WARMUP[bot_key] + 30:
                print(f"{Fore.RED}跳过（数据不足 {0 if df is None else len(df)} 根）{Style.RESET_ALL}")
                continue
            r = run_bot_on_df(bot_key, sym, df, params[bot_key])
            r["interval"] = interval_key
            r["exchange"]  = exchange
            results.append(r)
            trades_s = f"{r['total_trades']}笔"
            print(f"收益 {cr(r['return_pct'])}  "
                  f"胜率 {r['win_rate']:>4.0f}%  "
                  f"Sharpe {r['sharpe']:+.3f}  "
                  f"{trades_s}")
            time.sleep(0.12)

    return results


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", choices=["1h","4h","1d"],
                        default=None, help="K线周期 (默认: 全部)")
    args = parser.parse_args()

    intervals_to_run = [args.interval] if args.interval else ["1h", "4h", "1d"]

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║  ALPHA-BOT  公开 API 多周期真实数据回测                                  ║
║  自动对比 {' / '.join(intervals_to_run):<10}  |  Binance / OKX / Bybit         ║
╚══════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    # ── 交易所连通性测试 ────────────────────────────────────────────────────
    section("交易所连通性测试")
    available: list[tuple[str, callable]] = []
    for name, fetcher in FETCHERS.items():
        print(f"  测试 {name:<10}", end="", flush=True)
        df = fetcher("BTCUSDT", "1h", 10)
        if df is not None and len(df) >= 5:
            print(f"{Fore.GREEN}✅ 连接成功{Style.RESET_ALL}")
            available.append((name, fetcher))
        else:
            print(f"{Fore.RED}❌ 无法访问{Style.RESET_ALL}")
        time.sleep(0.5)

    if not available:
        print(f"\n{Fore.RED}所有交易所均无法访问，请检查网络连接。{Style.RESET_ALL}")
        sys.exit(1)

    ex_name, ex_fetcher = available[0]
    print(f"\n  主数据源: {Fore.GREEN}{ex_name}{Style.RESET_ALL}")

    # ── 多周期扫描 ──────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for ivl in intervals_to_run:
        cfg = INTERVALS[ivl]
        section(f"周期: {ivl.upper()}  ({cfg['desc']})  |  {ex_name}")
        results = run_interval(ivl, ex_name, ex_fetcher)
        all_results.extend(results)

    if not all_results:
        print(f"{Fore.RED}无有效结果，请尝试 --interval 1d{Style.RESET_ALL}")
        sys.exit(1)

    # ── 过滤：只保留有实际交易的结果 ────────────────────────────────────────
    traded = [r for r in all_results if r["total_trades"] > 0]
    if not traded:
        traded = all_results   # fallback

    # ── 综合排行榜 ──────────────────────────────────────────────────────────
    section("🏆 综合盈利排行榜 TOP 10（所有周期 × 所有币种）")
    top10 = sorted(traded, key=lambda x: x["return_pct"], reverse=True)[:10]
    medals = ["🥇","🥈","🥉"] + ["  "] * 10
    rows = []
    for i, r in enumerate(top10, 1):
        rows.append([
            f"{medals[i-1]} #{i}",
            r["interval"].upper(),
            r["symbol"],
            BOT_LABEL[r["bot"]].split("(")[0].strip(),
            cr(r["return_pct"]),
            f"{r['win_rate']:.0f}%",
            f"{r['sharpe']:+.3f}",
            f"{r['max_drawdown']:.1f}%",
            f"{r['total_trades']}",
            f"{r['profit_factor']:.2f}",
        ])
    print(tabulate(rows,
                   headers=["排名","周期","币种","机器人","收益%","胜率","Sharpe","最大回撤","交易数","PF"],
                   tablefmt="rounded_outline"))

    # ── 各周期汇总 ──────────────────────────────────────────────────────────
    section("各周期对比汇总")
    ivl_rows = []
    for ivl in intervals_to_run:
        rs = [r for r in traded if r["interval"] == ivl]
        if not rs:
            ivl_rows.append([ivl.upper(), INTERVALS[ivl]["desc"], "无数据","—","—","—","—"])
            continue
        rets = [r["return_pct"] for r in rs]
        best  = max(rs, key=lambda x: x["return_pct"])
        win_count = sum(1 for r in rs if r["return_pct"] > 0)
        ivl_rows.append([
            f"{Fore.CYAN}{ivl.upper()}{Style.RESET_ALL}",
            INTERVALS[ivl]["desc"],
            cr(float(np.mean(rets))),
            cr(max(rets)),
            f"{best['symbol']} + {BOT_LABEL[best['bot']].split('(')[0].strip()}",
            f"{win_count}/{len(rs)} 盈利",
            f"{np.mean([r['sharpe'] for r in rs]):+.3f}",
        ])
    print(tabulate(ivl_rows,
                   headers=["周期","数据范围","均收益%","最高收益%","最佳组合","盈利比","均Sharpe"],
                   tablefmt="rounded_outline"))

    # ── 各机器人汇总 ─────────────────────────────────────────────────────────
    section("各机器人汇总（所有周期合并）")
    bot_rows = []
    for bk in ("memo", "main", "alpha"):
        rs = [r for r in traded if r["bot"] == bk]
        if not rs:
            continue
        rets = [r["return_pct"] for r in rs]
        best  = max(rs, key=lambda x: x["return_pct"])
        bot_rows.append([
            BOT_LABEL[bk],
            cr(float(np.mean(rets))),
            cr(max(rets)),
            f"{best['symbol']} {best['interval'].upper()}",
            f"{np.mean([r['win_rate'] for r in rs]):.0f}%",
            f"{np.mean([r['sharpe'] for r in rs]):+.3f}",
        ])
    print(tabulate(bot_rows,
                   headers=["机器人","均收益%","最高收益%","最佳币种+周期","均胜率","均Sharpe"],
                   tablefmt="rounded_outline"))

    # ── 亏损最多 ─────────────────────────────────────────────────────────────
    section("⚠  亏损最多 BOTTOM 5（需回避的组合）")
    worst = sorted(traded, key=lambda x: x["return_pct"])[:5]
    wrows = [[r["interval"].upper(), r["symbol"],
              BOT_LABEL[r["bot"]].split("(")[0].strip(),
              cr(r["return_pct"]),
              f"{r['win_rate']:.0f}%", f"{r['total_trades']}"]
             for r in worst]
    print(tabulate(wrows,
                   headers=["周期","币种","机器人","收益%","胜率","交易数"],
                   tablefmt="rounded_outline"))

    # ── 保存 & 结论 ─────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "logs", "live_api_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": pd.Timestamp.now().isoformat(),
                   "intervals": intervals_to_run,
                   "exchange": ex_name,
                   "results": traded}, f, ensure_ascii=False, indent=2)

    best = top10[0]
    print(f"""
  {'─'*66}
  {Fore.GREEN}🏆  最高盈利组合:{Style.RESET_ALL}
      周期:   {best['interval'].upper()}  ({INTERVALS[best['interval']]['desc']})
      交易所: {best['exchange']}
      币种:   {best['symbol']}
      机器人: {BOT_LABEL[best['bot']]}
      收益:   {cr(best['return_pct'])}
      胜率:   {best['win_rate']:.0f}%  |  Sharpe: {best['sharpe']:+.3f}
      交易数: {best['total_trades']} 笔  |  PF: {best['profit_factor']:.2f}

  结果已保存: {out_path}
  {'─'*66}
""")


if __name__ == "__main__":
    main()
