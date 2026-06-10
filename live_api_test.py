#!/usr/bin/env python3
"""
公开 API 回测测试 — Live API Backtest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
从 Binance / OKX / Bybit 公开 API 抓取真实 OHLCV 数据，
用压力测试最优参数跑三个机器人，找出盈利最高的币种和交易所。

无需 API Key — 全部使用公开端点
用法: python live_api_test.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tabulate import tabulate

from simulation.vectorized_backtest import run_vectorized

init(autoreset=True)

# ── 压力测试最优参数 ────────────────────────────────────────────────────────────
BEST_PARAMS = {
    "memo":  {"rsi_oversold": 42, "rsi_confirm": 46, "sl_atr_mult": 1.5,
               "tp_rr": 2.5, "ema_slow": 50},
    "main":  {"rsi_low": 50, "rsi_high": 65, "sl_atr_mult": 1.5,
               "tp_rr": 3.5, "ema_fast": 20, "ema_slow": 50},
    "alpha": {"breakout_lookback": 15, "vol_mult": 1.8,
               "sl_atr_mult": 2.0, "tp_rr": 4.0},
}

CAPITAL  = 10_000.0
ALLOC    = {"memo": 0.20, "main": 0.50, "alpha": 0.30}
WARMUP   = {"memo": 55,   "main": 60,   "alpha": 65}
N_BARS   = 480   # 480 × 1h ≈ 20 天

# 各机器人对应的交易对（币安格式）
BOT_SYMBOLS = {
    "memo":  ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT"],
    "main":  ["BTCUSDT",  "ETHUSDT",  "BNBUSDT",  "SOLUSDT",   "XRPUSDT"],
    "alpha": ["SUIUSDT",  "APTUSDT",  "ARBUSDT",  "OPUSDT",    "INJUSDT"],
}

# 交易所之间共有的主流币（用于跨交易所比较）
CROSS_EXCHANGE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

BOT_LABEL = {
    "memo":  "MemeBot  (RSI超卖反弹)",
    "main":  "MainBot  (RSI动量回升)",
    "alpha": "AlphaBot (突破回踩)",
}

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── 数据获取函数 ───────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 10) -> dict | list:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_binance(symbol: str, n: int = N_BARS) -> pd.DataFrame | None:
    """Binance: 单次最多 1000 根 K 线，1h 间隔。"""
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={symbol}&interval=1h&limit={min(n, 1000)}")
    try:
        raw = _get(url)
        df = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","volume",
            "close_ts","quote_vol","trades","taker_base","taker_quote","_"
        ])
        df = df[["ts","open","high","low","close","volume"]].astype(float)
        df.index = pd.to_datetime(df["ts"], unit="ms")
        return df.drop(columns=["ts"]).tail(n)
    except Exception as e:
        return None


def fetch_okx(symbol: str, n: int = N_BARS) -> pd.DataFrame | None:
    """OKX: 每次最多 100 根，需要分页（倒序返回）。"""
    inst = symbol.replace("USDT", "-USDT")   # BTCUSDT → BTC-USDT
    candles = []
    after = ""
    pages = (n // 100) + 1
    try:
        for _ in range(pages):
            url = (f"https://www.okx.com/api/v5/market/candles"
                   f"?instId={inst}&bar=1H&limit=100"
                   + (f"&after={after}" if after else ""))
            data = _get(url).get("data", [])
            if not data:
                break
            candles.extend(data)
            after = data[-1][0]   # oldest timestamp → next page
            time.sleep(0.2)
        if not candles:
            return None
        candles = candles[::-1]   # 倒序 → 正序
        df = pd.DataFrame(candles, columns=[
            "ts","open","high","low","close","vol","vol_ccy","vol_quote","confirm"
        ])
        df = df[["ts","open","high","low","close","vol"]].rename(columns={"vol":"volume"})
        df = df.astype({"ts": float, "open": float, "high": float,
                         "low": float, "close": float, "volume": float})
        df.index = pd.to_datetime(df["ts"], unit="ms")
        return df.drop(columns=["ts"]).tail(n)
    except Exception:
        return None


def fetch_bybit(symbol: str, n: int = N_BARS) -> pd.DataFrame | None:
    """Bybit: 每次最多 200 根，需要分页（倒序返回）。"""
    candles = []
    end_ms = ""
    pages = (n // 200) + 1
    try:
        for _ in range(pages):
            url = (f"https://api.bybit.com/v5/market/kline"
                   f"?category=linear&symbol={symbol}&interval=60&limit=200"
                   + (f"&end={end_ms}" if end_ms else ""))
            result = _get(url).get("result", {})
            data = result.get("list", [])
            if not data:
                break
            candles.extend(data)
            end_ms = data[-1][0]   # oldest timestamp → next page
            time.sleep(0.2)
        if not candles:
            return None
        candles = candles[::-1]   # 倒序 → 正序
        df = pd.DataFrame(candles, columns=["ts","open","high","low","close","volume","turnover"])
        df = df[["ts","open","high","low","close","volume"]]
        df = df.astype(float)
        df.index = pd.to_datetime(df["ts"], unit="ms")
        return df.drop(columns=["ts"]).tail(n)
    except Exception:
        return None


FETCHERS = {
    "Binance": fetch_binance,
    "OKX":     fetch_okx,
    "Bybit":   fetch_bybit,
}


# ── 运行单个机器人 ─────────────────────────────────────────────────────────────

def run_bot_on_df(bot_key: str, symbol: str, df: pd.DataFrame) -> dict:
    cap = CAPITAL * ALLOC[bot_key]
    wm  = WARMUP[bot_key]
    trades, summary = run_vectorized(df, symbol, bot_key, cap,
                                     BEST_PARAMS[bot_key], warmup=wm)
    return {
        "symbol":       symbol,
        "bot":          bot_key,
        "return_pct":   round(summary["total_return_pct"], 3),
        "win_rate":     round(summary["win_rate"], 1),
        "sharpe":       round(summary["sharpe"], 3),
        "max_drawdown": round(summary["max_drawdown_pct"], 2),
        "total_trades": summary["total_trades"],
        "profit_factor":round(summary["profit_factor"], 2),
    }


# ── 输出工具 ──────────────────────────────────────────────────────────────────

def section(t: str):
    print(f"\n{Fore.CYAN}{'═'*72}\n  {t}\n{'═'*72}{Style.RESET_ALL}")


def color_ret(v: float) -> str:
    c = Fore.GREEN if v > 0 else Fore.RED
    return f"{c}{v:+.2f}%{Style.RESET_ALL}"


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║  ALPHA-BOT  公开 API 真实数据回测                                          ║
║  Binance / OKX / Bybit  |  三机器人对比  |  无需 API Key                  ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    # ── 阶段 1: 测试各交易所连通性 ───────────────────────────────────────────
    section("阶段 1 — 交易所连通性测试")
    available_exchanges: list[str] = []
    for name, fetcher in FETCHERS.items():
        print(f"  测试 {name:<10}", end="", flush=True)
        df = fetcher("BTCUSDT", n=10)
        if df is not None and len(df) >= 5:
            print(f"{Fore.GREEN}✅ 连接成功  ({len(df)} 根 K 线返回){Style.RESET_ALL}")
            available_exchanges.append(name)
        else:
            print(f"{Fore.RED}❌ 无法访问{Style.RESET_ALL}")
        time.sleep(0.5)

    if not available_exchanges:
        print(f"\n{Fore.RED}所有交易所均无法访问。")
        print("提示: 云服务器 IP 常被交易所封锁，请在本地机器上运行此脚本。{Style.RESET_ALL}")
        return

    primary = available_exchanges[0]
    print(f"\n  主数据源: {Fore.GREEN}{primary}{Style.RESET_ALL}")

    # ── 阶段 2: 全币种扫描（主数据源） ──────────────────────────────────────
    section(f"阶段 2 — 全币种盈利扫描 ({primary}，{N_BARS}根1h K线≈20天)")
    all_results: list[dict] = []
    fetcher = FETCHERS[primary]

    for bot_key, symbols in BOT_SYMBOLS.items():
        label = BOT_LABEL[bot_key]
        print(f"\n  {Fore.YELLOW}▶ {label}{Style.RESET_ALL}")
        for sym in symbols:
            print(f"    {sym:<12}", end="", flush=True)
            df = fetcher(sym, N_BARS)
            if df is None or len(df) < WARMUP[bot_key] + 20:
                print(f"{Fore.RED}跳过（数据不足）{Style.RESET_ALL}")
                continue
            r = run_bot_on_df(bot_key, sym, df)
            all_results.append({**r, "exchange": primary})
            ret_s = color_ret(r["return_pct"])
            print(f"收益 {ret_s}  胜率 {r['win_rate']:.0f}%  "
                  f"Sharpe {r['sharpe']:+.3f}  交易 {r['total_trades']}笔")
            time.sleep(0.15)

    # ── 阶段 3: 跨交易所比较（主流币） ──────────────────────────────────────
    if len(available_exchanges) > 1:
        section(f"阶段 3 — 跨交易所比较 (主流币: {', '.join(CROSS_EXCHANGE_SYMBOLS)})")
        cross_rows = []
        for sym in CROSS_EXCHANGE_SYMBOLS:
            for ex_name in available_exchanges:
                print(f"  {ex_name:<10} {sym:<12}", end="", flush=True)
                df = FETCHERS[ex_name](sym, N_BARS)
                if df is None or len(df) < WARMUP["main"] + 20:
                    print(f"{Fore.RED}跳过{Style.RESET_ALL}")
                    continue
                r = run_bot_on_df("main", sym, df)
                cross_rows.append({
                    "exchange": ex_name, **r
                })
                print(color_ret(r["return_pct"]))
                time.sleep(0.3)

        if cross_rows:
            section("交易所对比结果")
            rows = []
            for r in sorted(cross_rows, key=lambda x: x["return_pct"], reverse=True):
                rows.append([
                    r["exchange"], r["symbol"],
                    color_ret(r["return_pct"]),
                    f"{r['win_rate']:.0f}%",
                    f"{r['sharpe']:+.3f}",
                    f"{r['max_drawdown']:.1f}%",
                    f"{r['total_trades']}",
                ])
            print(tabulate(rows,
                           headers=["交易所","币种","收益%","胜率","Sharpe","最大回撤","交易数"],
                           tablefmt="rounded_outline"))

    # ── 阶段 4: 综合排行榜 ───────────────────────────────────────────────────
    if not all_results:
        print(f"{Fore.RED}无有效结果{Style.RESET_ALL}")
        return

    section("综合排行榜 — 盈利最高 TOP 10")
    sorted_results = sorted(all_results, key=lambda x: x["return_pct"], reverse=True)

    rows = []
    for i, r in enumerate(sorted_results[:10], 1):
        medal = ["🥇","🥈","🥉"] + ["  "] * 10
        rows.append([
            f"{medal[i-1]} #{i}",
            r["exchange"],
            r["symbol"],
            BOT_LABEL[r["bot"]].split("(")[0].strip(),
            color_ret(r["return_pct"]),
            f"{r['win_rate']:.0f}%",
            f"{r['sharpe']:+.3f}",
            f"{r['max_drawdown']:.1f}%",
            f"{r['total_trades']}",
            f"{r['profit_factor']:.2f}",
        ])
    print(tabulate(rows,
                   headers=["排名","交易所","币种","机器人","收益%","胜率","Sharpe","最大回撤","交易数","PF"],
                   tablefmt="rounded_outline"))

    # ── 阶段 5: 亏损最多 BOTTOM 5 ───────────────────────────────────────────
    section("⚠ 亏损最多 BOTTOM 5（需回避的组合）")
    worst = sorted(all_results, key=lambda x: x["return_pct"])[:5]
    wrows = []
    for r in worst:
        wrows.append([
            r["exchange"], r["symbol"],
            BOT_LABEL[r["bot"]].split("(")[0].strip(),
            color_ret(r["return_pct"]),
            f"{r['win_rate']:.0f}%",
            f"{r['total_trades']}",
        ])
    print(tabulate(wrows,
                   headers=["交易所","币种","机器人","收益%","胜率","交易数"],
                   tablefmt="rounded_outline"))

    # ── 阶段 6: 各机器人汇总 ─────────────────────────────────────────────────
    section("各机器人跨币种汇总")
    bot_rows = []
    for bk in ("memo", "main", "alpha"):
        rs = [r for r in all_results if r["bot"] == bk]
        if not rs:
            continue
        rets = [r["return_pct"] for r in rs]
        best_r = max(rs, key=lambda x: x["return_pct"])
        bot_rows.append([
            BOT_LABEL[bk],
            color_ret(float(np.mean(rets))),
            color_ret(max(rets)),
            f"{best_r['symbol']} ({best_r['exchange']})",
            f"{np.mean([r['win_rate'] for r in rs]):.0f}%",
            f"{np.mean([r['sharpe'] for r in rs]):+.3f}",
        ])
    print(tabulate(bot_rows,
                   headers=["机器人","均收益%","最高收益%","最佳币种","均胜率","均Sharpe"],
                   tablefmt="rounded_outline"))

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    out_path = "logs/live_api_results.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": pd.Timestamp.now().isoformat(),
                   "results": all_results}, f, ensure_ascii=False, indent=2)

    best = sorted_results[0]
    print(f"""
  {'─'*68}
  {Fore.GREEN}🏆 最高盈利组合:{Style.RESET_ALL}
      交易所: {best['exchange']}
      币种:   {best['symbol']}
      机器人: {BOT_LABEL[best['bot']]}
      收益:   {color_ret(best['return_pct'])}
      胜率:   {best['win_rate']:.0f}%  |  Sharpe: {best['sharpe']:+.3f}  |  PF: {best['profit_factor']:.2f}

  结果已保存: {out_path}
  {'─'*68}
""")


if __name__ == "__main__":
    main()
