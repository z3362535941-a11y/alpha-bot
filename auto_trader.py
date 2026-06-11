#!/usr/bin/env python3
"""
Alpha-Bot 多币种扫描交易 — 1H 纸面交易 (Paper Trading)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
同时扫描 BTC/ETH/BNB/SOL/XRP/DOGE，找最佳信号入场
1小时K线，最多同时持2个仓位，每笔3%风险

用法:
  python auto_trader.py          # 持续运行，每小时检查
  python auto_trader.py --once   # 检查一次后退出
  python auto_trader.py --status # 只看状态
"""
import sys, os, json, time, argparse
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request
import numpy as np
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

# ── 配置 ──────────────────────────────────────────────────────────────────────
COINS    = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
INTERVAL = "1h"
N_BARS   = 300        # 300根1H K线 ≈ 12天
CAPITAL  = 10_000.0
RISK_PCT = 0.03       # 每笔风险3%
MAX_POSITIONS = 2     # 最多同时持2个仓位

PARAMS = dict(
    rsi_low     = 42,   # RSI回调阈值（放宽以获取更多信号）
    rsi_high    = 56,   # RSI回升确认
    sl_atr_mult = 1.5,
    tp_rr       = 2.5,  # 止盈=2.5×止损
    ema_fast    = 20,
    ema_slow    = 50,
)

STATE_FILE = os.path.join(os.path.dirname(__file__), "logs", "auto_trader_state.json")
LOG_FILE   = os.path.join(os.path.dirname(__file__), "logs", "auto_trader.log")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
    "Accept":     "application/json",
}

_KLINE_URLS = [
    "https://api.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit={n}",
    "https://api1.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit={n}",
    "https://api2.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit={n}",
]

COIN_ICON = {
    "BTCUSDT": "₿ BTC", "ETHUSDT": "Ξ ETH", "BNBUSDT": "⬡ BNB",
    "SOLUSDT": "◎ SOL", "XRPUSDT": "✕ XRP", "DOGEUSDT": "Ð DOGE",
}


# ── 工具 ──────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def beep():
    try:
        import winsound
        for freq, dur in [(800,200),(1000,200),(1200,300)]:
            winsound.Beep(freq, dur); time.sleep(0.05)
    except Exception:
        pass

def cr(v: float, suffix: str = "%") -> str:
    c = Fore.GREEN if v > 0 else (Fore.YELLOW if v == 0 else Fore.RED)
    return f"{c}{v:+.2f}{suffix}{Style.RESET_ALL}"

def section(t: str):
    print(f"\n{Fore.CYAN}{'─'*62}\n  {t}\n{'─'*62}{Style.RESET_ALL}")


# ── 数据获取 ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, n: int = N_BARS) -> pd.DataFrame | None:
    for url_tpl in _KLINE_URLS:
        url = url_tpl.format(sym=symbol, iv=INTERVAL, n=n)
        for retry in range(3):
            try:
                req = urllib.request.Request(url, headers=_HEADERS)
                with urllib.request.urlopen(req, timeout=15) as r:
                    raw = json.loads(r.read().decode())
                df = pd.DataFrame(raw, columns=[
                    "ts","open","high","low","close","volume",
                    "close_ts","qvol","trades","tbvol","tqvol","_"
                ])
                df = df[["ts","open","high","low","close","volume"]].astype(float)
                df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df.drop(columns=["ts"])
            except Exception as e:
                time.sleep(2 ** retry)
    return None


# ── 指标 ──────────────────────────────────────────────────────────────────────

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(high, low, close, n=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def calc_indicators(df):
    d = df.copy()
    d["rsi"]      = _rsi(d["close"])
    d["ema_fast"] = _ema(d["close"], PARAMS["ema_fast"])
    d["ema_slow"] = _ema(d["close"], PARAMS["ema_slow"])
    d["atr"]      = _atr(d["high"], d["low"], d["close"])
    d["obv"]      = _obv(d["close"], d["volume"])
    return d


# ── 信号检测 ──────────────────────────────────────────────────────────────────

def check_signal(df: pd.DataFrame, symbol: str) -> dict:
    d   = calc_indicators(df)
    bar = d.iloc[-2]   # 最新完整K线
    rsi = d["rsi"].values
    ema_slow = d["ema_slow"].values
    obv = d["obv"].values
    idx = len(d) - 2

    rsi_dipped  = bool(np.any(rsi[max(0,idx-6):idx] < PARAMS["rsi_low"]))
    rsi_high_ok = bar["rsi"] > PARAMS["rsi_high"]
    rsi_rising  = idx >= 3 and bar["rsi"] > rsi[idx-3]
    price_ok    = bar["close"] > bar["ema_fast"]
    trend_up    = (bar["ema_fast"] > bar["ema_slow"] and
                   bar["ema_slow"] > ema_slow[max(0,idx-8)])
    obv_rising  = idx >= 5 and obv[idx] > obv[idx-5]

    conds   = [rsi_dipped, rsi_high_ok, rsi_rising, price_ok, trend_up, obv_rising]
    n_met   = sum(conds)
    signal  = all(conds)

    atr_pct  = bar["atr"] / bar["close"] if bar["close"] > 0 else 0.01
    sl_pct   = min(max(atr_pct * PARAMS["sl_atr_mult"], 0.005), 0.06)
    tp_pct   = sl_pct * PARAMS["tp_rr"]
    entry    = bar["close"]
    live     = round(float(d.iloc[-1]["close"]), 6)

    return {
        "symbol":    symbol,
        "signal":    signal,
        "n_met":     n_met,       # 满足条件数（用于排序）
        "entry":     round(entry, 6),
        "live":      live,
        "sl_price":  round(entry * (1-sl_pct), 6),
        "tp_price":  round(entry * (1+tp_pct), 6),
        "sl_pct":    round(sl_pct*100, 2),
        "tp_pct":    round(tp_pct*100, 2),
        "rsi":       round(float(bar["rsi"]), 1),
        "ema_fast":  round(float(bar["ema_fast"]), 4),
        "ema_slow":  round(float(bar["ema_slow"]), 4),
        "bar_time":  str(d.index[-2]),
        "cond": {
            "rsi_dipped": rsi_dipped, "rsi_high": rsi_high_ok,
            "rsi_rising": rsi_rising, "price_ok": price_ok,
            "trend_up":   trend_up,   "obv_rising": obv_rising,
        },
    }


# ── 持仓管理 ──────────────────────────────────────────────────────────────────

def load_state() -> dict:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"equity": CAPITAL, "positions": {}, "trades": [], "checks": 0,
            "started_at": datetime.now().isoformat()}

def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def calc_qty(equity, entry, sl_pct):
    risk_usd = equity * RISK_PCT
    sl_frac  = sl_pct / 100
    if sl_frac <= 0: return 0.0
    qty = risk_usd / (entry * sl_frac)
    return round(min(qty, equity * 0.4 / entry), 8)

def check_exits(state: dict, signals: dict) -> dict:
    """检查所有持仓是否触发止盈止损"""
    closed = []
    for symbol, pos in state["positions"].items():
        sig = signals.get(symbol)
        cur = sig["live"] if sig else pos.get("last_price", pos["entry"])
        pos["last_price"] = cur
        pnl_pct = (cur - pos["entry"]) / pos["entry"] * 100

        reason = None
        if cur <= pos["sl"]: reason = "止损"
        elif cur >= pos["tp"]: reason = "止盈"

        if reason:
            pnl_usd = (cur - pos["entry"]) * pos["qty"]
            state["equity"] += pos["entry"] * pos["qty"] + pnl_usd
            state["trades"].append({
                "symbol": symbol, "entry": pos["entry"],
                "exit": round(cur, 6), "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2), "reason": reason,
                "time": datetime.now().isoformat(),
            })
            closed.append(symbol)
            icon = Fore.GREEN if pnl_usd > 0 else Fore.RED
            log(f"{icon}{reason}: {symbol}  {pnl_pct:+.2f}%  {pnl_usd:+.2f}USD{Style.RESET_ALL}")
        else:
            pos["pnl_pct"] = round(pnl_pct, 2)

    for s in closed:
        del state["positions"][s]
    return state


# ── 显示 ──────────────────────────────────────────────────────────────────────

def show_dashboard(state: dict, signals: list[dict], next_mins: int):
    trades  = state.get("trades", [])
    equity  = state.get("equity", CAPITAL)
    pos_map = state.get("positions", {})
    wins    = [t for t in trades if t.get("pnl_usd", 0) > 0]
    gain    = (equity - CAPITAL) / CAPITAL * 100

    print(f"\n{Fore.CYAN}{'═'*62}")
    print(f"  ALPHA-BOT  多币种1H扫描  纸面交易")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*62}{Style.RESET_ALL}")

    section("账户状态")
    print(f"  初始资金: ${CAPITAL:,.2f}  当前权益: ${equity:,.2f}  ({cr(gain)})")
    wr = len(wins)/len(trades)*100 if trades else 0
    print(f"  完成交易: {len(trades)}笔  胜率: {wr:.0f}%  持仓: {len(pos_map)}/{MAX_POSITIONS}")

    section("币种信号扫描")
    for sig in signals:
        sym   = sig["symbol"]
        icon  = COIN_ICON.get(sym, sym)
        rsi_c = Fore.GREEN if sig["rsi"] > 55 else Fore.RED if sig["rsi"] < 40 else Fore.YELLOW
        trend = "上升" if sig["ema_fast"] > sig["ema_slow"] else "下行"
        bar   = f"{'█'*sig['n_met']}{'░'*(6-sig['n_met'])}"
        sig_c = Fore.GREEN if sig["signal"] else (Fore.YELLOW if sig["n_met"]>=4 else Fore.WHITE)
        in_pos = " [持仓中]" if sym in pos_map else ""
        print(f"  {sig_c}{icon:10s}{Style.RESET_ALL}  "
              f"${sig['live']:,.4f}  "
              f"RSI:{rsi_c}{sig['rsi']:5.1f}{Style.RESET_ALL}  "
              f"趋势:{trend}  "
              f"信号:{sig_c}{bar}{Style.RESET_ALL}{Fore.GREEN}{in_pos}{Style.RESET_ALL}")

    if any(s["signal"] for s in signals):
        section("买入信号触发！")
        for sig in signals:
            if sig["signal"]:
                print(f"  {Fore.GREEN}★ {COIN_ICON.get(sig['symbol'],sig['symbol'])}  "
                      f"入场:${sig['entry']:,.4f}  "
                      f"止损:${sig['sl_price']:,.4f}(-{sig['sl_pct']}%)  "
                      f"止盈:${sig['tp_price']:,.4f}(+{sig['tp_pct']}%){Style.RESET_ALL}")

    section("当前持仓")
    if pos_map:
        for sym, pos in pos_map.items():
            pnl = pos.get("pnl_pct", 0)
            c   = Fore.GREEN if pnl > 0 else Fore.RED
            print(f"  {COIN_ICON.get(sym,sym)}  入场:${pos['entry']:,.4f}  "
                  f"止损:${pos['sl']:,.4f}  止盈:${pos['tp']:,.4f}  "
                  f"盈亏:{c}{pnl:+.2f}%{Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}空仓（等待信号）{Style.RESET_ALL}")

    if trades:
        section("最近5笔交易")
        for t in trades[-5:][::-1]:
            c = Fore.GREEN if t["pnl_usd"] > 0 else Fore.RED
            print(f"  {'✅' if t['pnl_usd']>0 else '❌'}  "
                  f"{t.get('time','')[:16]}  {t.get('symbol','')}  "
                  f"{t['reason']}  {c}{t['pnl_usd']:+.2f}USD{Style.RESET_ALL}")

    if next_mins > 0:
        h, m = divmod(next_mins, 60)
        print(f"\n  {Fore.CYAN}下次检查: {h}小时{m}分钟后{Style.RESET_ALL}\n")


# ── 主循环 ──────────────────────────────────────────────────────────────────

def run_check(state: dict, status_only: bool = False) -> tuple[dict, list]:
    log(f"扫描 {len(COINS)} 个币种...")
    signals = []
    for symbol in COINS:
        df = fetch_ohlcv(symbol)
        if df is None or len(df) < 60:
            continue
        try:
            sig = check_signal(df, symbol)
            signals.append(sig)
        except Exception as e:
            log(f"  {symbol} 信号计算失败: {e}")

    state = check_exits(state, {s["symbol"]: s for s in signals})
    state["checks"] += 1

    if not status_only:
        # 按信号强度排序，依次尝试入场
        best = sorted([s for s in signals if s["signal"]], key=lambda x: x["n_met"], reverse=True)
        for sig in best:
            if len(state["positions"]) >= MAX_POSITIONS:
                break
            if sig["symbol"] in state["positions"]:
                continue
            qty = calc_qty(state["equity"], sig["entry"], sig["sl_pct"])
            if qty <= 0:
                continue
            cost = sig["entry"] * qty
            state["equity"] -= cost
            state["positions"][sig["symbol"]] = {
                "entry":      sig["entry"],
                "sl":         sig["sl_price"],
                "tp":         sig["tp_price"],
                "qty":        qty,
                "sl_pct":     sig["sl_pct"],
                "last_price": sig["live"],
                "pnl_pct":    0.0,
                "entry_time": sig["bar_time"],
            }
            log(f"买入 {sig['symbol']} @{sig['entry']}  止损:{sig['sl_price']}  止盈:{sig['tp_price']}")
            beep()

    save_state(state)
    return state, signals


def minutes_to_next_hour() -> int:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(hours=1)).replace(minute=2, second=0, microsecond=0)
    return max(1, int((nxt - now).total_seconds() / 60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",   action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    print(f"{Fore.CYAN}╔═══════════════════════════════════════════════════╗")
    print(f"║  ALPHA-BOT  多币种1H扫描  纸面交易                  ║")
    print(f"║  币种: BTC ETH BNB SOL XRP DOGE                    ║")
    print(f"╚═══════════════════════════════════════════════════╝{Style.RESET_ALL}")

    state = load_state()
    log(f"历史交易{len(state['trades'])}笔  权益${state['equity']:.2f}")

    consec_errors = 0
    while True:
        try:
            state, signals = run_check(state, status_only=args.status)
            consec_errors  = 0
            next_mins      = 0 if (args.once or args.status) else minutes_to_next_hour()
            show_dashboard(state, signals, next_mins)
            if args.once or args.status:
                break
            log(f"休眠{next_mins}分钟，等待下根1H K线...")
            time.sleep(next_mins * 60)
        except KeyboardInterrupt:
            log("用户退出")
            save_state(state)
            break
        except Exception as e:
            consec_errors += 1
            wait = min(300 * consec_errors, 1800)
            log(f"错误({consec_errors}): {e}  {wait//60}分钟后重试")
            time.sleep(wait)


if __name__ == "__main__":
    main()
