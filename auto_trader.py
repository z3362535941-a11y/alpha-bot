#!/usr/bin/env python3
"""
Alpha-Bot 自动交易 — BTC/USDT 4H 纸面交易 (Paper Trading)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于真实数据回测: 最优组合 → BTC + MainBot + 4H  (胜率100%, +3.47%)

用法:
  python auto_trader.py            # 持续运行，每4小时自动检查
  python auto_trader.py --once     # 立刻检查一次后退出
  python auto_trader.py --status   # 只看当前状态，不交易
"""
import sys, os, json, time, argparse
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request
import numpy as np
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

# ── 配置（可修改） ─────────────────────────────────────────────────────────────
SYMBOL        = "BTCUSDT"
INTERVAL      = "4h"
N_BARS        = 200          # 抓取最近 200 根4小时K线
CAPITAL       = 10_000.0    # 初始资金（美元）
RISK_PCT      = 0.02        # 每笔交易风险 2%

# MainBot 4H 最优参数
PARAMS = dict(
    rsi_low     = 45,    # RSI 回调阈值
    rsi_high    = 62,    # RSI 动量回升阈值
    sl_atr_mult = 1.5,   # 止损 = 1.5×ATR
    tp_rr       = 3.0,   # 止盈 = 3×止损距离
    ema_fast    = 20,
    ema_slow    = 50,
)

STATE_FILE = os.path.join(os.path.dirname(__file__), "logs", "auto_trader_state.json")
LOG_FILE   = os.path.join(os.path.dirname(__file__), "logs", "auto_trader.log")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
    "Accept":     "application/json",
}


# ── 工具 ──────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def beep():
    """Windows 提示音（信号触发时响铃）"""
    try:
        import winsound
        for freq, dur in [(800, 200), (1000, 200), (1200, 300)]:
            winsound.Beep(freq, dur)
            time.sleep(0.05)
    except Exception:
        pass   # 非 Windows 或无音频，静默跳过


def section(t: str):
    print(f"\n{Fore.CYAN}{'─'*60}\n  {t}\n{'─'*60}{Style.RESET_ALL}")


def cr(v: float, suffix: str = "%") -> str:
    c = Fore.GREEN if v > 0 else (Fore.YELLOW if v == 0 else Fore.RED)
    return f"{c}{v:+.2f}{suffix}{Style.RESET_ALL}"


# ── 数据获取 ──────────────────────────────────────────────────────────────────

_KLINE_URLS = [
    "https://api.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit={n}",
    "https://api1.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit={n}",
    "https://api2.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit={n}",
]

def fetch_ohlcv(symbol: str = SYMBOL, n: int = N_BARS) -> pd.DataFrame | None:
    for attempt, url_tpl in enumerate(_KLINE_URLS):
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
                wait = 2 ** retry
                log(f"⚠ 数据获取失败(节点{attempt+1} 第{retry+1}次): {e}  {wait}s后重试")
                time.sleep(wait)
    log("❌ 所有节点均失败，跳过本次检查")
    return None


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rsi"]     = _rsi(d["close"])
    d["ema_fast"] = _ema(d["close"], PARAMS["ema_fast"])
    d["ema_slow"] = _ema(d["close"], PARAMS["ema_slow"])
    d["atr"]     = _atr(d["high"], d["low"], d["close"])
    d["obv"]     = _obv(d["close"], d["volume"])
    return d


# ── 信号检测 ──────────────────────────────────────────────────────────────────

def check_signal(df: pd.DataFrame) -> dict:
    """
    检查最新收盘K线是否有买入信号。
    返回: {"signal": bool, "sl_price": float, "tp_price": float, ...指标}
    """
    d   = calc_indicators(df)
    # 使用倒数第2根（最新完整收盘K线）
    bar = d.iloc[-2]
    rsi_series     = d["rsi"].values
    ema_slow_series = d["ema_slow"].values
    obv_series     = d["obv"].values
    idx = len(d) - 2   # 最新完整K线的索引

    # 信号条件
    rsi_dipped   = bool(np.any(rsi_series[max(0,idx-8):idx] < PARAMS["rsi_low"]))
    rsi_high_ok  = bar["rsi"] > PARAMS["rsi_high"]
    rsi_rising   = idx >= 3 and bar["rsi"] > rsi_series[idx - 3]
    price_ok     = bar["close"] > bar["ema_fast"]
    trend_up     = (bar["ema_fast"] > bar["ema_slow"] and
                    bar["ema_slow"] > ema_slow_series[max(0, idx-8)])
    obv_rising   = idx >= 5 and obv_series[idx] > obv_series[idx - 5]

    signal = all([rsi_dipped, rsi_high_ok, rsi_rising,
                  price_ok, trend_up, obv_rising])

    # 止损/止盈计算
    atr_pct  = bar["atr"] / bar["close"] if bar["close"] > 0 else 0.01
    sl_pct   = min(max(atr_pct * PARAMS["sl_atr_mult"], 0.01), 0.08)
    tp_pct   = sl_pct * PARAMS["tp_rr"]
    entry    = bar["close"]
    sl_price = round(entry * (1 - sl_pct), 2)
    tp_price = round(entry * (1 + tp_pct), 2)

    live_price = round(float(d.iloc[-1]["close"]), 2)   # 最新未收盘K线的最新价

    return {
        "signal":     signal,
        "entry":      round(entry, 2),        # 信号入场价（上根完整K线收盘）
        "live_price": live_price,             # 实时市场价（最新K线当前价）
        "sl_price":   sl_price,
        "tp_price":   tp_price,
        "sl_pct":     round(sl_pct * 100, 2),
        "tp_pct":     round(tp_pct * 100, 2),
        "rsi":        round(float(bar["rsi"]), 1),
        "ema_fast":   round(float(bar["ema_fast"]), 2),
        "ema_slow":   round(float(bar["ema_slow"]), 2),
        "atr":        round(float(bar["atr"]), 2),
        "bar_time":   str(d.index[-2]),
        # 条件明细（用于调试）
        "cond": {
            "rsi_dipped":  rsi_dipped,
            "rsi_high":    rsi_high_ok,
            "rsi_rising":  rsi_rising,
            "price_ok":    price_ok,
            "trend_up":    trend_up,
            "obv_rising":  obv_rising,
        },
    }


# ── 持仓管理 ──────────────────────────────────────────────────────────────────

def load_state() -> dict:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "equity":     CAPITAL,
        "position":   None,   # {"entry":..,"sl":..,"tp":..,"qty":..,"entry_time":..}
        "trades":     [],
        "checks":     0,
        "started_at": datetime.now().isoformat(),
    }


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def calc_qty(equity: float, entry: float, sl_pct: float) -> float:
    """Kelly-inspired 固定2%风险仓位计算"""
    risk_usd = equity * RISK_PCT
    sl_frac  = sl_pct / 100
    if sl_frac <= 0:
        return 0.0
    qty = risk_usd / (entry * sl_frac)
    max_qty = equity * 0.5 / entry   # 最多动用50%资金
    return round(min(qty, max_qty), 6)


def check_position_exit(state: dict, df: pd.DataFrame) -> dict:
    """检查新K线中是否触发止损或止盈"""
    pos = state.get("position")
    if not pos:
        return state

    # 找到入场之后的所有新K线
    entry_time = pd.Timestamp(pos["entry_time"], tz="UTC")
    new_bars   = df[df.index > entry_time]

    for ts, bar in new_bars.iterrows():
        # 止损先检查（悲观优先）
        if bar["low"] <= pos["sl"]:
            exit_price = pos["sl"]
            pnl = (exit_price - pos["entry"]) * pos["qty"]
            reason = "止损"
        elif bar["high"] >= pos["tp"]:
            exit_price = pos["tp"]
            pnl = (exit_price - pos["entry"]) * pos["qty"]
            reason = "止盈"
        else:
            continue   # 未触发

        state["equity"] += pnl
        state["trades"].append({
            "entry_time":  pos["entry_time"],
            "exit_time":   str(ts),
            "entry_price": pos["entry"],
            "exit_price":  round(exit_price, 2),
            "qty":         pos["qty"],
            "pnl_usd":     round(pnl, 2),
            "pnl_pct":     round(pnl / (pos["entry"] * pos["qty"]) * 100, 2),
            "reason":      reason,
        })
        state["position"] = None
        color = Fore.GREEN if pnl > 0 else Fore.RED
        log(f"{color}【{reason}】出场 @{exit_price:.2f}  PnL: {pnl:+.2f} USD{Style.RESET_ALL}")
        if pnl > 0:
            beep()
        break   # 每次只处理第一个触发事件

    return state


# ── 仪表盘 ────────────────────────────────────────────────────────────────────

def show_dashboard(state: dict, sig: dict, next_check_mins: int):
    equity    = state["equity"]
    gain_pct  = (equity - CAPITAL) / CAPITAL * 100
    trades    = state["trades"]
    wins      = [t for t in trades if t["pnl_usd"] > 0]
    losses    = [t for t in trades if t["pnl_usd"] <= 0]
    win_rate  = len(wins)/len(trades)*100 if trades else 0
    pos       = state.get("position")

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║  ALPHA-BOT  BTCUSDT 4H  纸面交易 (Paper Trading)          ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
  更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")

    section("💰  账户状态")
    print(f"  初始资金: ${CAPITAL:,.2f}")
    print(f"  当前权益: ${equity:,.2f}  ({cr(gain_pct)})")
    print(f"  完成交易: {len(trades)} 笔  |  "
          f"盈利: {Fore.GREEN}{len(wins)}{Style.RESET_ALL}  "
          f"亏损: {Fore.RED}{len(losses)}{Style.RESET_ALL}  "
          f"胜率: {win_rate:.0f}%")
    if trades:
        total_pnl = sum(t["pnl_usd"] for t in trades)
        print(f"  累计盈亏: {cr(total_pnl, ' USD')}")

    section("📊  当前市场 (最新4H收盘)")
    rsi_c = Fore.GREEN if sig["rsi"] > 55 else Fore.RED if sig["rsi"] < 40 else Fore.YELLOW
    trend_icon = "✅ 上升" if sig["ema_fast"] > sig["ema_slow"] else "❌ 下行"
    live = sig.get("live_price", sig["entry"])
    print(f"  币种:    {SYMBOL}  |  周期: {INTERVAL.upper()}")
    print(f"  实时价格: ${live:,.2f}  (信号参考: ${sig['entry']:,.2f})")
    print(f"  RSI:     {rsi_c}{sig['rsi']}{Style.RESET_ALL}")
    print(f"  EMA20:   ${sig['ema_fast']:,.2f}  |  EMA50: ${sig['ema_slow']:,.2f}  |  趋势: {trend_icon}")
    print(f"  ATR:     ${sig['atr']:,.2f}")

    section("📋  信号条件检查")
    cond = sig["cond"]
    for key, label in [
        ("rsi_dipped", f"RSI 在近8根K线内跌破 {PARAMS['rsi_low']}"),
        ("rsi_high",   f"RSI 当前 > {PARAMS['rsi_high']}"),
        ("rsi_rising", "RSI 正在上升（vs 3根前）"),
        ("price_ok",   "价格在 EMA20 上方"),
        ("trend_up",   "EMA20 > EMA50 且 EMA50 上升"),
        ("obv_rising", "OBV 成交量确认上升"),
    ]:
        icon = f"{Fore.GREEN}✅" if cond[key] else f"{Fore.RED}❌"
        print(f"  {icon}{Style.RESET_ALL}  {label}")

    if sig["signal"]:
        print(f"\n  {Fore.GREEN}{'★'*50}")
        print(f"  ★  买入信号触发！  ★")
        print(f"  ★  入场: ${sig['entry']:,.2f}  "
              f"止损: ${sig['sl_price']:,.2f} (-{sig['sl_pct']}%)  "
              f"止盈: ${sig['tp_price']:,.2f} (+{sig['tp_pct']}%)")
        print(f"  {'★'*50}{Style.RESET_ALL}")

    section("📌  当前持仓")
    if pos:
        current_price = sig.get("live_price", sig["entry"])
        float_pnl = (current_price - pos["entry"]) * pos["qty"]
        float_pct  = (current_price - pos["entry"]) / pos["entry"] * 100
        print(f"  方向:  多单 (Long)")
        print(f"  入场:  ${pos['entry']:,.2f}  |  数量: {pos['qty']} BTC")
        print(f"  止损:  ${pos['sl']:,.2f}  |  止盈: ${pos['tp']:,.2f}")
        print(f"  浮动盈亏:  {cr(float_pnl, ' USD')}  ({cr(float_pct)})")
        print(f"  入场时间: {pos['entry_time']}")
    else:
        print(f"  {Fore.YELLOW}空仓（等待信号）{Style.RESET_ALL}")

    if trades:
        section("📜  最近5笔交易")
        for t in trades[-5:][::-1]:
            pnl_c = Fore.GREEN if t["pnl_usd"] > 0 else Fore.RED
            icon  = "✅" if t["pnl_usd"] > 0 else "❌"
            print(f"  {icon}  {t['exit_time'][:16]}  "
                  f"入 ${t['entry_price']:,.0f} → 出 ${t['exit_price']:,.0f}  "
                  f"{t['reason']}  "
                  f"{pnl_c}{t['pnl_usd']:+.2f} USD{Style.RESET_ALL}")

    if next_check_mins > 0:
        h, m = divmod(next_check_mins, 60)
        print(f"\n  {Fore.CYAN}⏰  下次检查: {h}小时{m}分钟后{Style.RESET_ALL}")
    print()


# ── 时间调度 ──────────────────────────────────────────────────────────────────

def minutes_to_next_candle(interval_hours: int = 4) -> int:
    """距离下一根4小时K线收盘还有多少分钟（收盘后1分钟再检查）"""
    now   = datetime.now(timezone.utc)
    mod   = now.hour % interval_hours
    h_to  = (interval_hours - mod) % interval_hours or interval_hours
    nxt   = now.replace(minute=1, second=0, microsecond=0) + timedelta(hours=h_to)
    delta = max(0, int((nxt - now).total_seconds() / 60))
    return delta


# ══════════════════════════════════════════════════════════════════════════════
def run_check(state: dict, status_only: bool = False) -> dict:
    """执行一次完整检查：获取数据 → 检测退出 → 检测信号 → 更新状态"""
    log("🔍 开始检查...")
    df = fetch_ohlcv()
    if df is None or len(df) < 80:
        log("❌ 数据不足，跳过本次检查")
        return state

    # 检查是否有持仓需要退出
    state = check_position_exit(state, df)

    # 计算信号
    sig = check_signal(df)
    state["checks"] += 1

    # 如果有信号且无持仓 → 入场
    if sig["signal"] and not state.get("position") and not status_only:
        qty = calc_qty(state["equity"], sig["entry"], sig["sl_pct"])
        if qty > 0:
            state["position"] = {
                "entry":      sig["entry"],
                "sl":         sig["sl_price"],
                "tp":         sig["tp_price"],
                "qty":        qty,
                "sl_pct":     sig["sl_pct"],
                "entry_time": sig["bar_time"],
            }
            cost = sig["entry"] * qty
            log(f"🚀 买入信号！入场 @{sig['entry']:.2f}  "
                f"数量 {qty} BTC  成本 ${cost:.2f}  "
                f"止损 ${sig['sl_price']:.2f}  止盈 ${sig['tp_price']:.2f}")
            beep()

    save_state(state)
    return state, sig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",   action="store_true", help="检查一次后退出")
    parser.add_argument("--status", action="store_true", help="只查看状态，不执行交易")
    args = parser.parse_args()

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║  ALPHA-BOT  自动交易启动                                    ║
║  最优组合: BTCUSDT + MainBot + 4H                           ║
║  模式: {'只看状态' if args.status else '纸面交易（模拟）'}                              ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    state = load_state()
    log(f"已载入状态，历史交易 {len(state['trades'])} 笔，当前权益 ${state['equity']:.2f}")

    if args.status:
        df = fetch_ohlcv()
        if df is not None:
            sig = check_signal(df)
            show_dashboard(state, sig, 0)
        return

    if args.once:
        result = run_check(state, status_only=False)
        state, sig = result
        show_dashboard(state, sig, 0)
        return

    # ── 持续运行模式 ──────────────────────────────────────────────────────────
    log("进入持续运行模式（Ctrl+C 退出）")
    consecutive_errors = 0
    while True:
        try:
            result = run_check(state, status_only=False)
            state, sig = result
            consecutive_errors = 0   # 成功后重置错误计数

            next_mins = minutes_to_next_candle()
            show_dashboard(state, sig, next_mins)

            wait_secs = max(next_mins * 60, 60)
            log(f"💤 休眠 {next_mins} 分钟，等待下根4H K线收盘...")
            time.sleep(wait_secs)

        except KeyboardInterrupt:
            log("用户中断，保存状态退出。")
            save_state(state)
            break
        except Exception as e:
            consecutive_errors += 1
            wait = min(300 * consecutive_errors, 1800)  # 最长等30分钟
            log(f"⚠ 运行错误({consecutive_errors}次): {e}  {wait//60}分钟后重试...")
            try:
                save_state(state)
            except Exception:
                pass
            time.sleep(wait)


if __name__ == "__main__":
    main()
