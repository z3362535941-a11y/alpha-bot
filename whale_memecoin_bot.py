#!/usr/bin/env python3
"""
MemeBot 鲸鱼追踪器 — 自动扫描小市值 Memecoin 爆发机会
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据源: DexScreener + GeckoTerminal (公开API，无需Key)
扫描链: Solana / BSC / Base / Ethereum
目标:   找出鲸鱼入场迹象 + 低市值 + 高爆发概率的 Memecoin

鲸鱼信号判断:
  • 成交量/市值 > 50%（大额买入拉升）
  • 平均单笔交易额 > $5,000（机构/鲸鱼级别）
  • 1小时涨幅 > 30% + 买单量 >> 卖单量
  • 极低市值（$100K-$5M）= 10-100倍空间

用法:
  python whale_memecoin_bot.py           # 持续扫描，每15分钟刷新
  python whale_memecoin_bot.py --once    # 扫描一次后退出
  python whale_memecoin_bot.py --top 20  # 只显示前N个机会
"""
import sys, os, json, time, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request
import urllib.error
import numpy as np
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)

# ── 配置 ──────────────────────────────────────────────────────────────────────
CHAINS = ["solana", "bsc", "base", "ethereum"]

# 筛选条件（可调整）
MIN_LIQUIDITY_USD  = 30_000    # 最低流动性（防止假代币）
MAX_MARKET_CAP     = 10_000_000  # 最高市值 $10M（超过则潜力有限）
MIN_MARKET_CAP     = 50_000    # 最低市值 $50K（太小容易跑路）
MIN_VOL_1H_USD     = 50_000    # 1小时最低成交量
MIN_PRICE_CHANGE_1H = 10.0     # 1小时涨幅 > 10%
MAX_TOKEN_AGE_DAYS  = 30       # 上线不超过30天

# 纸面交易配置
CAPITAL            = 10_000.0  # 总资金
MAX_POSITION_PCT   = 0.05      # 每笔最多5%（高风险小仓位）
TAKE_PROFIT_X      = [2.0, 5.0, 10.0]  # 分批止盈：2倍、5倍、10倍
STOP_LOSS_PCT      = -0.40     # 止损 -40%（Memecoin高波动）

STATE_FILE = os.path.join(os.path.dirname(__file__), "logs", "whale_bot_state.json")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    "Accept":     "application/json",
}

CHAIN_ICON = {
    "solana": "◎ SOL", "bsc": "⬡ BSC",
    "base": "🔵 BASE", "ethereum": "Ξ ETH",
}


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 12) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


# ── DexScreener 数据抓取 ──────────────────────────────────────────────────────

def fetch_dexscreener_trending() -> list[dict]:
    """抓取 DexScreener 上热门代币（所有链）"""
    tokens = []

    # 1. 获取置顶推广代币（通常是有热度的）
    boosts = _get("https://api.dexscreener.com/token-boosts/top/v1")
    if boosts:
        addresses = [b.get("tokenAddress","") for b in (boosts if isinstance(boosts, list) else []) if b.get("tokenAddress")]
        if addresses:
            chunk = ",".join(addresses[:30])
            data = _get(f"https://api.dexscreener.com/latest/dex/tokens/{chunk}")
            if data and "pairs" in data:
                tokens.extend(data["pairs"] or [])

    # 2. 逐链搜索 meme 相关词
    keywords = ["meme", "pepe", "doge", "cat", "inu", "moon", "elon", "baby", "bonk"]
    for chain in CHAINS[:3]:   # 主要扫 SOL / BSC / BASE
        for kw in keywords[:3]:
            result = _get(f"https://api.dexscreener.com/latest/dex/search?q={kw}+{chain}")
            if result and "pairs" in result:
                tokens.extend(result["pairs"] or [])
            time.sleep(0.15)

    return tokens


def fetch_geckoterminal_new(chain: str = "solana") -> list[dict]:
    """从 GeckoTerminal 获取指定链上的新代币池"""
    url = f"https://api.geckoterminal.com/api/v2/networks/{chain}/new_pools?page=1"
    data = _get(url)
    if not data:
        return []
    pools = data.get("data", [])
    results = []
    for p in pools:
        attr = p.get("attributes", {})
        rel  = p.get("relationships", {})
        results.append({
            "_source":         "geckoterminal",
            "_chain":          chain,
            "name":            attr.get("name", ""),
            "address":         attr.get("address", ""),
            "price_usd":       _safe_float(attr.get("base_token_price_usd")),
            "market_cap":      _safe_float(attr.get("market_cap_usd")),
            "liquidity_usd":   _safe_float(attr.get("reserve_in_usd")),
            "volume_1h":       _safe_float(attr.get("volume_usd", {}).get("h1")),
            "volume_24h":      _safe_float(attr.get("volume_usd", {}).get("h24")),
            "price_change_1h": _safe_float(attr.get("price_change_percentage", {}).get("h1")),
            "price_change_24h":_safe_float(attr.get("price_change_percentage", {}).get("h24")),
            "created_at":      attr.get("pool_created_at", ""),
            "txns_1h_buys":    0,
            "txns_1h_sells":   0,
        })
    return results


def parse_dexscreener_pair(p: dict) -> dict | None:
    """将 DexScreener 的 pair 数据解析成统一格式"""
    try:
        chain   = (p.get("chainId") or "").lower()
        base    = p.get("baseToken", {})
        name    = base.get("name", "") or base.get("symbol", "")
        symbol  = base.get("symbol", "")
        addr    = base.get("address", "")

        mc      = _safe_float(p.get("marketCap") or p.get("fdv"))
        liq     = _safe_float(p.get("liquidity", {}).get("usd"))
        vol1h   = _safe_float(p.get("volume", {}).get("h1"))
        vol24h  = _safe_float(p.get("volume", {}).get("h24"))
        pc1h    = _safe_float(p.get("priceChange", {}).get("h1"))
        pc24h   = _safe_float(p.get("priceChange", {}).get("h24"))
        price   = _safe_float(p.get("priceUsd"))

        txns    = p.get("txns", {}).get("h1", {})
        buys    = int(txns.get("buys", 0))
        sells   = int(txns.get("sells", 0))

        # 计算代币年龄（天）
        created = p.get("pairCreatedAt")
        age_days = 9999
        if created:
            age_days = (time.time() - created / 1000) / 86400

        return {
            "_source":         "dexscreener",
            "_chain":          chain,
            "name":            name,
            "symbol":          symbol,
            "address":         addr,
            "pair_url":        p.get("url", ""),
            "price_usd":       price,
            "market_cap":      mc,
            "liquidity_usd":   liq,
            "volume_1h":       vol1h,
            "volume_24h":      vol24h,
            "price_change_1h": pc1h,
            "price_change_24h":pc24h,
            "txns_1h_buys":    buys,
            "txns_1h_sells":   sells,
            "age_days":        round(age_days, 1),
        }
    except Exception:
        return None


def _safe_float(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


# ── 评分系统 ──────────────────────────────────────────────────────────────────

def score_token(t: dict) -> tuple[float, dict]:
    """
    综合评分 (0-100):
    - 鲸鱼信号（大额买入）   40分
    - 涨幅动量               20分
    - 市值甜区               20分
    - 安全性（流动性）       20分
    """
    score  = 0.0
    detail = {}

    mc     = t.get("market_cap", 0) or 0
    liq    = t.get("liquidity_usd", 0) or 0
    vol1h  = t.get("volume_1h", 0) or 0
    vol24h = t.get("volume_24h", 0) or 0
    pc1h   = t.get("price_change_1h", 0) or 0
    buys   = t.get("txns_1h_buys", 0) or 0
    sells  = t.get("txns_1h_sells", 0) or 0
    age    = t.get("age_days", 9999)

    # ── 鲸鱼信号 (40分) ──────────────────────────────────────────────────────
    # Vol/MCap 比 (1小时成交量/市值 越高 = 鲸鱼砸盘或拉盘越激烈)
    vol_mc_ratio = (vol1h / mc) if mc > 0 else 0
    if vol_mc_ratio > 1.0:   s = 25
    elif vol_mc_ratio > 0.5: s = 18
    elif vol_mc_ratio > 0.2: s = 10
    elif vol_mc_ratio > 0.1: s =  5
    else:                    s =  0
    score += s; detail["vol_mc_ratio"] = round(vol_mc_ratio, 3)

    # 平均单笔交易额（鲸鱼：>$5000/笔）
    total_txns = buys + sells
    avg_tx_size = vol1h / total_txns if total_txns > 0 else 0
    if avg_tx_size > 20000:   s = 15
    elif avg_tx_size > 5000:  s = 10
    elif avg_tx_size > 1000:  s =  5
    else:                     s =  0
    score += s; detail["avg_tx_usd"] = round(avg_tx_size, 0)

    # 买单压倒卖单（买>卖 = 鲸鱼净买入）
    if total_txns > 0:
        buy_ratio = buys / total_txns
        if buy_ratio > 0.7:   score += 10; detail["buy_dominance"] = "强 (>70%)"
        elif buy_ratio > 0.55: score += 5; detail["buy_dominance"] = "中 (>55%)"
        else:                  detail["buy_dominance"] = f"{buy_ratio*100:.0f}%"
    else:
        detail["buy_dominance"] = "无数据"

    # ── 涨幅动量 (20分) ──────────────────────────────────────────────────────
    if pc1h > 100:    s = 20
    elif pc1h > 50:   s = 15
    elif pc1h > 30:   s = 10
    elif pc1h > 10:   s =  5
    elif pc1h > 0:    s =  2
    else:             s =  0
    score += s; detail["price_change_1h"] = f"{pc1h:+.1f}%"

    # ── 市值甜区 (20分) ──────────────────────────────────────────────────────
    if   100_000 <= mc <= 500_000:   s = 20  # $100K-$500K: 最大空间
    elif 500_000 < mc <= 2_000_000:  s = 15  # $500K-$2M: 好区间
    elif 2_000_000 < mc <= 5_000_000:s = 10  # $2M-$5M: 尚可
    elif mc < 100_000 and mc > 50_000: s = 8  # 偏小，有风险
    elif 5_000_000 < mc <= 10_000_000: s = 5  # 偏大
    else:                             s =  0
    score += s; detail["market_cap"] = _fmt_usd(mc)

    # 新上线加分（<7天 = 早期机会）
    if age < 1:     score += 10; detail["age"] = f"{age*24:.1f}小时"
    elif age < 3:   score +=  7; detail["age"] = f"{age:.1f}天"
    elif age < 7:   score +=  4; detail["age"] = f"{age:.1f}天"
    elif age < 14:  score +=  2; detail["age"] = f"{age:.1f}天"
    else:                        detail["age"] = f"{age:.0f}天"

    # ── 安全性 (20分) ────────────────────────────────────────────────────────
    if liq > 500_000:    s = 20
    elif liq > 200_000:  s = 15
    elif liq > 100_000:  s = 10
    elif liq > 50_000:   s =  5
    elif liq > 30_000:   s =  2
    else:                s =  0
    score += s; detail["liquidity"] = _fmt_usd(liq)

    # ── 惩罚项 ───────────────────────────────────────────────────────────────
    if liq < 10_000:   score -= 30   # 极低流动性 = 高跑路风险
    if mc < 50_000:    score -= 20   # 市值过低
    if total_txns < 10: score -= 10  # 无人交易

    return round(max(0, score), 1), detail


def _fmt_usd(v: float) -> str:
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:.0f}"


def _potential(mc: float) -> str:
    """估算基于市值的潜在涨幅"""
    if mc <= 0: return "?"
    if mc < 100_000:   return f"🚀🚀🚀 1000x+"
    if mc < 300_000:   return f"🚀🚀  100-500x"
    if mc < 1_000_000: return f"🚀   20-100x"
    if mc < 3_000_000: return f"⚡   5-20x"
    if mc < 10_000_000:return f"📈  2-5x"
    return "➡️  <2x"


# ── 主扫描器 ──────────────────────────────────────────────────────────────────

def scan_all() -> list[dict]:
    """扫描所有链，返回评分后的机会列表"""
    raw_tokens: list[dict] = []

    print(f"  {Fore.YELLOW}▶ 扫描 DexScreener...{Style.RESET_ALL}", end="", flush=True)
    ds_pairs = fetch_dexscreener_trending()
    seen_addrs = set()
    for p in ds_pairs:
        t = parse_dexscreener_pair(p)
        if t and t["address"] not in seen_addrs:
            seen_addrs.add(t["address"])
            raw_tokens.append(t)
    print(f" {len(raw_tokens)} 个原始代币")

    for chain in ["solana", "bsc"]:
        print(f"  {Fore.YELLOW}▶ GeckoTerminal {chain}...{Style.RESET_ALL}",
              end="", flush=True)
        gt = fetch_geckoterminal_new(chain)
        for t in gt:
            if t.get("address") not in seen_addrs:
                seen_addrs.add(t.get("address",""))
                raw_tokens.append(t)
        print(f" +{len(gt)}")
        time.sleep(0.3)

    # 过滤
    filtered = []
    for t in raw_tokens:
        if (t.get("liquidity_usd", 0) < MIN_LIQUIDITY_USD): continue
        if (t.get("market_cap", 0) > MAX_MARKET_CAP): continue
        if (t.get("market_cap", 0) > 0 and t.get("market_cap", 0) < MIN_MARKET_CAP): continue
        if (t.get("volume_1h", 0) < MIN_VOL_1H_USD): continue
        if (t.get("price_change_1h", 0) < MIN_PRICE_CHANGE_1H): continue
        if (t.get("age_days", 9999) > MAX_TOKEN_AGE_DAYS): continue
        filtered.append(t)

    # 评分
    scored = []
    for t in filtered:
        sc, detail = score_token(t)
        scored.append({**t, "_score": sc, "_detail": detail})

    return sorted(scored, key=lambda x: x["_score"], reverse=True)


# ── 纸面交易 ──────────────────────────────────────────────────────────────────

def load_state() -> dict:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"equity": CAPITAL, "positions": {}, "trades": [], "scans": 0}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def update_positions(state: dict, opportunities: list[dict]) -> dict:
    """更新已持仓代币的当前价格，检查止盈/止损"""
    price_map = {t["address"]: t["price_usd"] for t in opportunities if t.get("address")}

    closed = []
    for addr, pos in state["positions"].items():
        cur_price = price_map.get(addr, pos.get("last_price", pos["entry"]))
        pos["last_price"] = cur_price
        pnl_pct = (cur_price - pos["entry"]) / pos["entry"] * 100 if pos["entry"] > 0 else 0

        # 止损检查
        if pnl_pct <= STOP_LOSS_PCT * 100:
            profit = (cur_price - pos["entry"]) * pos["qty"]
            state["equity"] += pos["entry"] * pos["qty"] + profit
            state["trades"].append({
                "name": pos["name"], "entry": pos["entry"],
                "exit": round(cur_price, 8), "pnl_pct": round(pnl_pct, 1),
                "reason": "止损", "time": datetime.now().isoformat(),
            })
            closed.append(addr)
            print(f"  {Fore.RED}💀 止损出场: {pos['name']}  {pnl_pct:+.1f}%{Style.RESET_ALL}")
            continue

        # 分批止盈
        for target_x in sorted(TAKE_PROFIT_X):
            target_pct = (target_x - 1) * 100
            if pnl_pct >= target_pct and not pos.get(f"tp_{target_x}x_done"):
                take_qty = pos["qty"] * 0.33   # 每次卖出1/3
                profit = (cur_price - pos["entry"]) * take_qty
                state["equity"] += pos["entry"] * take_qty + profit
                pos["qty"] -= take_qty
                pos[f"tp_{target_x}x_done"] = True
                state["trades"].append({
                    "name": pos["name"], "entry": pos["entry"],
                    "exit": round(cur_price, 8), "pnl_pct": round(pnl_pct, 1),
                    "reason": f"止盈{target_x}x", "time": datetime.now().isoformat(),
                })
                print(f"  {Fore.GREEN}💰 止盈 {target_x}x: {pos['name']}  {pnl_pct:+.1f}%{Style.RESET_ALL}")

        pos["pnl_pct"] = round(pnl_pct, 1)

    for addr in closed:
        del state["positions"][addr]

    return state


def try_enter(state: dict, top_tokens: list[dict]) -> dict:
    """尝试对高分代币建仓（最多同时持有5个，每笔≤5%）"""
    max_positions = 5
    if len(state["positions"]) >= max_positions:
        return state

    for t in top_tokens:
        if len(state["positions"]) >= max_positions:
            break
        addr = t.get("address", "")
        if not addr or addr in state["positions"]:
            continue
        if t["_score"] < 40:   # 评分不够不买
            break

        entry_price = t.get("price_usd", 0)
        if entry_price <= 0:
            continue

        invest = state["equity"] * MAX_POSITION_PCT
        qty    = invest / entry_price
        state["equity"] -= invest
        state["positions"][addr] = {
            "name":       t.get("symbol") or t.get("name", addr[:8]),
            "chain":      t.get("_chain", ""),
            "entry":      entry_price,
            "last_price": entry_price,
            "qty":        qty,
            "invest_usd": round(invest, 2),
            "score":      t["_score"],
            "url":        t.get("pair_url", ""),
            "entered_at": datetime.now().isoformat(),
            "pnl_pct":    0.0,
        }
        print(f"  {Fore.GREEN}🚀 买入: {state['positions'][addr]['name']}  "
              f"@${entry_price:.8f}  投入: ${invest:.2f}{Style.RESET_ALL}")

    return state


# ── 显示 ──────────────────────────────────────────────────────────────────────

def section(t: str):
    print(f"\n{Fore.CYAN}{'═'*72}\n  {t}\n{'═'*72}{Style.RESET_ALL}")


def show_opportunities(tokens: list[dict], top_n: int = 10):
    section(f"🐋  鲸鱼追踪排行榜  TOP {top_n}  (实时扫描)")
    rows = []
    for i, t in enumerate(tokens[:top_n], 1):
        d   = t.get("_detail", {})
        sc  = t["_score"]
        pc1h = t.get("price_change_1h", 0) or 0
        chain_label = CHAIN_ICON.get(t.get("_chain",""), t.get("_chain","")[:6])
        score_c = (Fore.GREEN if sc >= 60 else
                   Fore.YELLOW if sc >= 40 else Fore.RED)
        pc_c = Fore.GREEN if pc1h > 0 else Fore.RED
        rows.append([
            f"#{i}",
            chain_label,
            (t.get("symbol") or t.get("name","?"))[:12],
            f"{score_c}{sc:.0f}分{Style.RESET_ALL}",
            f"{pc_c}{pc1h:+.0f}%{Style.RESET_ALL}",
            d.get("market_cap", _fmt_usd(t.get("market_cap",0))),
            d.get("liquidity",  _fmt_usd(t.get("liquidity_usd",0))),
            f"${d.get('avg_tx_usd',0):,.0f}",
            d.get("buy_dominance", "-"),
            d.get("age", "-"),
            _potential(t.get("market_cap", 0)),
        ])
    print(tabulate(rows,
        headers=["排名","链","代币","评分","1h涨幅","市值","流动性","均笔额","买压","上线","潜力"],
        tablefmt="rounded_outline"))


def show_portfolio(state: dict):
    section("💼  纸面持仓")
    equity = state["equity"]
    pos    = state["positions"]
    trades = state["trades"]
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    total_invested = sum(p["invest_usd"] for p in pos.values())
    total_float_pnl = sum(
        (p["last_price"] - p["entry"]) / p["entry"] * p["invest_usd"]
        for p in pos.values() if p["entry"] > 0
    )

    init_equity = CAPITAL
    gain_pct = (equity + total_invested + total_float_pnl - init_equity) / init_equity * 100

    print(f"  初始资金: ${init_equity:,.2f}  |  "
          f"当前总值: ${equity + total_invested + total_float_pnl:,.2f}  "
          f"({Fore.GREEN if gain_pct > 0 else Fore.RED}{gain_pct:+.2f}%{Style.RESET_ALL})")
    print(f"  闲置现金: ${equity:,.2f}  |  "
          f"在途投资: ${total_invested:,.2f}  |  "
          f"历史交易: {len(trades)}笔  胜率: {len(wins)/len(trades)*100:.0f}%" if trades else
          f"  闲置现金: ${equity:,.2f}  |  历史交易: 0")

    if pos:
        rows = []
        for addr, p in pos.items():
            pnl = p["pnl_pct"]
            c = Fore.GREEN if pnl > 0 else Fore.RED
            rows.append([
                p["name"], p["chain"],
                f"${p['entry']:.8f}",
                f"${p['last_price']:.8f}",
                f"{c}{pnl:+.1f}%{Style.RESET_ALL}",
                f"${p['invest_usd']:.0f}",
                p.get("entered_at","")[:16],
            ])
        print(tabulate(rows,
            headers=["代币","链","入场价","当前价","盈亏%","投入","时间"],
            tablefmt="simple"))
    else:
        print(f"  {Fore.YELLOW}空仓{Style.RESET_ALL}")


def show_recent_trades(state: dict, n: int = 10):
    trades = state.get("trades", [])
    if not trades:
        return
    section("📜  最近交易记录")
    rows = []
    for t in trades[-n:][::-1]:
        c = Fore.GREEN if t["pnl_pct"] > 0 else Fore.RED
        icon = "✅" if t["pnl_pct"] > 0 else "💀"
        rows.append([
            icon, t["name"], t["reason"],
            f"{c}{t['pnl_pct']:+.1f}%{Style.RESET_ALL}",
            t.get("time","")[:16],
        ])
    print(tabulate(rows, headers=["","代币","原因","盈亏%","时间"], tablefmt="simple"))


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",  action="store_true", help="扫描一次后退出")
    parser.add_argument("--top",   type=int, default=10, help="显示前N个机会")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="启用纸面交易（默认开启）")
    args = parser.parse_args()

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║  ALPHA-BOT  MemeBot 鲸鱼追踪器                                        ║
║  扫描链: Solana / BSC / Base  |  目标: 10x-100x 小市值 Memecoin        ║
║  数据源: DexScreener + GeckoTerminal (公开API)                         ║
╚══════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
  筛选条件:
    市值区间: {_fmt_usd(MIN_MARKET_CAP)} ~ {_fmt_usd(MAX_MARKET_CAP)}
    1h成交量: > {_fmt_usd(MIN_VOL_1H_USD)}
    最低流动性: > {_fmt_usd(MIN_LIQUIDITY_USD)}
    1h涨幅: > {MIN_PRICE_CHANGE_1H}%
    上线天数: ≤ {MAX_TOKEN_AGE_DAYS}天
""")

    state = load_state()
    interval_mins = 15

    while True:
        try:
            now = datetime.now().strftime("%H:%M:%S")
            section(f"🔍  开始扫描  [{now}]  (第{state['scans']+1}次)")

            opportunities = scan_all()
            state["scans"] += 1

            if not opportunities:
                print(f"  {Fore.YELLOW}本次未找到符合条件的代币{Style.RESET_ALL}")
            else:
                show_opportunities(opportunities, args.top)

                # 纸面交易
                state = update_positions(state, opportunities)
                state = try_enter(state, opportunities)
                save_state(state)

            show_portfolio(state)
            show_recent_trades(state)

            if args.once:
                break

            print(f"\n  {Fore.CYAN}⏰  {interval_mins}分钟后刷新扫描..."
                  f"  (Ctrl+C 退出){Style.RESET_ALL}\n")
            time.sleep(interval_mins * 60)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}用户退出，状态已保存。{Style.RESET_ALL}")
            save_state(state)
            break
        except Exception as e:
            print(f"  {Fore.RED}⚠ 错误: {e}  5分钟后重试...{Style.RESET_ALL}")
            time.sleep(300)


if __name__ == "__main__":
    main()
