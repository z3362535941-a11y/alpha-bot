#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安合约交易榜 — 按胜率排名前100名交易员
数据来源: 币安公开排行榜 API（无需登录）
用法: python find_top_traders.py
"""
import urllib.request, json, time, os
from datetime import datetime

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "logs", "top_traders.txt")

def _post(url, payload):
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(url, data=data, headers=_HEADERS, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  请求失败: {e}")
        return None

def _get(url):
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  请求失败: {e}")
        return None

def fetch_leaderboard(period_type="WEEKLY", page_size=50, pages=6):
    """抓取币安合约排行榜"""
    url  = "https://www.binance.com/bapi/futures/v3/public/future/leaderboard/getLeaderboardRank"
    rows = []
    for page in range(1, pages + 1):
        print(f"  抓取排行榜第{page}页...", end="", flush=True)
        payload = {
            "isShared": True,
            "isTrader": False,
            "periodType": period_type,  # DAILY / WEEKLY / MONTHLY / ALL
            "statisticsType": "ROI",
            "pageNum": page,
            "pageSize": page_size,
        }
        res = _post(url, payload)
        if res and res.get("data"):
            batch = res["data"]
            rows.extend(batch)
            print(f" +{len(batch)}")
        else:
            print(" 无数据")
            break
        time.sleep(0.5)
    return rows

def fetch_trader_performance(encrypt_id):
    """获取单个交易员的详细业绩（含胜率）"""
    url = "https://www.binance.com/bapi/futures/v2/public/future/leaderboard/getOtherPerformance"
    payload = {"encryptedUid": encrypt_id, "tradeType": "PERPETUAL"}
    res = _post(url, payload)
    if res and res.get("data"):
        return res["data"]
    return None

def fetch_trader_base(encrypt_id):
    """获取单个交易员基本信息"""
    url = "https://www.binance.com/bapi/futures/v2/public/future/leaderboard/getOtherLeaderboardBaseInfo"
    payload = {"encryptedUid": encrypt_id, "tradeType": "PERPETUAL"}
    res = _post(url, payload)
    if res and res.get("data"):
        return res["data"]
    return None

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║  币安合约排行榜  —  胜率最高前100名交易员             ║
║  数据来源: 币安公开API (无需账号)                     ║
╚══════════════════════════════════════════════════════╝
""")

    # 抓取多个周期排行榜，合并去重
    all_traders = {}
    for period in ["WEEKLY", "MONTHLY"]:
        print(f"\n▶ 抓取 {period} 排行榜...")
        rows = fetch_leaderboard(period_type=period, page_size=50, pages=4)
        for r in rows:
            uid = r.get("encryptedUid", "")
            if uid and uid not in all_traders:
                all_traders[uid] = r

    print(f"\n共获取 {len(all_traders)} 个交易员，开始查询胜率...\n")

    results = []
    for i, (uid, base) in enumerate(list(all_traders.items())[:200], 1):
        nick = base.get("nickName", uid[:8])
        print(f"  [{i:3d}] {nick[:20]:20s} 查询中...", end="", flush=True)

        perf = fetch_trader_performance(uid)
        if not perf:
            print(" 无数据")
            time.sleep(0.3)
            continue

        # 解析业绩数据
        win_rate   = 0.0
        roi        = 0.0
        pnl        = 0.0
        trade_cnt  = 0
        avg_hold_h = 0.0

        for item in (perf if isinstance(perf, list) else []):
            key = item.get("key", "")
            val = item.get("value", "0") or "0"
            try:
                fval = float(val)
            except Exception:
                fval = 0.0
            if key == "winRate":        win_rate   = fval * 100
            elif key == "ROI":          roi        = fval * 100
            elif key == "PNL":          pnl        = fval
            elif key == "tradeCount":   trade_cnt  = int(fval)
            elif key == "avgHoldingTime": avg_hold_h = fval / 3600

        results.append({
            "uid":       uid,
            "nick":      nick,
            "win_rate":  round(win_rate, 1),
            "roi":       round(roi, 1),
            "pnl":       round(pnl, 2),
            "trades":    trade_cnt,
            "hold_h":    round(avg_hold_h, 1),
            "followers": base.get("followerCount", 0),
        })
        print(f" 胜率:{win_rate:.1f}%  ROI:{roi:.1f}%  交易:{trade_cnt}笔")
        time.sleep(0.4)

    # 按胜率排序
    results.sort(key=lambda x: (x["win_rate"], x["roi"]), reverse=True)
    top100 = [r for r in results if r["trades"] >= 10][:100]  # 至少10笔交易

    # 输出报告
    lines = []
    lines.append("=" * 90)
    lines.append(f"  币安合约 胜率TOP100 交易员  |  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 90)
    lines.append(f"{'排名':4s}  {'昵称':20s}  {'胜率':7s}  {'ROI':8s}  {'PnL(U)':12s}  {'交易数':6s}  {'均持仓(h)':8s}  {'粉丝':6s}")
    lines.append("-" * 90)

    for rank, r in enumerate(top100, 1):
        lines.append(
            f"#{rank:<3d}  {r['nick'][:20]:20s}  "
            f"{r['win_rate']:6.1f}%  "
            f"{r['roi']:+7.1f}%  "
            f"${r['pnl']:>11,.0f}  "
            f"{r['trades']:6d}笔  "
            f"{r['hold_h']:7.1f}h  "
            f"{r['followers']:6d}"
        )

    lines.append("=" * 90)
    lines.append(f"\n共分析 {len(results)} 名交易员，筛选出 {len(top100)} 名（≥10笔交易）")

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n结果已保存到: {OUT_FILE}")

if __name__ == "__main__":
    main()
