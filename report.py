#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键生成运行报告 — 复制内容发给 Claude 进行分析优化
用法: python report.py
"""
import json, os, csv
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _read_log_tail(path, n=30):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return "(无日志)"

def _read_csv_tail(path, n=20):
    try:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return "(空)"
        header = rows[0]
        tail   = rows[max(1, len(rows)-n):]
        return "\n".join([",".join(header)] + [",".join(r) for r in tail])
    except Exception:
        return "(无数据)"

lines = []
sep = "=" * 60

lines.append(sep)
lines.append(f"  Alpha-Bot 运行报告  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append(sep)

# ── BTC 自动交易机器人 ─────────────────────────────────────
lines.append("\n[1] BTC 自动交易机器人 (auto_trader)")
at = _read_json(os.path.join(BASE, "logs", "auto_trader_state.json"))
if at:
    trades = at.get("trades", [])
    wins   = [t for t in trades if t.get("pnl_usd", 0) > 0]
    equity = at.get("equity", 10000)
    pnl    = equity - 10000
    wr     = len(wins)/len(trades)*100 if trades else 0
    lines.append(f"  初始资金: $10,000  当前权益: ${equity:,.2f}  盈亏: {pnl:+.2f}")
    lines.append(f"  交易次数: {len(trades)}  胜率: {wr:.0f}%  检查次数: {at.get('checks',0)}")
    if trades:
        lines.append("  最近5笔:")
        for t in trades[-5:][::-1]:
            lines.append(f"    {t.get('exit_time','')[:16]}  "
                         f"入${t.get('entry_price',0):,.0f}→出${t.get('exit_price',0):,.0f}  "
                         f"{t.get('reason','')}  {t.get('pnl_usd',0):+.2f}USD")
else:
    lines.append("  (尚未启动或无交易记录)")

lines.append("\n  最近日志:")
lines.append(_read_log_tail(os.path.join(BASE, "logs", "auto_trader.log"), 15))

# ── 鲸鱼 MemeBot ──────────────────────────────────────────
lines.append("\n[2] 鲸鱼 MemeBot (whale_memecoin_bot)")
wb = _read_json(os.path.join(BASE, "logs", "whale_bot_state.json"))
if wb:
    trades = wb.get("trades", [])
    wins   = [t for t in trades if t.get("pnl_pct", 0) > 0]
    equity = wb.get("equity", 10000)
    pos    = wb.get("positions", {})
    pnl    = equity - 10000
    wr     = len(wins)/len(trades)*100 if trades else 0
    lines.append(f"  扫描次数: {wb.get('scans',0)}  交易次数: {len(trades)}  胜率: {wr:.0f}%")
    lines.append(f"  当前权益: ${equity:,.2f}  盈亏: {pnl:+.2f}  持仓数: {len(pos)}")
    if pos:
        lines.append("  当前持仓:")
        for addr, p in pos.items():
            lines.append(f"    {p.get('name',addr[:8])}  {p.get('chain','')}  "
                         f"入${p.get('entry',0):.8f}  盈亏:{p.get('pnl_pct',0):+.1f}%")
    if trades:
        lines.append("  最近5笔:")
        for t in trades[-5:][::-1]:
            lines.append(f"    {t.get('time','')[:16]}  {t.get('name','')}  "
                         f"{t.get('reason','')}  {t.get('pnl_pct',0):+.1f}%")
else:
    lines.append("  (尚未启动或无交易记录)")

lines.append("\n  最近发现的高分代币 (最新20条):")
lines.append(_read_csv_tail(os.path.join(BASE, "logs", "whale_opportunities.csv"), 20))

lines.append("\n  最近日志:")
lines.append(_read_log_tail(os.path.join(BASE, "logs", "whale_bot.log"), 15))

lines.append("\n" + sep)
lines.append("  把以上内容完整发给 Claude 即可获得优化建议")
lines.append(sep)

report = "\n".join(lines)
print(report)

# 同时保存到文件
report_path = os.path.join(BASE, "logs", "latest_report.txt")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n报告已保存到: {report_path}")
