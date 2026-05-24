#!/usr/bin/env python3
"""
Alpha-Bot 主入口
用法:
  python main.py --sim          # 运行模拟试跑
  python main.py --live         # 实盘 (需配置 .env)
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Alpha-Bot Trading System")
    parser.add_argument("--sim", action="store_true", help="运行模拟试跑")
    parser.add_argument("--live", action="store_true", help="实盘交易 (需配置API)")
    args = parser.parse_args()

    if args.live:
        print("实盘模式: 请先配置 .env 文件，并确认风险承受能力")
        print("运行: cp .env.example .env && 编辑 .env 填入API密钥")
        sys.exit(0)

    # 默认运行模拟
    from run_simulation import main as run_sim
    run_sim()


if __name__ == "__main__":
    main()
