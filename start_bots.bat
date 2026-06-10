@echo off
title Alpha-Bot 启动器
color 0A
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║         Alpha-Bot 双机器人启动器                  ║
echo  ║  Bot 1: MainBot  (BTC 4H 自动交易)               ║
echo  ║  Bot 2: WhaleMemeBot (鲸鱼追踪 小市值Meme)        ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM 获取当前目录
set DIR=%~dp0

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo  正在启动 Bot 1: BTC 自动交易 (auto_trader.py)...
start "Alpha-Bot [BTC 4H 自动交易]" cmd /k "cd /d "%DIR%" && color 0B && python auto_trader.py"

REM 等待1秒，避免两个窗口同时抢资源
timeout /t 1 /nobreak >nul

echo  正在启动 Bot 2: 鲸鱼MemeBot (whale_memecoin_bot.py)...
start "Alpha-Bot [鲸鱼 MemeBot]" cmd /k "cd /d "%DIR%" && color 0E && python whale_memecoin_bot.py"

echo.
echo  ✅ 两个机器人已在独立窗口中启动！
echo.
echo  窗口说明:
echo    蓝色窗口  = BTC 4H 自动交易  (每4小时检查一次)
echo    黄色窗口  = 鲸鱼 MemeBot     (每15分钟扫描一次)
echo.
echo  关闭方法: 在对应窗口按 Ctrl+C，然后输入 Y 确认
echo.
pause
