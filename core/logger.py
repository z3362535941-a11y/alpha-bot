import logging
import os
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

TRADE_LOG: list = []


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(fh)

    return logger


def log_trade(bot_name: str, action: str, symbol: str, price: float,
              qty: float, pnl: float = 0.0, reason: str = ""):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "bot": bot_name,
        "action": action,
        "symbol": symbol,
        "price": price,
        "qty": qty,
        "pnl": pnl,
        "reason": reason,
    }
    TRADE_LOG.append(entry)
    color = Fore.GREEN if pnl >= 0 else Fore.RED
    pnl_str = f"{color}PnL={pnl:+.2f}$" if pnl != 0.0 else ""
    print(f"  {Fore.YELLOW}[TRADE]{Style.RESET_ALL} {bot_name} | {action:4s} {symbol} @ {price:.4f} qty={qty:.4f} {pnl_str} {reason}")
