#!/usr/bin/env python3
"""
main.py — Entry point for the NSE ORB Strategy Backtester.

Usage (CLI)
-----------
  # Interactive wizard (no arguments needed):
  python main.py

  # Fully scripted:
  python main.py --instrument "NSE NIFTY" --from-date 2024-01-01 --to-date 2024-03-31

  # Use local CSV files instead of Angel Broking API:
  python main.py --instrument "NSE BANKNIFTY" \\
                 --from-date 2024-01-01 --to-date 2024-01-31 \\
                 --csv --data-dir ./data

  # Save trade log to CSV:
  python main.py --instrument "NSE NIFTY" \\
                 --from-date 2024-01-01 --to-date 2024-03-31 \\
                 --save-csv

  # Verbose debug output:
  python main.py ... --verbose

TO-DO FOR LIVE TRADING (future phase):
  • Replace BacktestEngine with a LiveEngine that streams real-time ticks.
  • Replace CSVDataFetcher / AngelBrokingFetcher with a WebSocket-based feed.
  • Add a TelegramNotifier that fires on each TradeSignal.
  • All strategy logic (strategy.py) stays completely unchanged.
"""

import argparse
import logging
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 65


def _banner():
    print(SEPARATOR)
    print("   NSE ORB Strategy Backtester")
    print("   Opening Range Breakout + FVG + Volume Confirmation")
    print(SEPARATOR)
    print()


def _print_report(report: dict, instrument: str):
    if "error" in report:
        print(f"\n[ERROR] {report['error']}\n")
        return

    print(f"\n{SEPARATOR}")
    print(f"  BACKTEST RESULTS — {instrument}")
    print(SEPARATOR)

    print("\n  TRADE STATISTICS")
    print(f"  {'Total trades':<28} {report['total_trades']}")
    print(f"  {'Winning trades':<28} {report['winning_trades']}")
    print(f"  {'Losing trades':<28} {report['losing_trades']}")
    print(f"  {'Win rate':<28} {report['win_rate']} %")
    print(f"  {'Long trades':<28} {report['long_trades']}")
    print(f"  {'Short trades':<28} {report['short_trades']}")

    print("\n  P&L")
    print(f"  {'Total P&L (points)':<28} {report['total_pnl_points']}")
    print(f"  {'Total P&L (INR)':<28} Rs {report['total_pnl_amount']:,.2f}")
    print(f"  {'Avg win (points)':<28} {report['avg_win_points']}")
    print(f"  {'Avg loss (points)':<28} {report['avg_loss_points']}")
    print(f"  {'Expectancy (points)':<28} {report['expectancy']}")
    print(f"  {'Profit factor':<28} {report['profit_factor']}")

    print("\n  CAPITAL")
    print(f"  {'Initial capital':<28} Rs {report['initial_capital']:,.2f}")
    print(f"  {'Final capital':<28} Rs {report['final_capital']:,.2f}")
    print(f"  {'Net return':<28} {report['return_pct']} %")
    print(f"  {'Max drawdown':<28} Rs {report['max_drawdown']:,.2f}  ({report['max_drawdown_pct']} %)")

    print("\n  EXIT REASONS")
    ex = report["exit_reasons"]
    print(f"  {'Target hit':<28} {ex['target']}")
    print(f"  {'Stop hit':<28} {ex['stop']}")
    print(f"  {'End-of-day':<28} {ex['eod']}")

    # Monthly breakdown
    if report.get("monthly"):
        print("\n  MONTHLY BREAKDOWN")
        print(f"  {'Month':<10} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'P&L pts':>10} {'P&L INR':>13}")
        print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*10} {'-'*13}")
        for month, m in sorted(report["monthly"].items()):
            wr = m["wins"] / m["trades"] * 100 if m["trades"] else 0
            print(
                f"  {month:<10} {m['trades']:>7} {m['wins']:>6} {wr:>6.1f}%"
                f" {m['pnl_points']:>10.2f} Rs {m['pnl_amount']:>10,.2f}"
            )

    # Last 10 trades
    trades = report.get("trades", [])
    if trades:
        print("\n  TRADE LOG (last 10)")
        print(
            f"  {'Date':<12} {'Dir':<6} {'Entry':>8} {'Exit':>8}"
            f" {'Pts':>8} {'INR':>11} {'Reason'}"
        )
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*11} {'-'*7}")
        for t in trades[-10:]:
            dt    = t.entry_time.strftime("%Y-%m-%d")
            exit_ = f"{t.exit_price:.2f}" if t.exit_price else "N/A"
            print(
                f"  {dt:<12} {t.signal.direction:<6}"
                f" {t.entry_price:>8.2f} {exit_:>8}"
                f" {t.pnl_points:>8.2f} Rs {t.pnl_amount:>8,.2f}"
                f" {t.exit_reason or 'N/A'}"
            )

    print(f"\n{SEPARATOR}\n")


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    instrument: str,
    from_date_str: str,
    to_date_str: str,
    csv_mode: bool,
    data_dir: str,
    save_csv: bool,
):
    from config import (
        ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET,
        INSTRUMENTS, TRADING_CONFIG, BACKTEST_CONFIG,
    )
    from data_fetcher import AngelBrokingFetcher, CSVDataFetcher
    from strategy import ORBStrategy
    from backtest import BacktestEngine, export_trades_csv

    # Validate instrument
    if instrument not in INSTRUMENTS:
        logger.error("Unknown instrument '%s'. Available: %s", instrument, list(INSTRUMENTS))
        sys.exit(1)

    inst      = INSTRUMENTS[instrument]
    bt_config = dict(BACKTEST_CONFIG)
    bt_config["lot_size"] = BACKTEST_CONFIG["lot_size"].get(instrument, 50)

    # ── Data fetcher ──────────────────────────────────────────────────
    if csv_mode:
        logger.info("Data source: CSV files in '%s'", data_dir)
        fetcher = CSVDataFetcher(data_dir=data_dir)
    else:
        logger.info("Data source: Angel Broking Smart API")
        fetcher = AngelBrokingFetcher(
            api_key     = ANGEL_API_KEY,
            client_id   = ANGEL_CLIENT_ID,
            password    = ANGEL_PASSWORD,
            totp_secret = ANGEL_TOTP_SECRET,
        )

    if not fetcher.connect():
        logger.error("Failed to connect to data source")
        sys.exit(1)

    try:
        from_dt = datetime.strptime(from_date_str, "%Y-%m-%d").replace(hour=9,  minute=0)
        to_dt   = datetime.strptime(to_date_str,   "%Y-%m-%d").replace(hour=16, minute=0)

        logger.info(
            "Fetching 1-min data for %s  [%s → %s]",
            instrument, from_dt.date(), to_dt.date(),
        )

        df = fetcher.get_historical_data(
            symbol    = inst["trading_symbol"],
            exchange  = inst["exchange"],
            token     = inst["token"],
            interval  = "1min",
            from_date = from_dt,
            to_date   = to_dt,
        )

        if df.empty:
            logger.error(
                "No data returned. Check your API credentials and ensure the "
                "date range falls within market-open days."
            )
            sys.exit(1)

        logger.info("Bars fetched: %d", len(df))

        # ── Run backtest ──────────────────────────────────────────────
        strategy = ORBStrategy(TRADING_CONFIG)
        engine   = BacktestEngine(strategy, bt_config)
        report   = engine.run(
            all_data   = df,
            start_date = datetime.strptime(from_date_str, "%Y-%m-%d").date(),
            end_date   = datetime.strptime(to_date_str,   "%Y-%m-%d").date(),
        )

        _print_report(report, instrument)

        if save_csv and "trades" in report:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{instrument.replace(' ', '_')}_{ts}.csv"
            export_trades_csv(report, filename)
            print(f"Trade log saved to: {filename}\n")

    finally:
        fetcher.disconnect()


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------

def interactive_mode():
    _banner()

    instruments = ["NSE NIFTY", "NSE BANKNIFTY", "NSE HDFCBANK"]
    print("Select instrument to backtest:")
    for i, name in enumerate(instruments, 1):
        print(f"  {i}. {name}")

    while True:
        try:
            choice = int(input(f"\nEnter choice [1-{len(instruments)}]: ").strip())
            if 1 <= choice <= len(instruments):
                instrument = instruments[choice - 1]
                break
        except ValueError:
            pass
        print(f"  Please enter 1 to {len(instruments)}.")

    print(f"\nSelected: {instrument}")
    print("\nEnter the backtest date range (format: YYYY-MM-DD):")

    while True:
        from_date = input("  From date: ").strip()
        try:
            datetime.strptime(from_date, "%Y-%m-%d")
            break
        except ValueError:
            print("  Invalid format — use YYYY-MM-DD.")

    while True:
        to_date = input("  To date:   ").strip()
        try:
            datetime.strptime(to_date, "%Y-%m-%d")
            break
        except ValueError:
            print("  Invalid format — use YYYY-MM-DD.")

    print("\nData source:")
    print("  1. Angel Broking Smart API  (live credentials required)")
    print("  2. Local CSV files          (for offline testing)")

    while True:
        try:
            choice = int(input("\nEnter choice [1-2]: ").strip())
            if choice in [1, 2]:
                csv_mode = choice == 2
                break
        except ValueError:
            pass
        print("  Please enter 1 or 2.")

    data_dir = "./data"
    if csv_mode:
        data_dir = input(f"  CSV directory [{data_dir}]: ").strip() or data_dir

    save = input("\nSave trade log to CSV? [y/N]: ").strip().lower() == "y"

    run_backtest(
        instrument    = instrument,
        from_date_str = from_date,
        to_date_str   = to_date,
        csv_mode      = csv_mode,
        data_dir      = data_dir,
        save_csv      = save,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog        = "main.py",
        description = "NSE ORB Strategy Backtester — FVG + Volume Confirmation",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python main.py
  python main.py --instrument "NSE NIFTY" --from-date 2024-01-01 --to-date 2024-03-31
  python main.py --instrument "NSE BANKNIFTY" --from-date 2024-01-01 --to-date 2024-01-31 --csv
  python main.py --instrument "NSE NIFTY" --from-date 2024-01-01 --to-date 2024-03-31 --save-csv
        """,
    )

    parser.add_argument(
        "--instrument", "-i",
        choices = ["NSE NIFTY", "NSE BANKNIFTY", "NSE HDFCBANK"],
        help    = "Instrument to backtest",
    )
    parser.add_argument(
        "--from-date", dest="from_date",
        metavar = "YYYY-MM-DD",
        help    = "Backtest start date",
    )
    parser.add_argument(
        "--to-date", dest="to_date",
        metavar = "YYYY-MM-DD",
        help    = "Backtest end date",
    )
    parser.add_argument(
        "--csv",
        dest    = "csv_mode",
        action  = "store_true",
        help    = "Use local CSV files instead of the Angel Broking API",
    )
    parser.add_argument(
        "--data-dir",
        dest    = "data_dir",
        default = "./data",
        metavar = "DIR",
        help    = "Directory containing CSV data files (default: ./data)",
    )
    parser.add_argument(
        "--save-csv",
        dest   = "save_csv",
        action = "store_true",
        help   = "Export trade log to a timestamped CSV file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action = "store_true",
        help   = "Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level   = log_level,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    # If no flags supplied → interactive wizard
    if not args.instrument and not args.from_date:
        interactive_mode()
        return

    # CLI mode — validate required arguments
    missing = [
        flag for flag, val in [
            ("--instrument", args.instrument),
            ("--from-date",  args.from_date),
            ("--to-date",    args.to_date),
        ] if not val
    ]
    if missing:
        parser.error(f"The following arguments are required in CLI mode: {', '.join(missing)}")

    _banner()
    run_backtest(
        instrument    = args.instrument,
        from_date_str = args.from_date,
        to_date_str   = args.to_date,
        csv_mode      = args.csv_mode,
        data_dir      = args.data_dir,
        save_csv      = args.save_csv,
    )


if __name__ == "__main__":
    main()
