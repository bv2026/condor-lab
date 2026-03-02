# Condor Lab

Iron condor backtesting and live-quote tool for ETFs. Estimates premium
via Black-Scholes when historical option chains are unavailable.

## Requirements

- Python 3.9+
- [yfinance](https://pypi.org/project/yfinance/) - market data
- [pandas](https://pypi.org/project/pandas/) / [numpy](https://pypi.org/project/numpy/) / [scipy](https://pypi.org/project/scipy/) - data & math
- [openpyxl](https://pypi.org/project/openpyxl/) *(optional)* - Excel export (falls back to CSV)

## Installation

```bash
pip install yfinance pandas numpy scipy openpyxl
```

## Quick Start

```bash
# Weekly backtest (default)
python Main.py

# Daily-entry backtest, export to Excel
python Main.py --backtest --daily --export

# DTE comparison with daily entry
python Main.py --dte --daily

# Parameter sweep (weekly entry)
python Main.py --sweep --export

# Parameter sweep (daily entry)
python Main.py --sweep --daily --export

# Live market quotes
python Main.py --live
```

## Configuration Reference

All configuration lives as top-level constants in `Main.py`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `SYMBOLS` | dict | SPY, DIA, GLD, XLP | Symbols to trade and their `wing_pct` (% OTM for short strikes) |
| `EXPIRATION_SCHEDULES` | dict | varies | `daily`/`weekly`/`monthly` proxy for available expirations per symbol |
| `DTE_TOLERANCE` | int | `1` | Max calendar-day gap between target and nearest valid expiration |
| `START_DATE` / `END_DATE` | str | `2020-01-01` / `2026-02-10` | Backtest date range |
| `VIX_THRESHOLD_MIN` | float\|None | `None` | Skip entry if VIX is below this value (`None` = no floor) |
| `VIX_THRESHOLD_MAX` | float\|None | `17.0` | Skip entry if VIX is at or above this value (`None` = no ceiling) |
| `WING_WIDTH_DOLLARS` | float | `5.0` | Width of each spread wing in dollars (long strike distance from short) |
| `TARGET_DTE_CALENDAR` | int | `7` | Target DTE (calendar days) used for live quote fetching |
| `DTE_CALENDAR_VALUES` | list | `[3,5,7,10]` | DTE values tested in `--dte` and `--sweep` modes |
| `RISK_FREE_RATE` | float | `0.05` | Risk-free rate used in Black-Scholes pricing |
| `STOP_LOSS_MULTIPLIER` | float | `2.0` | Stop-loss triggers when unrealized loss reaches `credit * multiplier` |
| `STOP_LOSS_MODE` | str | `conservative` | `conservative` (touch short strike = loss) or `bs_estimated` (mark-to-market) |
| `WING_PCT_VALUES` | list | `[0.025...0.06]` | Wing % values swept in `--sweep` mode |

## CLI Usage

```
python Main.py [--backtest | --dte | --sweep | --compare | --live] [--daily] [--export]
```

### Modes

| Mode | Description |
|---|---|
| *(default)* / `--backtest` | Run backtest for all configured symbols |
| `--dte` | Compare multiple DTE holding periods side-by-side |
| `--sweep` | Grid search over wing% x DTE combinations per symbol |
| `--compare` | Run both weekly and daily backtests and print comparison table |
| `--live` | Fetch current market option quotes (no backtest) |

### Flags

| Flag | Description |
|---|---|
| `--daily` | Use daily entry (overlapping positions) instead of weekly (Friday-only). Composable with `--backtest`, `--dte`, and `--sweep`. |
| `--export` | Save results to Excel (`.xlsx`) or CSV files |

## Strategy Concepts

### Iron Condors

An iron condor sells an out-of-the-money (OTM) put spread and an OTM call
spread simultaneously, collecting a net credit. The position profits if the
underlying stays between the short strikes through expiration.

### Wing Percentage (`wing_pct`)

Distance from spot price to each short strike, expressed as a fraction (e.g.
`0.03` = 3% OTM). Higher values are further OTM: lower premium but higher
probability of profit.

### Wing Width (`WING_WIDTH_DOLLARS`)

Dollar distance between the short and long strike on each side. Defines the
maximum risk per spread. A $5 wing width means max loss per side is $500
minus credit received.

### DTE (Days to Expiration)

Calendar days from entry to target expiration. Shorter DTEs decay faster
(higher theta) but have less room for the position to recover from adverse
moves.

### VIX Filter

Entries are skipped when VIX is outside the configured range. High-VIX
environments widen expected moves and increase the chance of breaching
short strikes, even though premiums are richer.

### Stop-Loss Modes

- **`conservative`**: Position is stopped out (loss) the moment intraday
  high/low touches a short strike. Simple and deterministic.
- **`bs_estimated`**: Re-prices the position via Black-Scholes each day.
  Stops when mark-to-market loss exceeds `credit * STOP_LOSS_MULTIPLIER`.
  Also includes a 50%-of-credit take-profit exit.

### Expiration Schedules

Since historical option chain data is not available, the backtest uses a
conservative proxy for each symbol's expiration availability:

- **`daily`** - Mon/Wed/Fri expirations (effectively always available)
- **`weekly`** - Friday expirations only
- **`monthly`** - 3rd Friday of each month only (most restrictive)

Unlisted symbols default to `monthly` to avoid overstating opportunity.

## Understanding Output

The backtest summary table includes these columns:

| Column | Meaning |
|---|---|
| Symbol | Ticker |
| OTM% | Wing percentage used for short strikes |
| Trades | Total number of trades entered |
| WinRate | Percentage of trades that expired profitably |
| AvgCred | Average credit received per trade (in dollars, per contract) |
| R/R | Risk/reward ratio (average max loss / average credit) |
| MaxDD | Maximum drawdown in dollar terms, computed from realized exit order |
| LoseStrk | Longest consecutive losing streak |
| WinStrk | Longest consecutive winning streak |
| Total P&L | Cumulative profit/loss across all trades |
| Score | Streak-penalized score (sweep mode only) |

## Parameter Sweep Scoring

The sweep ranks parameter combinations using a streak-penalized score:

```
score = total_pnl * 0.85^max_losing_streak    (if total_pnl > 0)
score = total_pnl                              (if total_pnl <= 0)
```

The 0.85 decay factor penalizes strategies with long losing streaks, even if
their raw P&L is high. This balances profitability with consistency:

| Losing Streak | Multiplier | Effect |
|---|---|---|
| 0 | 1.00x | Full P&L |
| 3 | 0.61x | 39% penalty |
| 5 | 0.44x | 56% penalty |
| 10 | 0.20x | 80% penalty |
| 15 | 0.09x | 91% penalty |
| 20 | 0.04x | 96% penalty |

A strategy earning $10,000 with a 5-trade losing streak scores $4,400, while
one earning $7,000 with only a 2-trade streak scores $5,060 - ranking higher
despite lower raw P&L.

## Examples

```bash
# Weekly backtest (default)
python Main.py

# Daily-entry backtest, export to Excel
python Main.py --backtest --daily --export

# DTE comparison with daily entry
python Main.py --dte --daily

# Parameter sweep (weekly entry)
python Main.py --sweep --export

# Parameter sweep (daily entry)
python Main.py --sweep --daily --export

# Live market quotes
python Main.py --live
```
