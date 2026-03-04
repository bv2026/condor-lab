import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- STRATEGY SETTINGS (Tweak these to test!) ---
# Multiple symbols with their specific settings
SYMBOLS = {
    'SPY': {'wing_pct': 0.03},    # 3.0% OTM for SPY
    'DIA': {'wing_pct': 0.03},    # 3.0% OTM for DIA (Dow Jones)
    'GLD': {'wing_pct': 0.035},   # 3.5% OTM for GLD
    'XLP': {'wing_pct': 0.035},   # 3.5% OTM for QQQ
    'XLU': {'wing_pct': 0.03},    # 3.0% OTM for XLU
    'IWM': {'wing_pct': 0.035},   # 3.5% OTM for IWM (Russell 2000)
    'SLV': {'wing_pct': 0.04},    # 4.0% OTM for SLV (Silver - volatile)
}

# Conservative expiration-availability proxy per symbol.
# Exact historical option chains are unavailable, so these proxies approximate
# each symbol's real expiration schedule based on known listing conventions.
# In daily-entry mode the backtest will skip entry dates
# where no viable expiration exists within DTE_TOLERANCE of the target.
#
# Schedule types:
#   'daily'   - Expirations Mon/Wed/Fri (0-DTE capable). Effectively always available.
#   'weekly'  - Expirations every Friday only. Entry allowed when a Friday falls
#               within DTE_TOLERANCE calendar days of entry_date + target_dte.
#   'monthly' - Expirations on the 3rd Friday of each month only. Most restrictive;
#               entry allowed only when a 3rd-Friday expiration is within tolerance.
#
# These are deliberately conservative: real availability may be broader for liquid
# names, but overstating opportunity is worse than understating it in a backtest.
EXPIRATION_SCHEDULES = {
    'SPY': 'daily',     # Mon/Wed/Fri expirations
    'QQQ': 'daily',     # Mon/Wed/Fri expirations
    'IWM': 'daily',     # Mon/Wed/Fri expirations
    'DIA': 'weekly',    # Friday weeklys
    'GLD': 'weekly',    # Friday weeklys
    'TLT': 'weekly',    # Friday weeklys
    'XLP': 'monthly',   # 3rd Friday monthlys
    'XLU': 'monthly',   # 3rd Friday monthlys
    'SLV': 'monthly',   # 3rd Friday monthlys
    'IEF': 'monthly',   # 3rd Friday monthlys
}

# Maximum calendar-day gap between the ideal target expiration and the nearest
# valid expiration for the entry to be accepted. Keeps the actual DTE close to
# the configured target so backtest assumptions remain valid.
DTE_TOLERANCE = 1

START_DATE = '2020-01-01'  # Long duration to find all "dormant" periods
END_DATE = '2026-02-10'

VIX_THRESHOLD_MIN = None   # Optional floor (None = no minimum)
VIX_THRESHOLD_MAX = 17.0   # Ceiling (None = no upper limit)
WING_WIDTH_DOLLARS = 5.0   # Width of wings in dollars
TARGET_DTE_CALENDAR = 7    # Target DTE in calendar days for live quote retrieval

# DTE values to test (calendar days)
DTE_CALENDAR_VALUES = [3, 5, 7, 10]    # Holding periods in calendar days

# REALISTIC PREMIUM ESTIMATION
RISK_FREE_RATE = 0.05      # ~5% risk-free rate
STOP_LOSS_MULTIPLIER = 2.0 # Lose 150% of credit on stop loss
STOP_LOSS_MODE = 'conservative'  # 'conservative' or 'bs_estimated'

# Parameter Sweep
WING_PCT_VALUES = [0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]

def vix_in_range(vix_value):
    """Check if VIX is within the configured range."""
    if VIX_THRESHOLD_MIN is not None and vix_value < VIX_THRESHOLD_MIN:
        return False
    if VIX_THRESHOLD_MAX is not None and vix_value >= VIX_THRESHOLD_MAX:
        return False
    return True

def format_vix_range():
    """Return a human-readable string for the current VIX filter."""
    if VIX_THRESHOLD_MIN is None and VIX_THRESHOLD_MAX is None:
        return "VIX: No filter"
    elif VIX_THRESHOLD_MIN is None:
        return f"VIX < {VIX_THRESHOLD_MAX}"
    elif VIX_THRESHOLD_MAX is None:
        return f"VIX >= {VIX_THRESHOLD_MIN}"
    else:
        return f"VIX: {VIX_THRESHOLD_MIN}-{VIX_THRESHOLD_MAX}"

def _third_friday(year, month):
    """Return the 3rd Friday of the given month as a datetime."""
    first_day = datetime(year, month, 1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    return first_day + timedelta(days=first_friday_offset + 14)


def has_viable_expiration(symbol, entry_date, target_dte):
    """Check if a symbol has a viable options expiration for the target DTE.

    Uses the conservative calendar proxy in EXPIRATION_SCHEDULES.  Symbols not
    listed default to 'monthly' (the most restrictive schedule) so the backtest
    never overstates opportunity for unknown symbols.

    Returns True when a valid expiration falls within DTE_TOLERANCE calendar
    days of entry_date + target_dte.
    """
    schedule = EXPIRATION_SCHEDULES.get(symbol, 'monthly')

    ideal_expiry = entry_date + timedelta(days=target_dte)
    ideal_weekday = ideal_expiry.weekday()  # 0=Mon ... 4=Fri

    if schedule == 'daily':
        # Mon/Wed/Fri expirations - always within 1 day of any business day
        return True

    if schedule == 'weekly':
        # Friday-only expirations
        days_to_friday = (4 - ideal_weekday) % 7
        nearest_gap = min(days_to_friday, 7 - days_to_friday) if days_to_friday != 0 else 0
        return nearest_gap <= DTE_TOLERANCE

    if schedule == 'monthly':
        # 3rd-Friday-only expirations - check current and adjacent months
        y, m = ideal_expiry.year, ideal_expiry.month
        for month_offset in (0, -1, 1):
            adj_m = m + month_offset
            adj_y = y
            if adj_m < 1:
                adj_m += 12
                adj_y -= 1
            elif adj_m > 12:
                adj_m -= 12
                adj_y += 1
            tf = _third_friday(adj_y, adj_m)
            if abs((ideal_expiry - tf).days) <= DTE_TOLERANCE:
                return True
        return False

    # Unknown schedule - conservative default: reject
    return False


def estimate_option_premium(spot_price, strike, dte_calendar_days, volatility, is_call=True):
    """
    Estimate option premium using Black-Scholes approximation.
    dte_calendar_days: days to expiration in calendar days
    volatility should be in decimal form (e.g., 0.15 for 15%)
    """
    if dte_calendar_days <= 0:
        return 0.0

    T = dte_calendar_days / 365.0
    S = spot_price
    K = strike
    r = RISK_FREE_RATE
    sigma = volatility

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0.01)  # Minimum $0.01
    except:
        return 0.10  # Fallback

def estimate_iron_condor_credit(symbol, spot_price, vix, wing_pct, wing_dollars, dte_calendar_days):
    """
    Estimate total credit for an iron condor using Black-Scholes.
    dte_calendar_days: days to expiration in calendar days
    Used for backtesting when historical options data isn't available.
    """
    # Calibrated multipliers based on real options data
    vol_multiplier = {
        'SPY': 1.0,
        'GLD': 1.30,
        'SLV': 1.50,
        'TLT': 1.10,
        'IEF': 0.70,
        'XLU': 0.90,
        'QQQ': 1.10,
    }

    multiplier = vol_multiplier.get(symbol, 1.0)
    base_vol = (vix / 100.0) * multiplier
    smile_adjustment = 1.0 + (wing_pct * 12)
    implied_vol = base_vol * smile_adjustment

    short_put = spot_price * (1 - wing_pct)
    short_call = spot_price * (1 + wing_pct)
    long_put = short_put - wing_dollars
    long_call = short_call + wing_dollars

    short_put_premium = estimate_option_premium(spot_price, short_put, dte_calendar_days, implied_vol, is_call=False)
    long_put_premium = estimate_option_premium(spot_price, long_put, dte_calendar_days, implied_vol, is_call=False)
    short_call_premium = estimate_option_premium(spot_price, short_call, dte_calendar_days, implied_vol, is_call=True)
    long_call_premium = estimate_option_premium(spot_price, long_call, dte_calendar_days, implied_vol, is_call=True)

    put_spread_credit = short_put_premium - long_put_premium
    call_spread_credit = short_call_premium - long_call_premium
    total_credit = put_spread_credit + call_spread_credit

    max_credit = wing_dollars * 100
    return max(min(total_credit * 100, max_credit), 0.0)


def evaluate_position_pnl(symbol, entry_price, current_spot, current_vix, wing_pct, wing_dollars, remaining_dte_calendar):
    """
    Re-evaluate an iron condor's current value using Black-Scholes with original strikes.

    remaining_dte_calendar: remaining days to expiration in calendar days
    Returns estimated current credit value of the position (what it would cost to buy back).
    Compare with initial_credit to determine P&L.
    """
    if remaining_dte_calendar <= 0:
        # At expiration: intrinsic value only
        short_put = entry_price * (1 - wing_pct)
        short_call = entry_price * (1 + wing_pct)
        long_put = short_put - wing_dollars
        long_call = short_call + wing_dollars

        put_spread_value = max(short_put - current_spot, 0) - max(long_put - current_spot, 0)
        call_spread_value = max(current_spot - short_call, 0) - max(current_spot - long_call, 0)
        return (put_spread_value + call_spread_value) * 100

    # Calibrated volatility (same logic as estimate_iron_condor_credit)
    vol_multiplier = {
        'SPY': 1.0, 'GLD': 1.30, 'SLV': 1.50, 'TLT': 1.10,
        'IEF': 0.70, 'XLU': 0.90, 'QQQ': 1.10,
    }
    multiplier = vol_multiplier.get(symbol, 1.0)
    base_vol = (current_vix / 100.0) * multiplier
    smile_adjustment = 1.0 + (wing_pct * 12)
    implied_vol = base_vol * smile_adjustment

    # Original strikes (fixed at entry)
    short_put = entry_price * (1 - wing_pct)
    short_call = entry_price * (1 + wing_pct)
    long_put = short_put - wing_dollars
    long_call = short_call + wing_dollars

    # Reprice each leg with CURRENT spot but ORIGINAL strikes
    short_put_premium = estimate_option_premium(current_spot, short_put, remaining_dte_calendar, implied_vol, is_call=False)
    long_put_premium = estimate_option_premium(current_spot, long_put, remaining_dte_calendar, implied_vol, is_call=False)
    short_call_premium = estimate_option_premium(current_spot, short_call, remaining_dte_calendar, implied_vol, is_call=True)
    long_call_premium = estimate_option_premium(current_spot, long_call, remaining_dte_calendar, implied_vol, is_call=True)

    put_spread_value = short_put_premium - long_put_premium
    call_spread_value = short_call_premium - long_call_premium
    current_value = (put_spread_value + call_spread_value) * 100

    return max(current_value, 0)


def fetch_live_iron_condor_credit_ts(symbol, wing_pct, wing_dollars, target_dte_calendar=7):
    """
    Fetch LIVE iron condor credit from the TradeStation REST API (v3).

    Requires the TS_ACCESS_TOKEN environment variable (OAuth Bearer token).
    Set it with:  export TS_ACCESS_TOKEN="<your_token>"

    Returns a details dict identical to fetch_live_iron_condor_credit, or None
    if the token is missing, a symbol has no viable expiration, or the condor
    would result in a net debit.
    """
    import os
    import requests as _requests

    token = os.environ.get('TS_ACCESS_TOKEN')
    if not token:
        return None  # No token - caller will fall back to yfinance

    BASE = "https://api.tradestation.com/v3"
    hdrs = {"Authorization": f"Bearer {token}"}

    try:
        # 1. Spot price
        resp = _requests.get(f"{BASE}/marketdata/quotes/{symbol}", headers=hdrs, timeout=10)
        resp.raise_for_status()
        quotes = resp.json().get('Quotes', [])
        if not quotes:
            return None
        spot = float(quotes[0]['Last'])

        # 2. Option expirations
        resp = _requests.get(f"{BASE}/marketdata/options/expirations",
                             params={"underlying": symbol}, headers=hdrs, timeout=10)
        resp.raise_for_status()
        expirations = resp.json().get('Expirations', [])
        if not expirations:
            return None

        # 3. Find expiration closest to target DTE
        today = datetime.now()
        best_exp = min(
            expirations,
            key=lambda e: abs((datetime.fromisoformat(e['Date'].replace('Z', '')) - today).days
                              - target_dte_calendar)
        )
        exp_date = datetime.fromisoformat(best_exp['Date'].replace('Z', ''))
        actual_dte = (exp_date - today).days
        exp_str = exp_date.strftime('%y%m%d')

        # 4. Compute strikes (same logic as yfinance version)
        short_put_strike  = round(spot * (1 - wing_pct))
        long_put_strike   = short_put_strike - wing_dollars
        short_call_strike = round(spot * (1 + wing_pct))
        long_call_strike  = short_call_strike + wing_dollars

        sp_sym = f"{symbol} {exp_str}P{int(short_put_strike)}"
        lp_sym = f"{symbol} {exp_str}P{int(long_put_strike)}"
        sc_sym = f"{symbol} {exp_str}C{int(short_call_strike)}"
        lc_sym = f"{symbol} {exp_str}C{int(long_call_strike)}"

        # 5. Option spread quotes
        resp = _requests.post(
            f"{BASE}/marketdata/options/quotes",
            headers=hdrs,
            json={"Legs": [
                {"Symbol": sp_sym, "Ratio": -1},
                {"Symbol": lp_sym, "Ratio":  1},
                {"Symbol": sc_sym, "Ratio": -1},
                {"Symbol": lc_sym, "Ratio":  1},
            ]},
            timeout=10,
        )
        resp.raise_for_status()
        sq = resp.json().get('Quote', {})

        # Negative Mid means we receive credit (our legs: sell spread)
        mid = float(sq.get('Mid', 0))
        total_credit = -mid * 100   # convert to dollars per contract

        if total_credit <= 0:
            return None             # Net debit - skip

        max_loss = (wing_dollars * 100) - total_credit

        return {
            'symbol':       symbol,
            'spot':         spot,
            'expiration':   exp_date.strftime('%Y-%m-%d'),
            'dte_calendar': actual_dte,
            'short_put':    short_put_strike,
            'long_put':     long_put_strike,
            'short_call':   short_call_strike,
            'long_call':    long_call_strike,
            'put_credit':   0,      # not broken out in spread quote
            'call_credit':  0,
            'total_credit': total_credit,
            'max_loss':     max_loss,
            'risk_reward':  max_loss / total_credit if total_credit > 0 else 999,
            'iv':           float(sq.get('ImpliedVolatility', 0)),
            'source':       'TradeStation',
        }

    except Exception as e:
        print(f"  TradeStation error for {symbol}: {e}")
        return None


def fetch_live_iron_condor_credit(symbol, wing_pct, wing_dollars, target_dte_calendar=7):
    """
    Fetch LIVE options data to get real iron condor credit.
    Uses TradeStation REST API when TS_ACCESS_TOKEN env var is set,
    falls back to yfinance otherwise.
    target_dte_calendar: target days to expiration in calendar days
    Returns: details_dict or None if failed
    """
    # Try TradeStation first (richer data, real-time)
    ts_result = fetch_live_iron_condor_credit_ts(symbol, wing_pct, wing_dollars, target_dte_calendar)
    if ts_result is not None:
        return ts_result
    try:
        ticker = yf.Ticker(symbol)
        spot = ticker.history(period='1d')['Close'].iloc[-1]

        short_put_strike = spot * (1 - wing_pct)
        short_call_strike = spot * (1 + wing_pct)
        long_put_strike = short_put_strike - wing_dollars
        long_call_strike = short_call_strike + wing_dollars

        # Find expiration closest to target DTE
        exps = ticker.options
        if not exps:
            return None

        today = datetime.now()
        best_exp = None
        best_diff = float('inf')

        for exp in exps:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            diff = abs((exp_date - today).days - target_dte_calendar)
            if diff < best_diff:
                best_diff = diff
                best_exp = exp

        actual_dte_calendar = (datetime.strptime(best_exp, '%Y-%m-%d') - today).days
        chain = ticker.option_chain(best_exp)

        # Find closest strikes and get mid prices
        puts = chain.puts
        calls = chain.calls

        # Short put
        sp_row = puts.iloc[(puts['strike'] - short_put_strike).abs().argsort()[:1]]
        sp_mid = (sp_row['bid'].values[0] + sp_row['ask'].values[0]) / 2
        sp_strike = sp_row['strike'].values[0]

        # Long put
        lp_row = puts.iloc[(puts['strike'] - long_put_strike).abs().argsort()[:1]]
        lp_mid = (lp_row['bid'].values[0] + lp_row['ask'].values[0]) / 2
        lp_strike = lp_row['strike'].values[0]

        # Short call
        sc_row = calls.iloc[(calls['strike'] - short_call_strike).abs().argsort()[:1]]
        sc_mid = (sc_row['bid'].values[0] + sc_row['ask'].values[0]) / 2
        sc_strike = sc_row['strike'].values[0]

        # Long call
        lc_row = calls.iloc[(calls['strike'] - long_call_strike).abs().argsort()[:1]]
        lc_mid = (lc_row['bid'].values[0] + lc_row['ask'].values[0]) / 2
        lc_strike = lc_row['strike'].values[0]

        # Calculate credits
        put_spread_credit = sp_mid - lp_mid
        call_spread_credit = sc_mid - lc_mid
        total_credit = (put_spread_credit + call_spread_credit) * 100

        # Calculate max loss
        put_width = sp_strike - lp_strike
        call_width = lc_strike - sc_strike
        max_width = max(put_width, call_width) * 100
        max_loss = max_width - total_credit

        details = {
            'spot': spot,
            'expiration': best_exp,
            'dte_calendar': actual_dte_calendar,
            'short_put': sp_strike,
            'long_put': lp_strike,
            'short_call': sc_strike,
            'long_call': lc_strike,
            'put_credit': put_spread_credit * 100,
            'call_credit': call_spread_credit * 100,
            'total_credit': total_credit,
            'max_loss': max_loss,
            'risk_reward': max_loss / total_credit if total_credit > 0 else 999
        }

        return details

    except Exception as e:
        print(f"  Error fetching live data for {symbol}: {e}")
        return None

def fetch_symbol_data(symbol):
    """Fetch and return historical data for a symbol + VIX. Returns DataFrame or None."""
    data = yf.download([symbol, '^VIX'], start=START_DATE, end=END_DATE, progress=False)
    df = pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            df['Close'] = data['Close'][symbol]
            df['High'] = data['High'][symbol]
            df['Low'] = data['Low'][symbol]
            df['VIX'] = data['Close']['^VIX']
        else:
            return None
    except Exception as e:
        return None
    return df

def run_backtest_single(symbol, wing_pct, daily_entry=False, dte_calendar=7, df=None):
    """Run backtest for a single symbol. Returns results dict.

    If daily_entry=True, opens a new iron condor every day (overlapping positions).
    If daily_entry=False (default), only opens on Fridays (one position at a time).
    dte_calendar: Days to expiration in calendar days (holding period)
    df: Pre-fetched DataFrame (optional, avoids redundant downloads in sweep mode)
    """

    # Fetch Data if not provided
    if df is None:
        df = fetch_symbol_data(symbol)
        if df is None:
            return None

    # Entry filter: every day or just Fridays
    if daily_entry:
        potential_entries = df.copy()  # Every trading day
    else:
        potential_entries = df[df.index.dayofweek == 4].copy()  # Fridays only

    trades = []
    total_credits = []
    total_max_losses = []
    pnl_entry_order = []  # Track P&L in entry order (for reference)

    for date, row in potential_entries.iterrows():
        try:
            entry_vix = float(row['VIX'])
            if not vix_in_range(entry_vix):
                continue

            # Skip dates where the symbol lacks a viable expiration near
            # the target DTE (e.g. monthly-only chains on a non-3rd-Friday).
            if not has_viable_expiration(symbol, date, dte_calendar):
                continue

            entry_price = float(row['Close'])
            short_call = entry_price * (1 + wing_pct)
            short_put = entry_price * (1 - wing_pct)

            estimated_credit = estimate_iron_condor_credit(
                symbol, entry_price, entry_vix, wing_pct, WING_WIDTH_DOLLARS, dte_calendar
            )
            max_loss = max(0, (WING_WIDTH_DOLLARS * 100) - estimated_credit)
            stop_loss_amount = estimated_credit * STOP_LOSS_MULTIPLIER

            current_loc = df.index.get_loc(date)
            expiry_date = date + timedelta(days=dte_calendar)
            result = 'WIN'
            pnl = estimated_credit
            exit_date = expiry_date  # default: held to expiry

            # Step through trading days up to the calendar-day expiry
            for i in range(1, len(df) - current_loc):
                next_loc = current_loc + i
                if next_loc >= len(df):
                    break
                trade_date = df.index[next_loc]
                if trade_date > expiry_date:
                    break

                day_data = df.iloc[next_loc]
                day_high = float(day_data['High'])
                day_low = float(day_data['Low'])

                if STOP_LOSS_MODE == 'conservative':
                    # Current behavior: loss if price touches short strike
                    if day_high >= short_call or day_low <= short_put:
                        result = 'LOSS'
                        pnl = min(-min(stop_loss_amount, max_loss), 0)
                        exit_date = trade_date
                        break

                elif STOP_LOSS_MODE == 'bs_estimated':
                    remaining_dte_calendar = (expiry_date - trade_date).days
                    day_vix = float(day_data['VIX'])

                    # Check both sides: high threatens calls, low threatens puts
                    value_at_high = evaluate_position_pnl(
                        symbol, entry_price, day_high, day_vix,
                        wing_pct, WING_WIDTH_DOLLARS, remaining_dte_calendar
                    )
                    value_at_low = evaluate_position_pnl(
                        symbol, entry_price, day_low, day_vix,
                        wing_pct, WING_WIDTH_DOLLARS, remaining_dte_calendar
                    )
                    # Higher value = more expensive to close = worse for us
                    current_value = max(value_at_high, value_at_low)

                    # P&L = initial_credit - current_value (cost to close)
                    current_pnl = estimated_credit - current_value

                    # Stop loss: if loss exceeds threshold
                    if current_pnl <= -stop_loss_amount:
                        result = 'LOSS'
                        pnl = min(-min(stop_loss_amount, max_loss), 0)
                        exit_date = trade_date
                        break

                    # Take profit: position value decayed to < 50% of credit
                    day_close = float(day_data['Close'])
                    close_value = evaluate_position_pnl(
                        symbol, entry_price, day_close, day_vix,
                        wing_pct, WING_WIDTH_DOLLARS, remaining_dte_calendar
                    )
                    close_pnl = estimated_credit - close_value
                    if close_pnl >= estimated_credit * 0.5:
                        result = 'WIN'
                        pnl = close_pnl  # Take actual profit at TP
                        exit_date = trade_date
                        break

                else:
                    raise ValueError(f"Unknown STOP_LOSS_MODE: {STOP_LOSS_MODE}")

            trades.append({
                'Date': date,
                'ExitDate': exit_date,
                'Result': result,
                'P&L': pnl,
                'Credit': estimated_credit,
                'MaxLoss': max_loss
            })
            total_credits.append(estimated_credit)
            total_max_losses.append(max_loss)
            pnl_entry_order.append(pnl)

        except:
            continue

    if not trades:
        return None

    results = pd.DataFrame(trades)
    total_pnl = results['P&L'].sum()
    win_count = len(results[results['Result'] == 'WIN'])
    loss_count = len(results[results['Result'] == 'LOSS'])
    total_trades = len(results)
    win_rate = win_count / total_trades
    avg_credit = np.mean(total_credits)
    avg_max_loss = np.mean(total_max_losses)

    # Calculate max concurrent positions (for daily entry)
    # Convert calendar days to approximate trading days for concurrency estimate
    trading_days_approx = max(1, round(dte_calendar * 5 / 7))
    max_concurrent = min(trading_days_approx, total_trades) if daily_entry else 1

    # Sort trades by realized exit date for accurate P&L sequencing.
    # With overlapping positions (daily entry), trades entered later may
    # exit before trades entered earlier; exit order determines when each
    # P&L is actually realized and therefore drives drawdown and streak.
    trades_by_exit = sorted(trades, key=lambda t: (t['ExitDate'], t['Date']))
    pnl_by_exit = [t['P&L'] for t in trades_by_exit]

    # Calculate max drawdown (realized exit order)
    cumulative_pnl = np.cumsum(pnl_by_exit)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Find worst losing and winning streaks (realized exit order)
    losing_streak = 0
    max_losing_streak = 0
    winning_streak = 0
    max_winning_streak = 0
    for trade in trades_by_exit:
        if trade['Result'] == 'LOSS':
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
            winning_streak = 0
        else:
            winning_streak += 1
            max_winning_streak = max(max_winning_streak, winning_streak)
            losing_streak = 0

    return {
        'symbol': symbol,
        'wing_pct': wing_pct,
        'dte_calendar': dte_calendar,
        'trades': total_trades,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': win_rate,
        'avg_credit': avg_credit,
        'avg_max_loss': avg_max_loss,
        'risk_reward': avg_max_loss / avg_credit if avg_credit > 0 else 999,
        'total_pnl': total_pnl,
        'max_concurrent': max_concurrent,
        'daily_entry': daily_entry,
        'max_drawdown': max_drawdown,
        'max_losing_streak': max_losing_streak,
        'max_winning_streak': max_winning_streak
    }


def fetch_live_quotes():
    """Fetch live options quotes for all symbols."""
    import os
    source_label = "TradeStation" if os.environ.get('TS_ACCESS_TOKEN') else "yfinance"

    print("\n" + "="*80)
    print(f"              LIVE OPTIONS QUOTES  [source: {source_label}]")
    print("="*80)
    print(f"{'Symbol':<8} {'Spot':>10} {'Exp':>12} {'DTE':>5} {'Credit':>10} {'MaxLoss':>10} {'R/R':>7} {'IV':>7}")
    print("-"*80)

    live_results = []

    for symbol, settings in SYMBOLS.items():
        wing_pct = settings['wing_pct']
        details = fetch_live_iron_condor_credit(symbol, wing_pct, WING_WIDTH_DOLLARS, TARGET_DTE_CALENDAR)

        if details:
            iv_str = f"{details.get('iv', 0)*100:5.1f}%" if details.get('iv') else "  n/a"
            print(f"{symbol:<8} ${details['spot']:>8.2f} {details['expiration']:>12} {details['dte_calendar']:>5} "
                  f"${details['total_credit']:>8.2f} ${details['max_loss']:>8.2f} "
                  f"{details['risk_reward']:>6.1f}:1 {iv_str:>7}")
            live_results.append(details)
        else:
            print(f"{symbol:<8} {'ERROR - No options data available':<60}")

    print("-"*80)

    # Find best opportunity (lowest R/R = most efficient credit)
    if live_results:
        best = min(live_results, key=lambda x: x['risk_reward'])
        sym = best.get('symbol', '')
        print(f"\nBEST: {sym}  Spot ${best['spot']:.2f}  |  Credit ${best['total_credit']:.2f}"
              f"  |  Max Loss ${best['max_loss']:.2f}  |  R/R {best['risk_reward']:.1f}:1")
        print(f"   Strikes: {best['long_put']:.0f}/{best['short_put']:.0f}"
              f" x {best['short_call']:.0f}/{best['long_call']:.0f}"
              f"   Exp: {best['expiration']}  ({best['dte_calendar']} DTE)")

    return live_results


def run_all_backtests(daily_entry=False, dte_calendar=7):
    """Run backtests for all symbols and compare."""
    mode = "DAILY ENTRY" if daily_entry else "WEEKLY (Friday Only)"

    print("="*90)
    print(f"              IRON CONDOR BACKTEST COMPARISON - {mode} - {dte_calendar} cal-day DTE")
    print("="*90)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Filter: {format_vix_range()}")
    print(f"Wing Width: ${WING_WIDTH_DOLLARS} | DTE: {dte_calendar} calendar days")
    print(f"Stop Loss: {STOP_LOSS_MULTIPLIER}x credit")
    print(f"Stop Loss Mode: {STOP_LOSS_MODE}")
    if daily_entry:
        trading_days_approx = max(1, round(dte_calendar * 5 / 7))
        print(f"WARNING:  Max Concurrent Positions: Up to {trading_days_approx} overlapping trades")
    print("-"*90)
    print("Fetching historical data...")

    all_results = []

    for symbol, settings in SYMBOLS.items():
        wing_pct = settings['wing_pct']
        result = run_backtest_single(symbol, wing_pct, daily_entry=daily_entry, dte_calendar=dte_calendar)
        if result:
            all_results.append(result)

    if not all_results:
        print("No results found.")
        return

    # Sort by total P&L
    all_results.sort(key=lambda x: x['total_pnl'], reverse=True)

    print("\n" + "="*90)
    print(f"                    BACKTEST RESULTS SUMMARY - {mode} - {dte_calendar} cal-day DTE")
    print("="*90)
    print(f"{'Symbol':<6} {'OTM%':>5} {'Trades':>7} {'WinRate':>8} {'AvgCred':>9} {'R/R':>6} {'MaxDD':>10} {'LoseStrk':>8} {'WinStrk':>8} {'Total P&L':>12}")
    print("-"*90)

    for r in all_results:
        verdict = "[+]" if r['total_pnl'] > 0 else "[-]"
        print(f"{r['symbol']:<6} {r['wing_pct']*100:>4.1f}% {r['trades']:>7} {r['win_rate']:>7.1%} "
              f"${r['avg_credit']:>7.0f} {r['risk_reward']:>5.1f}:1 ${r['max_drawdown']:>8.0f} {r['max_losing_streak']:>8} "
              f"{r['max_winning_streak']:>8} ${r['total_pnl']:>10.0f} {verdict}")

    print("-"*90)

    # Find best performer
    best = all_results[0]
    print(f"\nBEST: BEST BACKTEST: {best['symbol']}")
    print(f"   Win Rate: {best['win_rate']:.1%} | Avg Credit: ${best['avg_credit']:.2f}")
    print(f"   Risk/Reward: {best['risk_reward']:.1f}:1 | Total P&L: ${best['total_pnl']:.2f}")
    print(f"   Max Drawdown: ${best['max_drawdown']:.2f} | Max Losing Streak: {best['max_losing_streak']} | Max Winning Streak: {best['max_winning_streak']}")

    if daily_entry:
        # Calculate capital requirements
        max_risk_per_trade = best['avg_max_loss']
        trading_days_approx = max(1, round(dte_calendar * 5 / 7))
        max_concurrent = trading_days_approx
        max_capital_needed = max_risk_per_trade * max_concurrent
        print(f"\n CAPITAL REQUIREMENTS (for {best['symbol']}):")
        print(f"   Max Risk per Trade: ${max_risk_per_trade:.2f}")
        print(f"   Max Concurrent Positions: {max_concurrent}")
        print(f"   Max Capital at Risk: ${max_capital_needed:.2f}")
        print(f"   Max Drawdown Seen: ${best['max_drawdown']:.2f}")
        print(f"   Suggested Account Size: ${max(max_capital_needed, best['max_drawdown']) * 2:.2f} (2x buffer)")

    return all_results


def run_dte_comparison(daily_entry=True):
    """Compare different DTE values (calendar days) for all symbols."""
    print("\n" + "="*90)
    print("              DTE COMPARISON - Finding Optimal Holding Period (calendar days)")
    print("="*90)
    print(f"Testing DTE values (calendar days): {DTE_CALENDAR_VALUES}")
    print(f"Mode: {'Daily Entry' if daily_entry else 'Weekly Entry'}")
    print(f"Filter: {format_vix_range()}")
    print("-"*90)

    all_dte_results = {}

    for dte_calendar in DTE_CALENDAR_VALUES:
        print(f"\nTesting {dte_calendar} calendar-day DTE...")
        all_dte_results[dte_calendar] = {}

        for symbol, settings in SYMBOLS.items():
            wing_pct = settings['wing_pct']
            result = run_backtest_single(symbol, wing_pct, daily_entry=daily_entry, dte_calendar=dte_calendar)
            if result:
                all_dte_results[dte_calendar][symbol] = result

    # Print comparison table
    print("\n" + "="*90)
    print("                    DTE COMPARISON RESULTS")
    print("="*90)

    # Header
    header = f"{'Symbol':<8}"
    for dte_calendar in DTE_CALENDAR_VALUES:
        header += f" | {dte_calendar}cd P&L | {dte_calendar}cd Win% | {dte_calendar}cd MaxDD"
    print(header)
    print("-"*90)

    # Find best DTE for each symbol
    best_dte_per_symbol = {}

    for symbol in SYMBOLS.keys():
        row = f"{symbol:<8}"
        best_pnl = float('-inf')
        best_dte_calendar = None

        for dte_calendar in DTE_CALENDAR_VALUES:
            if symbol in all_dte_results[dte_calendar]:
                r = all_dte_results[dte_calendar][symbol]
                row += f" | ${r['total_pnl']:>7.0f} | {r['win_rate']:>7.1%} | ${r['max_drawdown']:>7.0f}"
                if r['total_pnl'] > best_pnl:
                    best_pnl = r['total_pnl']
                    best_dte_calendar = dte_calendar
            else:
                row += f" | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"

        best_dte_per_symbol[symbol] = best_dte_calendar
        print(row)

    print("-"*90)

    # Summary
    print("\nBEST: OPTIMAL DTE PER SYMBOL (calendar days):")
    for symbol, dte_calendar in best_dte_per_symbol.items():
        if dte_calendar and symbol in all_dte_results[dte_calendar]:
            r = all_dte_results[dte_calendar][symbol]
            print(f"   {symbol}: {dte_calendar} cal-day DTE -> P&L ${r['total_pnl']:.0f} | Win Rate {r['win_rate']:.1%} | Max DD ${r['max_drawdown']:.0f}")

    # Find overall best combination
    best_overall = None
    best_overall_pnl = float('-inf')

    for dte_calendar in DTE_CALENDAR_VALUES:
        for symbol, result in all_dte_results[dte_calendar].items():
            if result['total_pnl'] > best_overall_pnl:
                best_overall_pnl = result['total_pnl']
                best_overall = (symbol, dte_calendar, result)

    if best_overall:
        symbol, dte_calendar, r = best_overall
        print(f"\n BEST OVERALL: {symbol} @ {dte_calendar} cal-day DTE")
        print(f"   Total P&L: ${r['total_pnl']:.2f}")
        print(f"   Win Rate: {r['win_rate']:.1%}")
        print(f"   Max Drawdown: ${r['max_drawdown']:.2f}")
        print(f"   Max Losing Streak: {r['max_losing_streak']} | Max Winning Streak: {r['max_winning_streak']}")

    return all_dte_results


def run_parameter_sweep(daily_entry=False):
    """Sweep wing_pct x DTE for all symbols. Find optimal parameters."""
    mode = "DAILY ENTRY" if daily_entry else "WEEKLY (Friday Only)"
    print("\n" + "="*90)
    print(f"              PARAMETER SWEEP - Wing% x DTE Grid Search - {mode}")
    print("="*90)
    print(f"Filter: {format_vix_range()}")
    print(f"Stop Loss Mode: {STOP_LOSS_MODE} | Multiplier: {STOP_LOSS_MULTIPLIER}x")
    print(f"Wing PCT values: {[f'{w*100:.1f}%' for w in WING_PCT_VALUES]}")
    print(f"DTE values (calendar days): {DTE_CALENDAR_VALUES}")
    print(f"Combinations per symbol: {len(WING_PCT_VALUES) * len(DTE_CALENDAR_VALUES)}")
    print("-"*90)
    print("Fetching historical data...")

    best_per_symbol = {}
    all_sweep_results = []

    for symbol in SYMBOLS.keys():
        symbol_results = []

        # Pre-fetch data once per symbol to avoid redundant downloads
        symbol_df = fetch_symbol_data(symbol)
        if symbol_df is None:
            print(f"\n{symbol}: Failed to fetch data.")
            continue

        for wing_pct in WING_PCT_VALUES:
            for dte_calendar in DTE_CALENDAR_VALUES:
                result = run_backtest_single(symbol, wing_pct, daily_entry=daily_entry, dte_calendar=dte_calendar, df=symbol_df)
                if result:
                    symbol_results.append(result)

        if not symbol_results:
            print(f"\n{symbol}: No results found for any parameter combination.")
            continue

        all_sweep_results.extend(symbol_results)

        # Score: balance profit and losing streak using exponential decay
        # Each additional loss in the streak reduces the score by ~15%
        # streak=3 -> 0.61x, streak=6 -> 0.38x, streak=11 -> 0.17x, streak=20 -> 0.04x
        STREAK_DECAY = 0.85
        for r in symbol_results:
            streak_penalty = STREAK_DECAY ** r['max_losing_streak']
            r['score'] = r['total_pnl'] * streak_penalty if r['total_pnl'] > 0 else r['total_pnl']
        symbol_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n{'='*90}")
        print(f"  {symbol} | {format_vix_range()} | Mode: {STOP_LOSS_MODE}")
        print(f"{'='*90}")
        print(f"{'Wing%':>6} {'DTE':>5} {'Trades':>7} {'Wins':>6} {'WinRate':>8} {'TotalP&L':>10} {'MaxDD':>8} {'AvgCred':>9} {'LoseStrk':>9} {'WinStrk':>8} {'Score':>9}")
        print("-"*100)

        for r in symbol_results:
            print(f"{r['wing_pct']*100:>5.1f}% {r['dte_calendar']:>5} {r['trades']:>7} {r['wins']:>6} "
                  f"{r['win_rate']:>7.1%} ${r['total_pnl']:>9.0f} ${r['max_drawdown']:>7.0f} "
                  f"${r['avg_credit']:>8.0f} {r['max_losing_streak']:>9} {r['max_winning_streak']:>8} {r['score']:>9.0f}")

        best = symbol_results[0]
        best_per_symbol[symbol] = best
        print(f"\n  Best: {best['wing_pct']*100:.1f}% wings, {best['dte_calendar']} cal-day DTE "
              f"-> P&L ${best['total_pnl']:.0f} | Win Rate {best['win_rate']:.1%} | Score {best['score']:.0f}")

    # Cross-symbol summary
    if best_per_symbol:
        print(f"\n{'='*90}")
        print("                    OPTIMAL PARAMETERS PER SYMBOL")
        print(f"{'='*90}")
        print(f"{'Symbol':<8} {'Wing%':>6} {'DTE(cd)':>7} {'WinRate':>8} {'TotalP&L':>10} {'MaxDD':>8} {'LoseStrk':>9} {'Score':>9}")
        print("-"*90)

        for symbol, r in best_per_symbol.items():
            print(f"{symbol:<8} {r['wing_pct']*100:>5.1f}% {r['dte_calendar']:>7} "
                  f"{r['win_rate']:>7.1%} ${r['total_pnl']:>9.0f} ${r['max_drawdown']:>7.0f} "
                  f"{r['max_losing_streak']:>9} {r['score']:>9.0f}")

        # Overall best
        overall_best_symbol = max(best_per_symbol, key=lambda s: best_per_symbol[s]['score'])
        ob = best_per_symbol[overall_best_symbol]
        print(f"\n  Overall Best: {overall_best_symbol} @ {ob['wing_pct']*100:.1f}% wings, {ob['dte_calendar']} cal-day DTE")
        print(f"  P&L: ${ob['total_pnl']:.0f} | Win Rate: {ob['win_rate']:.1%} | Max DD: ${ob['max_drawdown']:.0f}")

    return best_per_symbol, all_sweep_results


def export_to_file(df, name, use_excel=False):
    """Export DataFrame to Excel (.xlsx) or CSV. Falls back to CSV if openpyxl is missing."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if use_excel:
        try:
            import openpyxl  # noqa: F401
            path = f"{name}_{timestamp}.xlsx"
            df.to_excel(path, index=False)
            print(f"  Exported: {path}")
            return
        except ImportError:
            print("  openpyxl not available, falling back to CSV")
    path = f"{name}_{timestamp}.csv"
    df.to_csv(path, index=False)
    print(f"  Exported: {path}")


# Run it
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Iron Condor Backtesting & Live Quote Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python Main.py                          # weekly backtest (default)\n"
            "  python Main.py --backtest --daily       # daily-entry backtest\n"
            "  python Main.py --backtest --daily --export\n"
            "  python Main.py --dte --daily            # DTE comparison, daily entry\n"
            "  python Main.py --sweep --export         # parameter sweep + export\n"
            "  python Main.py --sweep --daily --export\n"
            "  python Main.py --compare                # weekly vs daily comparison\n"
            "  python Main.py --live                   # live market quotes\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--backtest", action="store_true",
                      help="Run backtest for all configured symbols (default)")
    mode.add_argument("--dte",      action="store_true",
                      help="Compare multiple DTE holding periods side-by-side")
    mode.add_argument("--sweep",    action="store_true",
                      help="Grid-search wing%% x DTE parameter combinations")
    mode.add_argument("--compare",  action="store_true",
                      help="Run weekly and daily backtests and print comparison table")
    mode.add_argument("--live",     action="store_true",
                      help="Fetch current market option quotes (no backtest)")

    parser.add_argument("--daily",  action="store_true",
                        help="Use daily entry (overlapping positions). Composable with "
                             "--backtest, --dte, and --sweep.")
    parser.add_argument("--export", action="store_true",
                        help="Save results to Excel (.xlsx) or CSV if openpyxl is unavailable")

    args = parser.parse_args()

    if args.live:
        results = fetch_live_quotes()
        if args.export and results:
            export_to_file(pd.DataFrame(results), "live_quotes", use_excel=True)

    elif args.dte:
        all_dte_results = run_dte_comparison(daily_entry=args.daily)
        if args.export and all_dte_results:
            rows = [r for dte_dict in all_dte_results.values() for r in dte_dict.values()]
            if rows:
                export_to_file(pd.DataFrame(rows), "dte_comparison", use_excel=True)

    elif args.sweep:
        best_per_symbol, all_sweep_results = run_parameter_sweep(daily_entry=args.daily)
        if args.export and all_sweep_results:
            export_to_file(pd.DataFrame(all_sweep_results), "sweep_results", use_excel=True)

    elif args.compare:
        print("\n" + "=" * 70)
        print("         COMPARING WEEKLY vs DAILY ENTRY STRATEGIES")
        print("=" * 70 + "\n")

        print("\nWEEKLY ENTRY (Friday only, 1 position at a time):")
        weekly_results = run_all_backtests(daily_entry=False, dte_calendar=7)

        print("\n\nDAILY ENTRY (Every day, up to 5 overlapping positions):")
        daily_results = run_all_backtests(daily_entry=True, dte_calendar=7)

        if weekly_results and daily_results:
            print("\n" + "=" * 90)
            print("                    WEEKLY vs DAILY COMPARISON")
            print("=" * 90)
            print(f"{'Symbol':<8} {'Weekly P&L':>12} {'Daily P&L':>12} {'Mult':>6} "
                  f"{'Wkly MaxDD':>12} {'Daily MaxDD':>12}")
            print("-" * 90)

            compare_rows = []
            for w in weekly_results:
                d = next((x for x in daily_results if x['symbol'] == w['symbol']), None)
                if d:
                    mult = d['total_pnl'] / w['total_pnl'] if w['total_pnl'] != 0 else 0
                    print(f"{w['symbol']:<8} ${w['total_pnl']:>10.0f} ${d['total_pnl']:>10.0f} "
                          f"{mult:>5.1f}x ${w['max_drawdown']:>10.0f} ${d['max_drawdown']:>10.0f}")
                    compare_rows.append({
                        'symbol': w['symbol'],
                        'weekly_pnl': w['total_pnl'],
                        'daily_pnl': d['total_pnl'],
                        'multiplier': mult,
                        'weekly_max_drawdown': w['max_drawdown'],
                        'daily_max_drawdown': d['max_drawdown'],
                    })
            print("-" * 90)

            if args.export and compare_rows:
                export_to_file(pd.DataFrame(compare_rows), "compare_weekly_daily", use_excel=True)

    else:
        # Default: backtest (handles both explicit --backtest and bare invocation)
        results = run_all_backtests(daily_entry=args.daily, dte_calendar=7)
        if args.export and results:
            export_to_file(pd.DataFrame(results), "backtest_results", use_excel=True)

        if not args.backtest:
            print("\nOptions:")
            print("  python Main.py [--backtest | --dte | --sweep | --compare | --live] [--daily] [--export]")
            print()
            print("  --backtest  Run backtest for all symbols (default)")
            print("  --dte       Compare DTE holding periods side-by-side")
            print("  --sweep     Grid-search wing% x DTE parameter combinations")
            print("  --compare   Compare weekly vs daily entry strategies")
            print("  --live      Fetch live market quotes (no backtest)")
            print("  --daily     Daily entry - composable with backtest / dte / sweep")
            print("  --export    Save results to Excel (.xlsx) or CSV")