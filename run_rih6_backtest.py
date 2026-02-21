#!/usr/bin/env python3
# Бэктест островной модели на тиковых данных RIH6 (фьючерс RTS, 5-мин свечи).
# Walk-forward: train 6 дней, test 1 день, embargo 12 баров, ретрейн каждый день.

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from island_model.features import _ema, _sma
from island_model.island_forest import IslandForestModel

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# Конфигурация

INITIAL_CAPITAL = 100_000       # руб.
PRICE_STEP = 10                 # шаг цены RTS = 10 пунктов
POINT_VALUE = 1.0               # 1 пункт RTS ~ 1 рубль (при текущих курсах ~1 USD)
# Точнее: стоимость шага = 1 USD * курс ЦБ / курс FORTS
# При USD/RUB ~ 100 и шаге 10 пунктов: стоимость 1 пункта ~ 100/100 ~ 1 руб
# Но для простоты используем пункт=1 руб (уточняется под конкретный контракт)
STEP_COST_RUB = 14.68           # стоимость шага цены в рублях (актуальная для RIH6)
POINT_COST_RUB = STEP_COST_RUB / PRICE_STEP  # стоимость 1 пункта RTS в рублях

EXCHANGE_FEE_PER_CONTRACT = 3.2  # руб. за контракт (биржевой сбор FORTS)
SLIPPAGE_POINTS = 10             # проскальзывание в пунктах RTS
GO_MARGIN = 25_000               # гарантийное обеспечение за 1 контракт RTS (руб.)

TARGET_HORIZON = 6               # 6 баров = 30 минут (на 5-мин свечах)
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45           # P(up) < 0.45 -> short

# Walk-Forward
TRAIN_BARS = 1050    # ~6 торговых дней по 175 свечей
EMBARGO_BARS = 12    # 1 час
RETRAIN_EVERY = 175  # каждый день


def generate_intraday_features(df):
    """Признаки для 5-мин свечей фьючерса."""
    f = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    windows = [6, 12, 24, 48, 96]

    # Returns & Volatility
    for w in windows:
        f[f"ret_{w}"] = close.pct_change(w)
        f[f"vol_{w}"] = close.pct_change().rolling(w).std()
        f[f"volume_ratio_{w}"] = volume / volume.rolling(w).mean().replace(0, np.nan)
        f[f"range_{w}"] = ((high - low) / close).rolling(w).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    f["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    f["macd_hist"] = macd_line - signal_line

    # Bollinger Bands
    sma20 = _sma(close, 20)
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    f["bb_pctb"] = (close - lower) / (upper - lower)
    f["bb_width"] = (upper - lower) / sma20

    # Stochastic
    for w in [14, 28]:
        lowest = low.rolling(w).min()
        highest = high.rolling(w).max()
        f[f"stoch_{w}"] = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / close

    # Volume Profile
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        buy_vol = df["buy_volume"].fillna(0)
        sell_vol = df["sell_volume"].fillna(0)
        total_vol = buy_vol + sell_vol
        f["buy_sell_imbalance"] = (buy_vol - sell_vol) / total_vol.replace(0, np.nan)
        for w in [6, 12, 24]:
            f[f"imbalance_ma_{w}"] = f["buy_sell_imbalance"].rolling(w).mean()

    # Open Interest
    if "oi" in df.columns:
        oi = df["oi"]
        f["oi_change"] = oi.diff()
        f["oi_change_pct"] = oi.pct_change()
        for w in [6, 12]:
            f[f"oi_change_ma_{w}"] = f["oi_change"].rolling(w).mean()

    # Session features
    if hasattr(df.index, "hour"):
        hour = df.index.hour
    else:
        hour = pd.to_datetime(df.index).hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Флаг основной сессии (10:00-18:45)
    f["main_session"] = ((hour >= 10) & (hour < 19)).astype(int)

    return f.dropna()


def generate_target(df, horizon=6):
    """Таргет: forward return за horizon баров > 0."""
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int).rename(f"target_{horizon}b")


def run_futures_backtest(model, model_name, X, y, ohlcv):
    """Walk-forward бэктест, сигнал на bar[i] -> вход по open bar[i+1]."""
    common_idx = X.index.intersection(y.dropna().index).intersection(ohlcv.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    ohlcv = ohlcv.loc[common_idx]

    n = len(X)
    if n < TRAIN_BARS + EMBARGO_BARS + 20:
        return {"error": f"Insufficient data: {n} bars"}

    # State
    cash = INITIAL_CAPITAL
    position = 0  # -1, 0, +1
    entry_price = 0.0
    equity_curve = []
    trades = []
    current_model = None
    last_train_bar = -999

    start_idx = TRAIN_BARS + EMBARGO_BARS + 1

    for i in range(start_idx, n - 1):
        # Retrain
        if i - last_train_bar >= RETRAIN_EVERY or current_model is None:
            train_end = i - EMBARGO_BARS
            train_start = max(0, train_end - TRAIN_BARS)

            if train_end - train_start < 100:
                continue

            X_tr = X.iloc[train_start:train_end]
            y_tr = y.iloc[train_start:train_end]

            if len(np.unique(y_tr)) < 2:
                continue

            current_model = clone(model)
            current_model.fit(X_tr.values, y_tr.values)
            last_train_bar = i

        if current_model is None:
            continue

        # Signal
        features = X.iloc[[i]].values
        try:
            proba = current_model.predict_proba(features)[0]
            p_up = proba[1] if len(proba) > 1 else 0.5
        except Exception:
            p_up = 0.5

        # Execution at next bar's open
        next_open = ohlcv.iloc[i + 1]["open"]
        current_close = ohlcv.iloc[i]["close"]

        # Target position: +1 (long), 0 (flat), -1 (short)
        if p_up > LONG_THRESHOLD:
            target_pos = 1
        elif p_up < SHORT_THRESHOLD:
            target_pos = -1
        else:
            target_pos = 0

        if target_pos != position:
            # Close existing position first
            if position != 0:
                if position == 1:
                    fill_price = next_open - SLIPPAGE_POINTS
                    pnl_points = fill_price - entry_price
                else:  # position == -1
                    fill_price = next_open + SLIPPAGE_POINTS
                    pnl_points = entry_price - fill_price

                pnl_rub = pnl_points * POINT_COST_RUB
                fee = EXCHANGE_FEE_PER_CONTRACT
                cash += pnl_rub - fee
                trades.append({
                    "bar": i + 1,
                    "time": ohlcv.index[i + 1],
                    "side": "CLOSE_LONG" if position == 1 else "CLOSE_SHORT",
                    "price": fill_price,
                    "pnl_points": pnl_points,
                    "pnl_rub": pnl_rub,
                    "fee": fee,
                })
                position = 0

            # Open new position
            if target_pos != 0:
                if target_pos == 1:
                    fill_price = next_open + SLIPPAGE_POINTS
                else:
                    fill_price = next_open - SLIPPAGE_POINTS

                fee = EXCHANGE_FEE_PER_CONTRACT
                cash -= fee
                position = target_pos
                entry_price = fill_price
                trades.append({
                    "bar": i + 1,
                    "time": ohlcv.index[i + 1],
                    "side": "BUY" if target_pos == 1 else "SHORT",
                    "price": fill_price,
                    "fee": fee,
                })

        # Mark-to-market
        unrealized = 0
        if position == 1:
            unrealized = (current_close - entry_price) * POINT_COST_RUB
        elif position == -1:
            unrealized = (entry_price - current_close) * POINT_COST_RUB

        equity = cash + unrealized
        equity_curve.append({
            "time": ohlcv.index[i],
            "equity": equity,
            "position": position,
            "close": current_close,
            "signal": p_up,
        })

    # Close any open position at the end
    if position != 0 and len(ohlcv) > 0:
        last_price = ohlcv.iloc[-1]["close"]
        if position == 1:
            pnl_points = last_price - entry_price
        else:
            pnl_points = entry_price - last_price
        pnl_rub = pnl_points * POINT_COST_RUB
        cash += pnl_rub - EXCHANGE_FEE_PER_CONTRACT
        trades.append({
            "bar": len(ohlcv) - 1,
            "time": ohlcv.index[-1],
            "side": f"CLOSE_{'LONG' if position == 1 else 'SHORT'} (end)",
            "price": last_price,
            "pnl_points": pnl_points,
            "pnl_rub": pnl_rub,
            "fee": EXCHANGE_FEE_PER_CONTRACT,
        })

    # Metrics
    eq = pd.DataFrame(equity_curve)
    if eq.empty:
        return {"error": "No equity data"}

    final_equity = eq["equity"].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    daily_ret = eq["equity"].pct_change().dropna()
    sharpe = 0.0
    if len(daily_ret) > 10 and daily_ret.std() > 0:
        # Для интрадея: annual factor = sqrt(252 * 175)  [175 баров/день]
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252 * 35)  # ~35 баров в активной сессии

    equities = eq["equity"].values
    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities - running_max) / running_max
    max_dd = drawdowns.min() * 100

    # Trade stats
    close_trades = [t for t in trades if "pnl_rub" in t]
    n_trades = len(close_trades)
    total_fees = sum(t["fee"] for t in trades)

    if n_trades > 0:
        wins = [t for t in close_trades if t["pnl_rub"] > 0]
        losses = [t for t in close_trades if t["pnl_rub"] <= 0]
        win_rate = len(wins) / n_trades * 100
        avg_win = np.mean([t["pnl_rub"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_rub"] for t in losses]) if losses else 0
        total_pnl = sum(t["pnl_rub"] for t in close_trades)
        gross_profit = sum(t["pnl_rub"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl_rub"] for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        n_longs = sum(1 for t in close_trades if "LONG" in t.get("side", ""))
        n_shorts = sum(1 for t in close_trades if "SHORT" in t.get("side", ""))
    else:
        win_rate = avg_win = avg_loss = total_pnl = profit_factor = 0
        n_longs = n_shorts = 0

    return {
        "model_name": model_name,
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 1),
        "avg_win_rub": round(avg_win, 2),
        "avg_loss_rub": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "total_fees_rub": round(total_fees, 2),
        "total_pnl_rub": round(total_pnl, 2) if n_trades > 0 else 0,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
        "equity_curve": eq,
        "trades": trades,
    }


def run_buy_and_hold_futures(ohlcv, start_idx):
    """Buy & Hold бенчмарк: 1 контракт."""
    if start_idx >= len(ohlcv) - 1:
        return {"error": "Insufficient data"}

    entry_price = ohlcv.iloc[start_idx]["open"] + SLIPPAGE_POINTS
    fee = EXCHANGE_FEE_PER_CONTRACT

    equity_curve = []
    cash = INITIAL_CAPITAL - fee

    for i in range(start_idx, len(ohlcv)):
        current_close = ohlcv.iloc[i]["close"]
        unrealized = (current_close - entry_price) * POINT_COST_RUB
        equity = cash + unrealized
        equity_curve.append({
            "time": ohlcv.index[i],
            "equity": equity,
            "position": 1,
            "close": current_close,
            "signal": 1.0,
        })

    # Close at end
    last_price = ohlcv.iloc[-1]["close"]
    pnl_points = last_price - entry_price
    pnl_rub = pnl_points * POINT_COST_RUB
    final_equity = INITIAL_CAPITAL + pnl_rub - 2 * fee  # open + close fees

    eq = pd.DataFrame(equity_curve)
    equities = eq["equity"].values
    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities - running_max) / running_max

    return {
        "model_name": "Buy & Hold",
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / INITIAL_CAPITAL - 1) * 100, 2),
        "sharpe": 0,
        "max_drawdown_pct": round(drawdowns.min() * 100, 2),
        "n_trades": 1,
        "win_rate_pct": 100.0 if pnl_rub > 0 else 0.0,
        "avg_win_rub": round(pnl_rub, 2),
        "avg_loss_rub": 0,
        "profit_factor": float("inf") if pnl_rub > 0 else 0,
        "total_fees_rub": round(2 * fee, 2),
        "total_pnl_rub": round(pnl_rub, 2),
        "equity_curve": eq,
        "trades": [],
    }


def main():
    print("RIH6 backtest (5-min bars, 10 days)")
    print(f"capital={INITIAL_CAPITAL}, GO={GO_MARGIN}, fee={EXCHANGE_FEE_PER_CONTRACT}, "
          f"slip={SLIPPAGE_POINTS}pts, step_cost={STEP_COST_RUB}rub/{PRICE_STEP}pts, "
          f"horizon={TARGET_HORIZON}bars")

    # Load data
    print("\nloading data...")
    ohlcv = pd.read_csv("/home/user/Moex/data/RIH6_5min.csv", index_col=0, parse_dates=True)
    print(f"  bars={len(ohlcv)}, {ohlcv.index[0]} -> {ohlcv.index[-1]}")
    print(f"  price range: {ohlcv['low'].min():,.0f} - {ohlcv['high'].max():,.0f}, "
          f"total vol: {ohlcv['volume'].sum():,.0f}")

    # Features
    print("\ngenerating features...")
    X = generate_intraday_features(ohlcv)
    y = generate_target(ohlcv, TARGET_HORIZON)
    common_idx = X.index.intersection(y.dropna().index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    print(f"  {X.shape[1]} features, {len(X)} obs, balance={y.mean():.1%} positive")
    print(f"  train_window={TRAIN_BARS} (~{TRAIN_BARS/175:.0f}d), embargo={EMBARGO_BARS}")

    # Models
    print("\nsetting up models...")

    models = {}

    models["IslandForest"] = IslandForestModel(
        n_islands=6,
        trees_per_island=50,
        n_migrations=3,
        migration_rate=0.1,
        val_fraction=0.2,
        n_jobs=-1,
        random_state=42,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )

    print(f"  models: {', '.join(models.keys())} + Buy&Hold")

    # Backtest
    print("\nrunning backtests...")

    results = {}
    for name, model in models.items():
        print(f"  {name}...", end=" ", flush=True)
        res = run_futures_backtest(model, name, X, y, ohlcv)
        results[name] = res
        if "error" not in res:
            print(f"equity={res['final_equity']:,.0f} ({res['total_return_pct']:+.2f}%), "
                  f"trades={res['n_trades']}, WR={res['win_rate_pct']:.1f}%")
        else:
            print(f"ERROR: {res['error']}")

    # Buy & Hold
    print(f"  Buy & Hold...", end=" ", flush=True)
    start_idx = TRAIN_BARS + EMBARGO_BARS + 1
    bnh = run_buy_and_hold_futures(ohlcv, start_idx)
    results["Buy & Hold"] = bnh
    if "error" not in bnh:
        print(f"equity={bnh['final_equity']:,.0f} ({bnh['total_return_pct']:+.2f}%)")

    # Results summary
    print("\n--- results ---")

    rows = []
    for name, res in results.items():
        if "error" in res:
            continue
        rows.append({
            "Модель": name,
            "Капитал": f"{res['final_equity']:,.0f}",
            "Доход%": f"{res['total_return_pct']:+.2f}%",
            "MaxDD%": f"{res['max_drawdown_pct']:.2f}%",
            "Сделок": res["n_trades"],
            "L/S": f"{res.get('n_longs',0)}/{res.get('n_shorts',0)}",
            "WR%": f"{res['win_rate_pct']:.1f}%",
            "PF": f"{res['profit_factor']:.2f}",
            "Комиссии": f"{res['total_fees_rub']:.0f}",
            "P&L": f"{res['total_pnl_rub']:+,.0f}",
        })

    df = pd.DataFrame(rows).set_index("Модель")
    print(f"\n{df.to_string()}")

    # Trade log for IslandForest
    if "IslandForest" in results and "error" not in results["IslandForest"]:
        trades = results["IslandForest"]["trades"]
        sell_trades = [t for t in trades if "pnl_rub" in t]
        if sell_trades:
            print(f"\nIslandForest trades (first 20):")
            for t in sell_trades[:20]:
                print(f"  {t['time']} {t['side']:10s} @ {t['price']:>10,.0f}  "
                      f"P&L: {t['pnl_points']:>+8.0f} пт = {t['pnl_rub']:>+8.0f} руб. "
                      f"(fee {t['fee']:.1f})")

    # Equity curve plot
    print("\nplotting...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    # Equity
    ax1 = axes[0]
    for name, res in results.items():
        if "error" in res or "equity_curve" not in res:
            continue
        eq = res["equity_curve"]
        ax1.plot(eq["time"], eq["equity"], label=name, linewidth=1.2)
    ax1.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("RIH6 Intraday Backtest — Equity Curve (100K RUB)", fontsize=14)
    ax1.set_ylabel("Equity (RUB)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Price
    ax2 = axes[1]
    ax2.plot(ohlcv.index, ohlcv["close"], color="black", linewidth=0.8, alpha=0.7)
    ax2.set_title("RIH6 Price", fontsize=12)
    ax2.set_ylabel("Price (pts)")
    ax2.grid(True, alpha=0.3)

    # Drawdown
    ax3 = axes[2]
    for name, res in results.items():
        if "error" in res or "equity_curve" not in res:
            continue
        eq = res["equity_curve"]
        equities = eq["equity"].values
        running_max = np.maximum.accumulate(equities)
        dd = (equities - running_max) / running_max * 100
        ax3.fill_between(eq["time"], dd, 0, alpha=0.3, label=name)
    ax3.set_title("Drawdown (%)", fontsize=12)
    ax3.set_ylabel("DD %")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/user/Moex/equity_rih6.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved equity_rih6.png")

    print("\ndone.")


if __name__ == "__main__":
    main()
