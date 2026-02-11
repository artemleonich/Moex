#!/usr/bin/env python3
"""
Бэктест на реальных данных RIH6 — сравнение подходов.

Проблема: 10 дней данных мало для ML walk-forward (нужно 30+).
Решение: добавляем rule-based стратегии, которые НЕ требуют обучения,
и оптимизируем ML через уменьшенный train window.

Стратегии:
1. IslandForest (ML) — walk-forward с коротким train
2. LightGBM (ML) — walk-forward
3. MeanReversion — RSI oversold/overbought + Bollinger Bands
4. Momentum — пробой ATR + volume confirmation
5. Combined — ансамбль ML + rules
6. Buy & Hold
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100_000
PRICE_STEP = 10
STEP_COST_RUB = 14.68
POINT_COST_RUB = STEP_COST_RUB / PRICE_STEP
EXCHANGE_FEE = 3.2
SLIPPAGE_PTS = 10
GO_MARGIN = 25_000


# ─────────────────────────────────────────────────────────────────────────────
# Feature Generation
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки для 5-мин свечей."""
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    for w in [6, 12, 24, 48, 96]:
        f[f"ret_{w}"] = close.pct_change(w)
        f[f"vol_{w}"] = close.pct_change().rolling(w).std()
        f[f"vratio_{w}"] = volume / volume.rolling(w).mean().replace(0, np.nan)
        f[f"range_{w}"] = ((high - low) / close).rolling(w).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    f["rsi"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    f["macd_hist"] = ema12 - ema26 - _ema(ema12 - ema26, 9)

    # Bollinger
    sma20 = _sma(close, 20)
    std20 = close.rolling(20).std()
    f["bb_upper"] = sma20 + 2 * std20
    f["bb_lower"] = sma20 - 2 * std20
    f["bb_pctb"] = (close - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"])
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / sma20

    # Stochastic
    lo14 = low.rolling(14).min()
    hi14 = high.rolling(14).max()
    f["stoch"] = 100 * (close - lo14) / (hi14 - lo14).replace(0, np.nan)

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr"] / close

    # Volume imbalance
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        bv = df["buy_volume"].fillna(0)
        sv = df["sell_volume"].fillna(0)
        f["imbalance"] = (bv - sv) / (bv + sv).replace(0, np.nan)
        f["imb_ma6"] = f["imbalance"].rolling(6).mean()
        f["imb_ma12"] = f["imbalance"].rolling(12).mean()

    # OI
    if "oi" in df.columns:
        f["oi_chg"] = df["oi"].diff()
        f["oi_chg_ma6"] = f["oi_chg"].rolling(6).mean()

    # Session
    hour = df.index.hour if hasattr(df.index, "hour") else pd.to_datetime(df.index).hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["main_session"] = ((hour >= 10) & (hour < 19)).astype(int)

    return f.dropna()


def build_target(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int).rename(f"target_{horizon}b")


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────

def backtest(
    signals: pd.Series,
    ohlcv: pd.DataFrame,
    name: str,
) -> dict:
    """
    Универсальный бэктестер.

    signals: pd.Series с индексом как у ohlcv, значения = target_position (-1, 0, +1)
    """
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    equity_curve = []
    trades = []

    sig_aligned = signals.reindex(ohlcv.index).fillna(0).astype(int)

    for i in range(len(ohlcv) - 1):
        target_pos = int(sig_aligned.iloc[i])
        current_close = ohlcv.iloc[i]["close"]
        next_open = ohlcv.iloc[i + 1]["open"]

        if target_pos != position:
            # Close existing
            if position != 0:
                if position == 1:
                    fill = next_open - SLIPPAGE_PTS
                    pnl_pts = fill - entry_price
                else:
                    fill = next_open + SLIPPAGE_PTS
                    pnl_pts = entry_price - fill
                pnl_rub = pnl_pts * POINT_COST_RUB
                cash += pnl_rub - EXCHANGE_FEE
                trades.append({
                    "time": ohlcv.index[i + 1],
                    "side": f"CLOSE_{'L' if position==1 else 'S'}",
                    "price": fill,
                    "pnl_pts": pnl_pts,
                    "pnl_rub": pnl_rub,
                })
                position = 0

            # Open new
            if target_pos != 0:
                fill = next_open + (SLIPPAGE_PTS if target_pos == 1 else -SLIPPAGE_PTS)
                cash -= EXCHANGE_FEE
                position = target_pos
                entry_price = fill
                trades.append({
                    "time": ohlcv.index[i + 1],
                    "side": "BUY" if target_pos == 1 else "SHORT",
                    "price": fill,
                })

        # Mark-to-market
        unreal = 0
        if position == 1:
            unreal = (current_close - entry_price) * POINT_COST_RUB
        elif position == -1:
            unreal = (entry_price - current_close) * POINT_COST_RUB

        equity_curve.append({
            "time": ohlcv.index[i],
            "equity": cash + unreal,
            "position": position,
        })

    # Close at end
    if position != 0:
        last = ohlcv.iloc[-1]["close"]
        pnl_pts = (last - entry_price) if position == 1 else (entry_price - last)
        pnl_rub = pnl_pts * POINT_COST_RUB
        cash += pnl_rub - EXCHANGE_FEE
        trades.append({
            "time": ohlcv.index[-1],
            "side": f"CLOSE_{'L' if position==1 else 'S'} (end)",
            "price": last,
            "pnl_pts": pnl_pts,
            "pnl_rub": pnl_rub,
        })

    eq = pd.DataFrame(equity_curve)
    if eq.empty:
        return {"error": "No data"}

    close_trades = [t for t in trades if "pnl_rub" in t]
    n_trades = len(close_trades)
    total_fees = sum(EXCHANGE_FEE for t in trades)

    final_eq = eq["equity"].iloc[-1]
    equities = eq["equity"].values
    running_max = np.maximum.accumulate(equities)
    dd = (equities - running_max) / running_max

    wins = [t for t in close_trades if t["pnl_rub"] > 0]
    losses = [t for t in close_trades if t["pnl_rub"] <= 0]
    gross_win = sum(t["pnl_rub"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_rub"] for t in losses)) if losses else 0

    return {
        "name": name,
        "final_equity": round(final_eq, 0),
        "return_pct": round((final_eq / INITIAL_CAPITAL - 1) * 100, 2),
        "max_dd_pct": round(dd.min() * 100, 2),
        "n_trades": n_trades,
        "n_longs": sum(1 for t in close_trades if "L" in t["side"]),
        "n_shorts": sum(1 for t in close_trades if "S" in t["side"]),
        "win_rate": round(len(wins) / n_trades * 100, 1) if n_trades > 0 else 0,
        "pf": round(gross_win / gross_loss, 2) if gross_loss > 0 else 999,
        "fees": round(total_fees, 0),
        "pnl": round(sum(t["pnl_rub"] for t in close_trades), 0) if close_trades else 0,
        "eq_df": eq,
        "trades": close_trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy: Mean Reversion
# ─────────────────────────────────────────────────────────────────────────────

def strategy_mean_reversion(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.Series:
    """
    Mean Reversion: покупаем oversold, шортим overbought.
    RSI < 30 + BB < 0.1 → LONG
    RSI > 70 + BB > 0.9 → SHORT
    Выход при RSI в нейтральной зоне (40-60).
    """
    signals = pd.Series(0, index=features.index)
    position = 0

    for i in range(len(features)):
        rsi = features.iloc[i].get("rsi", 50)
        bb = features.iloc[i].get("bb_pctb", 0.5)

        if position == 0:
            if rsi < 30 and bb < 0.15:
                position = 1
            elif rsi > 70 and bb > 0.85:
                position = -1
        elif position == 1:
            if rsi > 55 or bb > 0.7:
                position = 0
        elif position == -1:
            if rsi < 45 or bb < 0.3:
                position = 0

        signals.iloc[i] = position

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Strategy: Momentum / Trend Following
# ─────────────────────────────────────────────────────────────────────────────

def strategy_momentum(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.Series:
    """
    Momentum: следуем за трендом.
    MACD > 0 + ret_24 > 0 + volume rising → LONG
    MACD < 0 + ret_24 < 0 + volume rising → SHORT
    """
    signals = pd.Series(0, index=features.index)
    position = 0
    bars_in_position = 0

    for i in range(len(features)):
        row = features.iloc[i]
        macd = row.get("macd_hist", 0)
        ret24 = row.get("ret_24", 0)
        vratio = row.get("vratio_12", 1)
        atr_pct = row.get("atr_pct", 0)

        if position == 0:
            if macd > 0 and ret24 > 0 and vratio > 0.8:
                position = 1
                bars_in_position = 0
            elif macd < 0 and ret24 < 0 and vratio > 0.8:
                position = -1
                bars_in_position = 0
        else:
            bars_in_position += 1
            # Exit conditions
            if position == 1 and (macd < 0 or bars_in_position > 24):
                position = 0
            elif position == -1 and (macd > 0 or bars_in_position > 24):
                position = 0

        signals.iloc[i] = position

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Strategy: Volume Imbalance
# ─────────────────────────────────────────────────────────────────────────────

def strategy_volume_imbalance(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.Series:
    """
    Volume Imbalance: следуем за крупными покупателями/продавцами.
    Сильный buy imbalance + OI растёт → LONG
    Сильный sell imbalance + OI растёт → SHORT
    """
    signals = pd.Series(0, index=features.index)
    position = 0
    bars_in = 0

    for i in range(len(features)):
        row = features.iloc[i]
        imb = row.get("imb_ma6", 0)
        oi_chg = row.get("oi_chg_ma6", 0)
        main_session = row.get("main_session", 0)

        if not main_session:
            # Только основная сессия
            if position != 0:
                position = 0
            signals.iloc[i] = 0
            continue

        if position == 0:
            if imb > 0.15 and oi_chg > 0:
                position = 1
                bars_in = 0
            elif imb < -0.15 and oi_chg > 0:
                position = -1
                bars_in = 0
        else:
            bars_in += 1
            # Реверс или тайм-стоп
            if position == 1 and (imb < -0.1 or bars_in > 18):
                position = 0
            elif position == -1 and (imb > 0.1 or bars_in > 18):
                position = 0

        signals.iloc[i] = position

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# ML Strategy with shorter train
# ─────────────────────────────────────────────────────────────────────────────

def strategy_ml(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    ohlcv: pd.DataFrame,
    train_bars: int = 500,
    embargo: int = 12,
    retrain_every: int = 100,
    long_thr: float = 0.55,
    short_thr: float = 0.45,
) -> pd.Series:
    """ML walk-forward с настраиваемыми параметрами."""
    common_idx = X.index.intersection(y.dropna().index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    signals = pd.Series(0, index=X.index, dtype=int)
    current_model = None
    last_train = -999

    start = train_bars + embargo + 1
    if start >= len(X):
        return signals

    for i in range(start, len(X)):
        if i - last_train >= retrain_every or current_model is None:
            t_end = i - embargo
            t_start = max(0, t_end - train_bars)
            if t_end - t_start < 50:
                continue
            Xt = X.iloc[t_start:t_end]
            yt = y.iloc[t_start:t_end]
            if len(np.unique(yt)) < 2:
                continue
            current_model = clone(model)
            current_model.fit(Xt.values, yt.values)
            last_train = i

        if current_model is None:
            continue

        try:
            p = current_model.predict_proba(X.iloc[[i]].values)[0]
            p_up = p[1] if len(p) > 1 else 0.5
        except Exception:
            p_up = 0.5

        if p_up > long_thr:
            signals.iloc[i] = 1
        elif p_up < short_thr:
            signals.iloc[i] = -1

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Combined Ensemble Strategy
# ─────────────────────────────────────────────────────────────────────────────

def strategy_combined(
    ml_signals: pd.Series,
    mr_signals: pd.Series,
    mom_signals: pd.Series,
    vi_signals: pd.Series,
) -> pd.Series:
    """
    Ансамбль: голосование ML + rule-based стратегий.
    Позиция = sign(сумма сигналов), но только если >= 2 из 4 согласны.
    """
    common = ml_signals.index.intersection(mr_signals.index) \
        .intersection(mom_signals.index).intersection(vi_signals.index)

    ml = ml_signals.reindex(common).fillna(0)
    mr = mr_signals.reindex(common).fillna(0)
    mom = mom_signals.reindex(common).fillna(0)
    vi = vi_signals.reindex(common).fillna(0)

    vote = ml + mr + mom + vi
    signals = pd.Series(0, index=common, dtype=int)
    signals[vote >= 2] = 1
    signals[vote <= -2] = -1

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("БЭКТЕСТ RIH6 — ML + RULE-BASED СТРАТЕГИИ")
    print("Фьючерс RTS, 5-мин свечи, 10 дней реальных тиков")
    print(f"Портфель: {INITIAL_CAPITAL:,} руб. | ГО: {GO_MARGIN:,} руб.")
    print(f"Сбор: {EXCHANGE_FEE} руб. | Слипедж: {SLIPPAGE_PTS} пунктов")
    print("=" * 70)

    # Load
    ohlcv = pd.read_csv("/home/user/Moex/data/RIH6_5min.csv", index_col=0, parse_dates=True)
    print(f"\nДанные: {len(ohlcv)} свечей, {ohlcv.index[0].date()} — {ohlcv.index[-1].date()}")
    print(f"Цена: {ohlcv['low'].min():,.0f} — {ohlcv['high'].max():,.0f}")

    # Features
    features = build_features(ohlcv)
    target = build_target(ohlcv, horizon=6)
    print(f"Признаков: {features.shape[1]}, наблюдений: {len(features)}")

    # ── Rule-based signals ──
    print("\n[RULE-BASED СТРАТЕГИИ]")

    sig_mr = strategy_mean_reversion(features, ohlcv)
    sig_mom = strategy_momentum(features, ohlcv)
    sig_vi = strategy_volume_imbalance(features, ohlcv)

    print(f"  MeanReversion:    {(sig_mr != 0).sum()} баров в позиции "
          f"(L:{(sig_mr == 1).sum()}, S:{(sig_mr == -1).sum()})")
    print(f"  Momentum:         {(sig_mom != 0).sum()} баров в позиции "
          f"(L:{(sig_mom == 1).sum()}, S:{(sig_mom == -1).sum()})")
    print(f"  VolumeImbalance:  {(sig_vi != 0).sum()} баров в позиции "
          f"(L:{(sig_vi == 1).sum()}, S:{(sig_vi == -1).sum()})")

    # ── ML signals (shorter train window) ──
    print("\n[ML СТРАТЕГИИ]")

    island_model = IslandForestModel(
        n_islands=6, trees_per_island=50, n_migrations=3,
        migration_rate=0.1, val_fraction=0.2, n_jobs=-1, random_state=42,
    )
    sig_island = strategy_ml(island_model, features, target, ohlcv,
                              train_bars=500, embargo=12, retrain_every=100,
                              long_thr=0.54, short_thr=0.46)
    print(f"  IslandForest:     {(sig_island != 0).sum()} баров в позиции")

    lgbm_model = None
    sig_lgbm = pd.Series(0, index=features.index)
    if HAS_LGBM:
        lgbm_model = LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            num_leaves=15, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=2.0, min_child_samples=20,
            random_state=42, verbose=-1,
        )
        sig_lgbm = strategy_ml(lgbm_model, features, target, ohlcv,
                                train_bars=500, embargo=12, retrain_every=100,
                                long_thr=0.54, short_thr=0.46)
        print(f"  LightGBM:         {(sig_lgbm != 0).sum()} баров в позиции")

    # ── Combined ──
    sig_combined = strategy_combined(sig_island, sig_mr, sig_mom, sig_vi)
    print(f"  Combined (vote):  {(sig_combined != 0).sum()} баров в позиции")

    # ── Backtest all ──
    print("\n[БЭКТЕСТ]")
    print("=" * 70)

    strategies = {
        "MeanReversion": sig_mr,
        "Momentum": sig_mom,
        "VolumeImbalance": sig_vi,
        "IslandForest": sig_island,
        "Combined": sig_combined,
    }
    if HAS_LGBM:
        strategies["LightGBM"] = sig_lgbm

    # Buy & Hold
    sig_bnh = pd.Series(1, index=ohlcv.index)
    strategies["Buy & Hold"] = sig_bnh

    results = {}
    for sname, sig in strategies.items():
        res = backtest(sig, ohlcv, sname)
        results[sname] = res
        if "error" not in res:
            sign = "+" if res["return_pct"] > 0 else ""
            print(f"  {sname:20s}: {res['final_equity']:>8,.0f} руб. "
                  f"({sign}{res['return_pct']:.2f}%) "
                  f"DD={res['max_dd_pct']:.2f}% "
                  f"trades={res['n_trades']} "
                  f"WR={res['win_rate']:.0f}% "
                  f"PF={res['pf']:.2f}")

    # ── Summary table ──
    print(f"\n{'═' * 70}")
    print("ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'═' * 70}")

    rows = []
    for sname, res in results.items():
        if "error" in res:
            continue
        rows.append({
            "Стратегия": sname,
            "Капитал": f"{res['final_equity']:,.0f}",
            "Доход%": f"{res['return_pct']:+.2f}%",
            "MaxDD%": f"{res['max_dd_pct']:.2f}%",
            "Сделок": res["n_trades"],
            "L/S": f"{res['n_longs']}/{res['n_shorts']}",
            "WR%": f"{res['win_rate']:.0f}%",
            "PF": f"{res['pf']:.2f}",
            "P&L": f"{res['pnl']:+,.0f}",
        })
    df = pd.DataFrame(rows).set_index("Стратегия")
    print(f"\n{df.to_string()}")

    # ── Best trades ──
    for sname in ["MeanReversion", "Momentum", "Combined"]:
        if sname in results and "error" not in results[sname]:
            ct = results[sname]["trades"]
            if ct:
                print(f"\n{'─' * 50}")
                print(f"Сделки {sname}:")
                for t in ct[:15]:
                    print(f"  {t['time']} {t['side']:12s} @ {t['price']:>10,.0f}  "
                          f"P&L: {t['pnl_pts']:>+6.0f} пт = {t['pnl_rub']:>+7.0f} руб.")

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 13),
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    ax1 = axes[0]
    for sname, res in results.items():
        if "error" in res:
            continue
        eq = res["eq_df"]
        lw = 2.0 if sname in ["Combined", "MeanReversion"] else 1.0
        ax1.plot(eq["time"], eq["equity"], label=sname, linewidth=lw)
    ax1.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("RIH6 Backtest — Equity Curve (100K RUB)", fontsize=14)
    ax1.set_ylabel("Equity (RUB)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(ohlcv.index, ohlcv["close"], color="black", linewidth=0.8)
    ax2.set_title("RIH6 Price", fontsize=12)
    ax2.set_ylabel("Points")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for sname, res in results.items():
        if "error" in res:
            continue
        eq = res["eq_df"]
        equities = eq["equity"].values
        rm = np.maximum.accumulate(equities)
        dd = (equities - rm) / rm * 100
        ax3.fill_between(eq["time"], dd, 0, alpha=0.25, label=sname)
    ax3.set_title("Drawdown (%)", fontsize=12)
    ax3.set_ylabel("DD %")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/user/Moex/equity_rih6_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: equity_rih6_v2.png")

    print(f"\n{'━' * 70}")
    print("БЭКТЕСТ ЗАВЕРШЁН")
    print(f"{'━' * 70}")


if __name__ == "__main__":
    main()
