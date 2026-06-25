# walk-forward бэктестер

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


@dataclass
class FoldResult:
    """Результат одного фолда."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    log_loss_val: float
    n_train: int
    n_test: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    returns: np.ndarray  # фактические доходности на тестовом периоде


@dataclass
class BacktestResult:
    """Агрегированные результаты бэктеста."""
    model_name: str
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_accuracy(self):
        return np.mean([f.accuracy for f in self.folds])

    @property
    def mean_f1(self):
        return np.mean([f.f1 for f in self.folds])

    @property
    def mean_precision(self):
        return np.mean([f.precision for f in self.folds])

    @property
    def mean_recall(self):
        return np.mean([f.recall for f in self.folds])

    @property
    def mean_log_loss(self):
        return np.mean([f.log_loss_val for f in self.folds])

    def strategy_returns(self):
        """Доходности стратегии по всем фолдам подряд."""
        all_returns = []
        for fold in self.folds:
            # Стратегия: long если модель предсказывает рост, иначе cash (0)
            signals = fold.y_pred  # 1 = long, 0 = cash
            strat_ret = signals * fold.returns
            all_returns.extend(strat_ret)
        return np.array(all_returns)

    def sharpe_ratio(self, risk_free_daily=0.0):
        """Annualized Sharpe."""
        ret = self.strategy_returns()
        excess = ret - risk_free_daily
        if len(excess) == 0 or np.std(excess) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    def sortino_ratio(self, risk_free_daily=0.0):
        """Annualized Sortino."""
        ret = self.strategy_returns()
        excess = ret - risk_free_daily
        downside = excess[excess < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return np.mean(excess) / np.std(downside) * np.sqrt(252)

    def max_drawdown(self):
        ret = self.strategy_returns()
        if len(ret) == 0:
            return 0.0
        equity = np.cumprod(1 + ret)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        return float(np.min(drawdowns))

    def profit_factor(self):
        """Profit Factor = сумма прибылей / сумма убытков."""
        ret = self.strategy_returns()
        gains = ret[ret > 0].sum()
        losses = abs(ret[ret < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    def calmar_ratio(self):
        ret = self.strategy_returns()
        if len(ret) == 0:
            return 0.0
        ann_ret = np.mean(ret) * 252
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return ann_ret / mdd

    def total_return(self):
        ret = self.strategy_returns()
        if len(ret) == 0:
            return 0.0
        return float(np.prod(1 + ret) - 1)

    def summary(self) -> dict:
        # Дневная безрисковая ставка: ~21% годовых / 252
        rf_daily = 0.21 / 252
        return {
            "model": self.model_name,
            "n_folds": len(self.folds),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_log_loss": round(self.mean_log_loss, 4),
            "total_return": round(self.total_return(), 4),
            "sharpe_ratio": round(self.sharpe_ratio(rf_daily), 4),
            "sortino_ratio": round(self.sortino_ratio(rf_daily), 4),
            "max_drawdown": round(self.max_drawdown(), 4),
            "profit_factor": round(self.profit_factor(), 4),
            "calmar_ratio": round(self.calmar_ratio(), 4),
        }


class WalkForwardBacktester:
    """Walk-forward бэктестер со скользящим окном."""

    def __init__(self, train_days=252, test_days=63, embargo_days=5, step_days=None):
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_days = embargo_days
        self.step_days = step_days or test_days

    def run(self, model, X, y, returns, model_name="Model") -> BacktestResult:
        """Запуск walk-forward валидации."""
        result = BacktestResult(model_name=model_name)

        X_vals = X.values if hasattr(X, "values") else np.asarray(X)
        y_vals = y.values if hasattr(y, "values") else np.asarray(y)
        ret_vals = returns.values if hasattr(returns, "values") else np.asarray(returns)

        n = len(X_vals)
        fold_idx = 0
        start = 0

        while start + self.train_days + self.embargo_days + self.test_days <= n:
            train_end = start + self.train_days
            test_start = train_end + self.embargo_days
            test_end = min(test_start + self.test_days, n)

            X_train = X_vals[start:train_end]
            y_train = y_vals[start:train_end]
            X_test = X_vals[test_start:test_end]
            y_test = y_vals[test_start:test_end]
            ret_test = ret_vals[test_start:test_end]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                start += self.step_days
                continue

            # Clone модели для каждого фолда
            fold_model = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_model.fit(X_train, y_train)

            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)

            # Определяем индексы для дат
            if hasattr(X, "index"):
                idx = X.index
                train_start_date = str(idx[start])[:10]
                train_end_date = str(idx[train_end - 1])[:10]
                test_start_date = str(idx[test_start])[:10]
                test_end_date = str(idx[test_end - 1])[:10]
            else:
                train_start_date = str(start)
                train_end_date = str(train_end)
                test_start_date = str(test_start)
                test_end_date = str(test_end)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start=train_start_date,
                train_end=train_end_date,
                test_start=test_start_date,
                test_end=test_end_date,
                accuracy=accuracy_score(y_test, y_pred),
                f1=f1_score(y_test, y_pred, zero_division=0),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                log_loss_val=log_loss(y_test, y_proba),
                n_train=len(X_train),
                n_test=len(X_test),
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                returns=ret_test,
            )
            result.folds.append(fold_result)
            fold_idx += 1
            start += self.step_days

        return result


def run_buy_and_hold(
    returns: pd.Series | np.ndarray,
    test_indices: list[tuple[int, int]],
) -> "BacktestResult":
    """Бенчмарк Buy & Hold — всегда long."""
    result = BacktestResult(model_name="Buy & Hold")
    ret_vals = returns.values if hasattr(returns, "values") else np.asarray(returns)

    for fold_idx, (test_start, test_end) in enumerate(test_indices):
        ret_test = ret_vals[test_start:test_end]
        n_test = len(ret_test)

        fold_result = FoldResult(
            fold_idx=fold_idx,
            train_start="",
            train_end="",
            test_start=str(test_start),
            test_end=str(test_end),
            accuracy=0.0,
            f1=0.0,
            precision=0.0,
            recall=0.0,
            log_loss_val=0.0,
            n_train=0,
            n_test=n_test,
            y_true=np.ones(n_test, dtype=int),
            y_pred=np.ones(n_test, dtype=int),
            y_proba=np.column_stack([np.zeros(n_test), np.ones(n_test)]),
            returns=ret_test,
        )
        result.folds.append(fold_result)

    return result


def print_comparison(results: list["BacktestResult"]) -> None:
    """Печать сравнительной таблицы."""
    summaries = [r.summary() for r in results]
    df = pd.DataFrame(summaries).set_index("model")
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ — WALK-FORWARD BACKTEST")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    return df
