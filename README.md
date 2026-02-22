# Island Model — MOEX Random Forest Ensemble

[Русский](#русский) | [English](#english)

---

<a id="русский"></a>

## 🇷🇺 Русский

### Описание

Островная модель (Island Model) ансамбля случайных лесов с кольцевой миграцией деревьев для анализа и торговли на Московской бирже (MOEX). Walk-forward бэктестинг на акциях и фьючерсах.

### Идея

Классический Random Forest обучает все деревья в одном пространстве гиперпараметров. Островная модель разбивает ансамбль на **изолированные острова** с разными конфигурациями (глубина, количество признаков, ExtraTrees vs RF), а затем проводит **миграцию лучших деревьев** между островами по кольцевой топологии. Это повышает разнообразие ансамбля и снижает попарную корреляцию предсказаний.

### Структура проекта

```
Moex/
├── island_model/                  # Основная библиотека
│   ├── island_forest.py           # IslandForestModel — островная модель
│   ├── features.py                # FeatureGenerator (30+ технических индикаторов)
│   ├── backtester.py              # Walk-forward бэктестер (single-asset)
│   ├── portfolio_backtester.py    # Портфельный бэктестер с комиссиями MOEX
│   ├── data_loader.py             # Загрузка данных через MOEX ISS API (apimoex)
│   ├── csv_loader.py              # Загрузка из локальных CSV
│   ├── qsh_parser.py              # Парсер QSH (QScalp) бинарного формата
│   └── synthetic_data.py          # Синтетические данные (GARCH, кризисы, корреляции)
│
├── run_backtest.py                # Бэктест на дневных данных (SBER, GAZP, LKOH, GMKN, ROSN)
├── run_real_data_backtest.py      # Бэктест на реальных данных MOEX (SBER+GAZP, 2015-2024)
├── run_portfolio_backtest.py      # Портфельный бэктест (5 тикеров, 100k руб.)
├── run_rih6_backtest.py           # Интрадей бэктест на фьючерсе RTS (5-мин, long+short)
├── run_rih6_v2.py                 # ML + rule-based стратегии на RIH6
│
├── data/                          # Дневные данные (CSV)
│   ├── SBER_daily.csv
│   ├── GAZP_daily.csv
│   ├── LKOH_daily.csv
│   ├── GMKN_daily.csv
│   ├── ROSN_daily.csv
│   ├── USDRUB_daily.csv
│   ├── BRENT_daily.csv
│   ├── RIH6_5min.csv
│   └── RIH6_1h.csv
│
├── downloaded_data/csv_files/     # Скачанные данные для run_real_data_backtest
│   ├── SBER_D1.csv
│   ├── GAZP_D1.csv
│   └── ALLFUTRTSI_D1.csv
│
└── requirements.txt
```

### Компоненты

#### IslandForestModel

Ансамбль из `n_islands` островов (по умолчанию 6), каждый — RandomForest или ExtraTrees с уникальными гиперпараметрами. После обучения проводится `n_migrations` раундов кольцевой миграции: лучшие деревья острова `i` копируются в остров `(i+1) % K`. Финальные веса островов пропорциональны их точности на валидации.

#### Признаки (FeatureGenerator)

30+ технических индикаторов:

- Returns и волатильность (скользящие окна 5, 10, 20, 60 дней)
- RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ADX
- ATR, OBV, Ichimoku Cloud
- Сезонность (день недели, месяц, конец месяца)
- Макро-факторы: USD/RUB, Brent, корреляция нефть-рубль

#### Walk-Forward бэктестер

- Скользящее окно: train → embargo → test
- Метрики: accuracy, F1, Sharpe, Sortino, Max Drawdown, Profit Factor, Calmar
- Безрисковая ставка: 21% годовых (ключевая ставка ЦБ)

#### Портфельный бэктестер

- Реальные торговые издержки MOEX: комиссия брокера (0.03%) + биржевой сбор (0.01%)
- Проскальзывание с учётом market impact
- Лотность (SBER/GAZP = 10 акций, LKOH/GMKN = 1 акция)
- Задержка исполнения 1 день
- Переобучение модели каждые 21 день

#### QSH парсер

Парсер бинарного формата QScalp (v4, gzip) для тиковых данных с MOEX FORTS. Поддерживает Deals stream: delta-кодирование, Growing/LEB128, агрегация в OHLCV свечи.

### Запуск

```bash
pip install -r requirements.txt

# Бэктест на дневных данных (API или синтетика как fallback)
python run_backtest.py

# Портфельный бэктест (5 тикеров, CSV/API/синтетика)
python run_portfolio_backtest.py

# Бэктест на реальных данных SBER+GAZP 2015-2024
python run_real_data_backtest.py

# Интрадей бэктест фьючерса RTS (5-мин свечи)
python run_rih6_backtest.py

# ML + rule-based стратегии на RIH6
python run_rih6_v2.py
```

### Зависимости

- Python 3.10+
- scikit-learn, numpy, pandas
- apimoex (MOEX ISS API)
- lightgbm (опционально)
- matplotlib (графики equity curve)
- joblib (параллельное обучение островов)

### Данные

Проект поддерживает три источника:

1. **Локальные CSV** (`data/`) — приоритет
2. **MOEX ISS API** (apimoex) — загрузка дневных свечей, USD/RUB, Brent
3. **Синтетические данные** — fallback с калибровкой под реальный MOEX (GARCH-волатильность, кризисные периоды COVID/2022, межтикерные корреляции)

---

<a id="english"></a>

## 🇬🇧 English

### Overview

Island Model ensemble of random forests with ring-topology tree migration for Moscow Exchange (MOEX) market data analysis and trading. Walk-forward backtesting on stocks and futures.

### Concept

A standard Random Forest trains all trees within a single hyperparameter space. The Island Model partitions the ensemble into **isolated islands**, each with distinct configurations (depth, feature count, ExtraTrees vs RF), and then performs **migration of the best trees** between islands via a ring topology. This increases ensemble diversity and reduces pairwise prediction correlation.

### Project Structure

```
Moex/
├── island_model/                  # Core library
│   ├── island_forest.py           # IslandForestModel — island ensemble
│   ├── features.py                # FeatureGenerator (30+ technical indicators)
│   ├── backtester.py              # Walk-forward backtester (single-asset)
│   ├── portfolio_backtester.py    # Portfolio backtester with MOEX fees
│   ├── data_loader.py             # Data loading via MOEX ISS API (apimoex)
│   ├── csv_loader.py              # Local CSV loader
│   ├── qsh_parser.py              # QSH (QScalp) binary format parser
│   └── synthetic_data.py          # Synthetic data (GARCH, crises, correlations)
│
├── run_backtest.py                # Daily data backtest (SBER, GAZP, LKOH, GMKN, ROSN)
├── run_real_data_backtest.py      # Real MOEX data backtest (SBER+GAZP, 2015-2024)
├── run_portfolio_backtest.py      # Portfolio backtest (5 tickers, 100k RUB)
├── run_rih6_backtest.py           # Intraday backtest on RTS futures (5-min, long+short)
├── run_rih6_v2.py                 # ML + rule-based strategies on RIH6
│
├── data/                          # Daily data (CSV)
│   ├── SBER_daily.csv
│   ├── GAZP_daily.csv
│   ├── LKOH_daily.csv
│   ├── GMKN_daily.csv
│   ├── ROSN_daily.csv
│   ├── USDRUB_daily.csv
│   ├── BRENT_daily.csv
│   ├── RIH6_5min.csv
│   └── RIH6_1h.csv
│
├── downloaded_data/csv_files/     # Downloaded data for run_real_data_backtest
│   ├── SBER_D1.csv
│   ├── GAZP_D1.csv
│   └── ALLFUTRTSI_D1.csv
│
└── requirements.txt
```

### Components

#### IslandForestModel

An ensemble of `n_islands` islands (default: 6), each being a RandomForest or ExtraTrees with unique hyperparameters. After training, `n_migrations` rounds of ring migration are performed: the best trees from island `i` are copied to island `(i+1) % K`. Final island weights are proportional to their validation accuracy.

#### Features (FeatureGenerator)

30+ technical indicators:

- Returns and volatility (rolling windows of 5, 10, 20, 60 days)
- RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ADX
- ATR, OBV, Ichimoku Cloud
- Seasonality (day of week, month, end of month)
- Macro factors: USD/RUB, Brent, oil-ruble correlation

#### Walk-Forward Backtester

- Rolling window: train → embargo → test
- Metrics: accuracy, F1, Sharpe, Sortino, Max Drawdown, Profit Factor, Calmar
- Risk-free rate: 21% per annum (CBR key rate)

#### Portfolio Backtester

- Real MOEX trading costs: broker commission (0.03%) + exchange fee (0.01%)
- Slippage with market impact modeling
- Lot sizes (SBER/GAZP = 10 shares, LKOH/GMKN = 1 share)
- 1-day execution delay
- Model retraining every 21 days

#### QSH Parser

Parser for QScalp binary format (v4, gzip) for tick data from MOEX FORTS. Supports Deals stream: delta encoding, Growing/LEB128, aggregation into OHLCV candles.

### Getting Started

```bash
pip install -r requirements.txt

# Daily data backtest (API or synthetic fallback)
python run_backtest.py

# Portfolio backtest (5 tickers, CSV/API/synthetic)
python run_portfolio_backtest.py

# Real data backtest SBER+GAZP 2015-2024
python run_real_data_backtest.py

# Intraday RTS futures backtest (5-min candles)
python run_rih6_backtest.py

# ML + rule-based strategies on RIH6
python run_rih6_v2.py
```

### Dependencies

- Python 3.10+
- scikit-learn, numpy, pandas
- apimoex (MOEX ISS API)
- lightgbm (optional)
- matplotlib (equity curve plots)
- joblib (parallel island training)

### Data Sources

The project supports three data sources:

1. **Local CSV files** (`data/`) — highest priority
2. **MOEX ISS API** (apimoex) — daily candles, USD/RUB, Brent
3. **Synthetic data** — fallback calibrated to real MOEX characteristics (GARCH volatility, crisis periods for COVID/2022, inter-ticker correlations)

### License

This project is for educational and research purposes.
