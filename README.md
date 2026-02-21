# Island Model — ансамбль случайных лесов для MOEX

Островная модель (Island Model) ансамбля случайных лесов с кольцевой миграцией деревьев.
Walk-forward бэктестинг на акциях и фьючерсах Московской биржи.

## Идея

Классический Random Forest обучает все деревья в одном пространстве гиперпараметров.
Островная модель разбивает ансамбль на **изолированные острова** с разными конфигурациями
(глубина, количество признаков, ExtraTrees vs RF), а затем проводит **миграцию лучших деревьев**
между островами по кольцевой топологии. Это повышает разнообразие ансамбля и снижает
попарную корреляцию предсказаний.

## Структура проекта

```
Moex/
├── island_model/              # Основная библиотека
│   ├── island_forest.py       # IslandForestModel — островная модель
│   ├── features.py            # FeatureGenerator (30+ технических индикаторов)
│   ├── backtester.py          # Walk-forward бэктестер (single-asset)
│   ├── portfolio_backtester.py# Портфельный бэктестер с комиссиями MOEX
│   ├── data_loader.py         # Загрузка данных через MOEX ISS API (apimoex)
│   ├── csv_loader.py          # Загрузка из локальных CSV
│   ├── qsh_parser.py          # Парсер QSH (QScalp) бинарного формата
│   └── synthetic_data.py      # Синтетические данные (GARCH, кризисы, корреляции)
│
├── run_backtest.py            # Бэктест на дневных данных (SBER, GAZP, LKOH, GMKN, ROSN)
├── run_real_data_backtest.py  # Бэктест на реальных данных MOEX (SBER+GAZP, 2015-2024)
├── run_portfolio_backtest.py  # Портфельный бэктест (5 тикеров, 100k руб.)
├── run_rih6_backtest.py       # Интрадей бэктест на фьючерсе RTS (5-мин, long+short)
├── run_rih6_v2.py             # ML + rule-based стратегии на RIH6
│
├── data/                      # Дневные данные (CSV)
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
├── downloaded_data/csv_files/ # Скачанные данные для run_real_data_backtest
│   ├── SBER_D1.csv
│   ├── GAZP_D1.csv
│   └── ALLFUTRTSI_D1.csv
│
└── requirements.txt
```

## Компоненты

### IslandForestModel

Ансамбль из `n_islands` островов (по умолчанию 6), каждый — RandomForest или ExtraTrees
с уникальными гиперпараметрами. После обучения проводится `n_migrations` раундов
кольцевой миграции: лучшие деревья острова `i` копируются в остров `(i+1) % K`.
Финальные веса островов пропорциональны их точности на валидации.

### Признаки (FeatureGenerator)

30+ технических индикаторов:
- Returns и волатильность (скользящие окна 5, 10, 20, 60 дней)
- RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ADX
- ATR, OBV, Ichimoku Cloud
- Сезонность (день недели, месяц, конец месяца)
- Макро-факторы: USD/RUB, Brent, корреляция нефть-рубль

### Walk-Forward бэктестер

- Скользящее окно: train → embargo → test
- Метрики: accuracy, F1, Sharpe, Sortino, Max Drawdown, Profit Factor, Calmar
- Безрисковая ставка: 21% годовых (ключевая ставка ЦБ)

### Портфельный бэктестер

- Реальные торговые издержки MOEX: комиссия брокера (0.03%) + биржевой сбор (0.01%)
- Проскальзывание с учётом market impact
- Лотность (SBER/GAZP = 10 акций, LKOH/GMKN = 1 акция)
- Задержка исполнения 1 день
- Переобучение модели каждые 21 день

### QSH парсер

Парсер бинарного формата QScalp (v4, gzip) для тиковых данных с MOEX FORTS.
Поддерживает Deals stream: delta-кодирование, Growing/LEB128, агрегация в OHLCV свечи.

## Запуск

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

## Зависимости

- Python 3.10+
- scikit-learn, numpy, pandas
- apimoex (MOEX ISS API)
- lightgbm (опционально)
- matplotlib (графики equity curve)
- joblib (параллельное обучение островов)

## Данные

Проект поддерживает три источника:
1. **Локальные CSV** (`data/`) — приоритет
2. **MOEX ISS API** (apimoex) — загрузка дневных свечей, USD/RUB, Brent
3. **Синтетические данные** — fallback с калибровкой под реальный MOEX
   (GARCH-волатильность, кризисные периоды COVID/2022, межтикерные корреляции)
