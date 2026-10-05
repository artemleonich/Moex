<p align="center">
  <img src=".github/assets/banner.svg" width="100%" alt="Moex · Island Forest" />
</p>

# Moex · Island Forest

Исследовательский проект: островной ансамбль Random Forest / ExtraTrees и walk-forward бэктесты на данных Московской биржи.

**Python · scikit-learn · pandas · MOEX ISS**  
[Быстрый старт](#быстрый-старт) · [Эксперименты](#эксперименты) · [English](#english)

## Что внутри

- Ансамбль из нескольких «островов» с разными гиперпараметрами и кольцевой миграцией деревьев.
- Технические признаки: доходности, волатильность, RSI, MACD, Bollinger Bands, ATR, OBV и сезонность; дополнительные факторы USD/RUB и Brent.
- Walk-forward разделение `train → embargo → test` и сравнение с Random Forest, Buy & Hold и LightGBM при наличии зависимости.
- Портфельная симуляция с лотностью, комиссиями, моделью проскальзывания и задержкой исполнения.
- Отдельные эксперименты на пяти минутных свечах фьючерса RIH6, включая ML и правила.
- Загрузчики MOEX ISS и CSV, генератор синтетических данных, парсер QSH для потока сделок.

Это код для обучения и исследования. Результаты бэктестов зависят от источника данных и допущений модели.

## Быстрый старт

Нужен **Python 3.10+**.

```bash
git clone https://github.com/artemleonich/Moex.git
cd Moex
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python run_backtest.py
```

В Windows виртуальное окружение активируется командой `.venv\Scripts\activate`.

Основной скрипт запрашивает дневные данные через MOEX ISS. Если данные не получены, он переключается на синтетический набор и сообщает об этом в консоли. LightGBM включён в `requirements.txt`; код пропускает этот бенчмарк, если пакет недоступен.

## Эксперименты

| Скрипт | Что исследует | Источник данных |
| --- | --- | --- |
| [run_backtest.py](run_backtest.py) | Дневные модели для SBER, GAZP, LKOH, GMKN, ROSN | MOEX ISS → синтетический fallback |
| [run_portfolio_backtest.py](run_portfolio_backtest.py) | Портфель пяти акций, начальный капитал 100 000 ₽ | `data/` → MOEX ISS → синтетический fallback |
| [run_real_data_backtest.py](run_real_data_backtest.py) | Портфель SBER + GAZP, окно 2015–2024 | `downloaded_data/csv_files/*_D1.csv` |
| [run_rih6_backtest.py](run_rih6_backtest.py) | Walk-forward ML на RIH6, long/short | `data/RIH6_5min.csv` |
| [run_rih6_v2.py](run_rih6_v2.py) | Сравнение ML и стратегий по правилам на RIH6 | `data/RIH6_5min.csv` |

В четырёх дополнительных скриптах пути сохранения графиков заданы как `/home/user/Moex/...`. В двух скриптах RIH6 такой же абсолютный путь используется для входного CSV. Перед запуском замените эти пути на расположение своего клона.

```bash
# После настройки путей:
python run_portfolio_backtest.py
python run_real_data_backtest.py
python run_rih6_backtest.py
python run_rih6_v2.py
```

## Модель и настройки

[IslandForestModel](island_model/island_forest.py) использует по умолчанию **6 островов × 50 деревьев**, 3 раунда миграции и долю миграции 0,1. Лучшие деревья копируются в следующий остров, а финальные веса островов рассчитываются по точности на валидационной выборке.

Основные настройки находятся в начале соответствующего `run_*.py`: тикеры, даты, горизонт целевой переменной, окна обучения, embargo, начальный капитал и параметры издержек.

Для дневного эксперимента заданы 252 дня обучения, 63 дня теста и 5 дней embargo. Портфельные скрипты переобучают модель каждые 21 день и задают задержку исполнения в 1 день. В их конфигурации используются комиссия брокера 0,03% и биржевой сбор 0,01% за сторону.

В портфельных метриках используется фиксированная безрисковая ставка **21% годовых**. Это параметр расчёта, который нужно проверить для своего периода исследования. Денежные параметры контракта RIH6, комиссии и гарантийное обеспечение также заданы константами.

## Структура

```text
island_model/
├── island_forest.py        # ансамбль и миграция
├── features.py             # признаки и целевая переменная
├── backtester.py           # одиночный walk-forward бэктест
├── portfolio_backtester.py # портфель, сделки и издержки
├── data_loader.py          # MOEX ISS
├── csv_loader.py           # локальные CSV
├── qsh_parser.py           # QScalp v4 / Deals
└── synthetic_data.py       # генератор рыночных сценариев
data/                      # дневные и внутридневные CSV
downloaded_data/csv_files/  # отдельный набор для SBER + GAZP
tests/                     # проверки воспроизводимости синтетики
run_*.py                   # точки входа экспериментов
requirements.txt
```

Зависимости: numpy, pandas, scikit-learn, apimoex, requests, joblib, matplotlib и LightGBM. Метрики включают accuracy, F1, Sharpe, Sortino, Max Drawdown, Profit Factor и Calmar.

## English

Research project for an island ensemble of Random Forest and ExtraTrees models on Moscow Exchange data. It includes technical features, ring migration, walk-forward evaluation, portfolio simulations, CSV / MOEX ISS loaders, synthetic data and RIH6 intraday experiments.

Use Python 3.10+, install `requirements.txt`, then run `python run_backtest.py`. This entry point attempts MOEX ISS first and explicitly falls back to synthetic data. Before running the other experiments, update their hardcoded `/home/user/Moex/...` input or chart paths.

Transaction costs, contract values and the 21% annual risk-free rate are model assumptions. Results are for educational and research use.
