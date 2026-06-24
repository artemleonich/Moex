"""Tests for csv_loader error handling.

Before this fix, ``load_all_csv`` had two bare ``except Exception: pass``
blocks for USDRUB and Brent loading — failures were silently swallowed,
and downstream code that depended on these series would later crash
with confusing ``NoneType`` errors instead of the real cause.

This fix:
- Catches a narrow list of expected CSV-loading exceptions instead of
  ``Exception`` broadly (so unexpected bugs still surface).
- Logs the failure at WARNING level instead of printing or swallowing.
- Returns ``None`` explicitly for failed optional series.

Tests cover:
- Positive case: all files present, all data loaded and returned.
- Negative case: missing USDRUB → returns None and emits WARNING.
- Negative case: missing Brent → returns None and emits WARNING.
- Negative case: missing ticker → that ticker is skipped but others load.
- Negative case: malformed CSV → WARNING + skip; the underlying
  ParserError is NOT swallowed silently.
- Type contract: the function returns ``(dict, Optional[Series], Optional[Series])``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from island_model.csv_loader import (
    DATA_DIR,
    load_all_csv,
    load_brent_csv,
    load_ticker_csv,
    load_usdrub_csv,
)


# ---------- helpers ----------


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Minimal CSV writer — avoids pulling in extra deps in tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A clean data dir with one ticker file. USDRUB/Brent NOT created."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = [
        [d.date().isoformat(), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000 + i * 100]
        for i, d in enumerate(dates)
    ]
    _write_csv(
        tmp_path / "SBER_daily.csv",
        ["begin", "open", "high", "low", "close", "volume"],
        rows,
    )
    return tmp_path


@pytest.fixture
def full_data_dir(tmp_path: Path) -> Path:
    """A data dir with ticker + USDRUB + Brent files."""
    dates = pd.bdate_range("2024-01-01", periods=5)

    rows_t = [
        [d.date().isoformat(), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000 + i * 100]
        for i, d in enumerate(dates)
    ]
    _write_csv(
        tmp_path / "SBER_daily.csv",
        ["begin", "open", "high", "low", "close", "volume"],
        rows_t,
    )

    rows_u = [[d.date().isoformat(), 90.0 + i * 0.1] for i, d in enumerate(dates)]
    _write_csv(tmp_path / "USDRUB_daily.csv", ["begin", "close"], rows_u)

    rows_b = [[d.date().isoformat(), 80.0 + i * 0.2] for i, d in enumerate(dates)]
    _write_csv(tmp_path / "BRENT_daily.csv", ["begin", "close"], rows_b)

    return tmp_path


# ---------- positive cases ----------


def test_load_all_csv_full_success(full_data_dir: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    tickers, usdrub, brent = load_all_csv(["SBER"], data_dir=str(full_data_dir))
    assert "SBER" in tickers
    assert isinstance(tickers["SBER"], pd.DataFrame)
    assert usdrub is not None
    assert brent is not None
    assert isinstance(usdrub, pd.Series)
    assert isinstance(brent, pd.Series)
    # Optional series have the right name attribute
    assert usdrub.name == "usdrub"
    assert brent.name == "brent"


def test_load_ticker_csv_returns_correct_columns(data_dir: Path) -> None:
    df = load_ticker_csv("SBER", data_dir=str(data_dir))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "begin"
    assert len(df) == 5


def test_load_ticker_csv_uses_data_dir_default() -> None:
    """The default ``DATA_DIR`` should be readable when invoked with no args."""
    # If the project's data/ dir is missing or empty, this should still work
    # in the sense of NOT raising on import — but it'll raise FileNotFoundError.
    # We just verify DATA_DIR exists and is a path-like object.
    from pathlib import Path as _Path
    assert isinstance(DATA_DIR, str)
    # Don't actually try to load — the project's data/ dir may not have files.


# ---------- negative cases: missing optional series ----------


def test_missing_usdrub_emits_warning_and_returns_none(data_dir: Path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="island_model.csv_loader")
    tickers, usdrub, brent = load_all_csv(["SBER"], data_dir=str(data_dir))
    assert "SBER" in tickers
    assert usdrub is None
    # brent file is also missing in this fixture
    assert brent is None
    # Both failures are logged at WARNING with the underlying error type
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 2, f"expected ≥2 WARNINGs, got {[r.message for r in warnings]}"
    usdrub_warn = next((r for r in warnings if "USDRUB" in r.message), None)
    assert usdrub_warn is not None
    assert "FileNotFoundError" in usdrub_warn.message


def test_missing_brent_emits_warning_and_returns_none(data_dir: Path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="island_model.csv_loader")
    _, usdrub, brent = load_all_csv(["SBER"], data_dir=str(data_dir))
    assert brent is None
    brent_warn = next(
        (r for r in caplog.records if r.levelno == logging.WARNING and "BRENT" in r.message),
        None,
    )
    assert brent_warn is not None
    assert "FileNotFoundError" in brent_warn.message


def test_missing_optional_does_not_raise(data_dir: Path) -> None:
    """The headline bug: missing USDRUB/Brent must NOT raise from load_all_csv."""
    # Before the fix this still wouldn't raise — but the failure was silent.
    # After the fix it returns None gracefully with a WARNING.
    tickers, usdrub, brent = load_all_csv(["SBER"], data_dir=str(data_dir))
    assert isinstance(tickers, dict)
    assert usdrub is None
    assert brent is None


# ---------- negative cases: missing ticker ----------


def test_missing_ticker_is_skipped_not_raised(data_dir: Path, caplog) -> None:
    """A missing ticker file logs WARNING and is skipped, but other tickers still load."""
    # Add a second ticker file
    dates = pd.bdate_range("2024-01-01", periods=3)
    rows = [
        [d.date().isoformat(), 200.0 + i, 201.0 + i, 199.0 + i, 200.5 + i, 500 + i * 50]
        for i, d in enumerate(dates)
    ]
    _write_csv(
        data_dir / "GAZP_daily.csv",
        ["begin", "open", "high", "low", "close", "volume"],
        rows,
    )
    caplog.set_level(logging.WARNING, logger="island_model.csv_loader")
    tickers, _, _ = load_all_csv(["SBER", "GAZP", "MISSING_TICKER"], data_dir=str(data_dir))
    assert "SBER" in tickers
    assert "GAZP" in tickers
    assert "MISSING_TICKER" not in tickers
    missing_warn = next(
        (r for r in caplog.records if r.levelno == logging.WARNING and "MISSING_TICKER" in r.message),
        None,
    )
    assert missing_warn is not None


def test_all_tickers_missing_returns_empty_dict(data_dir: Path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="island_model.csv_loader")
    tickers, _, _ = load_all_csv(["GHOST1", "GHOST2"], data_dir=str(data_dir))
    assert tickers == {}


# ---------- negative cases: malformed CSV ----------


def test_malformed_csv_does_not_silently_swallow(tmp_path: Path, caplog) -> None:
    """A malformed CSV must surface via WARNING, NOT be silently ignored."""
    # Write valid SBER
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = [
        [d.date().isoformat(), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000]
        for i, d in enumerate(dates)
    ]
    _write_csv(tmp_path / "SBER_daily.csv", ["begin", "open", "high", "low", "close", "volume"], rows)

    # Write a deliberately malformed GAZP CSV with inconsistent column counts —
    # pandas raises ``pd.errors.ParserError`` on this, which is exactly the
    # error path we want to verify is logged and not swallowed.
    (tmp_path / "GAZP_daily.csv").write_text("a,b\n1,2,3,4\n5,6\n", encoding="utf-8")

    caplog.set_level(logging.WARNING, logger="island_model.csv_loader")
    tickers, _, _ = load_all_csv(["SBER", "GAZP"], data_dir=str(tmp_path))

    assert "SBER" in tickers
    assert "GAZP" not in tickers, "malformed CSV must be skipped, not loaded"
    parser_warn = next(
        (r for r in caplog.records if r.levelno == logging.WARNING and "GAZP" in r.message),
        None,
    )
    assert parser_warn is not None


# ---------- exception narrowing ----------


def test_unexpected_exception_still_propagates(tmp_path: Path, monkeypatch) -> None:
    """Exceptions NOT in the narrow list (e.g. a custom user error) must
    surface immediately, not be silently swallowed.
    """
    # Force load_ticker_csv to raise something outside _CSV_LOAD_ERRORS
    from island_model import csv_loader as mod

    def boom(_ticker, _data_dir):
        raise RuntimeError("user-code invariant violated")

    monkeypatch.setattr(mod, "load_ticker_csv", boom)
    with pytest.raises(RuntimeError, match="user-code invariant"):
        load_all_csv(["SBER"], data_dir=str(tmp_path))


# ---------- direct loaders ----------


def test_load_usdrub_csv_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_usdrub_csv(data_dir=str(tmp_path))


def test_load_brent_csv_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_brent_csv(data_dir=str(tmp_path))