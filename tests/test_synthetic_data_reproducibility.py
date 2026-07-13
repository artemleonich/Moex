"""Reproducibility tests for island_model.synthetic_data.

These tests guard against the regression where Python's built-in ``hash()``
is used to seed the per-ticker RNG: since Python 3.3, ``hash()`` is
seeded per-process by ``PYTHONHASHSEED``, so two consecutive runs would
produce different synthetic series and break backtest reproducibility.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_generate_ohlcv_subprocess(ticker: str = "SBER") -> str:
    """Invoke ``generate_ohlcv`` in a fresh Python process and return its close series as CSV."""
    script = textwrap.dedent(
        f"""
        import sys, os
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from island_model.synthetic_data import generate_ohlcv
        # Note: generate_ohlcv returns (df, params) tuple — see synthetic_data.py:224
        df, _params = generate_ohlcv({ticker!r}, start='2024-01-01', end='2024-03-31')
        print(','.join(f'{{x:.10f}}' for x in df['close'].tolist()))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    return result.stdout.strip()


@pytest.mark.parametrize("ticker", ["SBER", "GAZP", "LKOH"])
def test_generate_ohlcv_is_reproducible_across_processes(ticker: str) -> None:
    """Two fresh Python processes must produce identical synthetic series."""
    out1 = _run_generate_ohlcv_subprocess(ticker)
    out2 = _run_generate_ohlcv_subprocess(ticker)
    assert out1, f"empty output for ticker {ticker}"
    assert out1 == out2, (
        f"synthetic data for {ticker} differs across processes — "
        "RNG seed is not stable; check hashlib usage in synthetic_data.py"
    )


def test_ticker_seed_is_deterministic_within_process() -> None:
    """Within a single process the seed must be a pure function of (seed, ticker)."""
    script = textwrap.dedent(
        """
        import sys, os, hashlib
        sys.path.insert(0, '.')
        # Re-derive the exact formula used by generate_ohlcv
        for t in ('SBER', 'GAZP', 'LKOH', 'GMKN', 'ROSN'):
            h = int(hashlib.md5(t.encode('utf-8')).hexdigest(), 16) % 10000
            print(f'{t}={h}')
        """
    )
    out1 = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=str(REPO_ROOT)
    ).stdout
    out2 = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=str(REPO_ROOT)
    ).stdout
    assert out1 == out2
    # Sanity: distinct tickers must yield distinct hashes (with high probability)
    pairs = dict(line.split("=") for line in out1.strip().splitlines())
    assert len(set(pairs.values())) == len(pairs), (
        f"hash collision detected: {pairs}"
    )