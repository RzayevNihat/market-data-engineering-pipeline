"""Compare rolling-mean implementations and report time/RSS observations."""

from __future__ import annotations

import gc
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from numba import njit


@njit(cache=True)
def sma_numba(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(values.shape[0], dtype=np.float64)
    result[:] = np.nan
    running_sum = 0.0
    for index in range(values.shape[0]):
        running_sum += values[index]
        if index >= window:
            running_sum -= values[index - window]
        if index >= window - 1:
            result[index] = running_sum / window
    return result


def sma_numpy(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.empty(values.shape[0] + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(values, out=cumulative[1:])
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] >= window:
        result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def sma_pandas(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def _measure_peak_rss_mb(function, values: np.ndarray, window: int, repeats: int) -> tuple[float, float]:
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    samples = [before]
    stop_event = threading.Event()

    def monitor() -> None:
        while not stop_event.wait(0.005):
            try:
                samples.append(process.memory_info().rss)
            except psutil.Error:
                break

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    start = time.perf_counter()
    for _ in range(repeats):
        result = function(values, window)
        # Keep the most recent output alive until after the timer.
        _ = result[-1] if result.size else None
    elapsed = (time.perf_counter() - start) / repeats
    stop_event.set()
    monitor_thread.join()
    peak_delta_mb = max(0, max(samples) - before) / (1024 * 1024)
    return elapsed, peak_delta_mb


def run_benchmark(rows: int = 1_000_000, window: int = 60, repeats: int = 5) -> pd.DataFrame:
    if rows < window or repeats < 1:
        raise ValueError("rows must be >= window and repeats must be >= 1")
    rng = np.random.default_rng(42)
    values = rng.normal(loc=100.0, scale=2.0, size=rows).astype(np.float64)

    # Compile Numba before timing. JIT compilation is a one-time cost and is
    # reported separately in a real production benchmark if relevant.
    numba_output = sma_numba(values, window)
    pandas_output = sma_pandas(values, window)
    numpy_output = sma_numpy(values, window)
    np.testing.assert_allclose(pandas_output, numpy_output, rtol=1e-9, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(pandas_output, numba_output, rtol=1e-9, atol=1e-9, equal_nan=True)

    implementations = [
        ("pandas_rolling", sma_pandas),
        ("numpy_cumsum", sma_numpy),
        ("numba_jit", sma_numba),
    ]
    results = []
    for name, function in implementations:
        seconds, peak_rss_mb = _measure_peak_rss_mb(function, values, window, repeats)
        results.append({
            "implementation": name,
            "rows": rows,
            "window": window,
            "repeats": repeats,
            "avg_seconds": seconds,
            "rows_per_second": rows / seconds if seconds else None,
            "observed_peak_rss_increase_mb": peak_rss_mb,
            "correctness_checked_against_pandas": True,
        })
    return pd.DataFrame(results).sort_values("avg_seconds").reset_index(drop=True)
