"""Download public Binance Spot OHLCV candles and preserve the raw response."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .settings import BINANCE_BASE_URL, INTERVAL_TO_MS, KLINES_PATH, RAW_KLINE_COLUMNS, RAW_DIR

NUMERIC_COLUMNS = tuple(
    column
    for column in RAW_KLINE_COLUMNS
    if column not in {
        "open_time_ms",
        "close_time_ms",
        "number_of_trades",
        "ignore",
    }
)

def _get_page(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
) -> list[list[Any]]:
    """Fetch one page, retrying temporary rate-limit/server errors."""
    url = f"{BINANCE_BASE_URL}{KLINES_PATH}"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": min(limit, 1000),
    }

    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code in (418, 429):
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else min(2 ** attempt, 30)
                time.sleep(wait_seconds)
                continue
            if response.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected API response: {payload}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Could not download Binance klines: {exc}") from exc
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError("Binance request failed after the maximum number of retries.")


def download_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 30,
    output_path: Path | None = None,
) -> tuple[Path, int]:
    """Download the last N days of closed candles and save the raw JSON pages.

    Binance's kline endpoint has a maximum page size of 1000 records. We move
    the cursor to the next candle after each page to avoid duplicate requests.
    """
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}. Choose from {list(INTERVAL_TO_MS)}")
    if days < 1:
        raise ValueError("days must be at least 1")

    interval_ms = INTERVAL_TO_MS[interval]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Exclude the currently forming candle. The endpoint's endTime is inclusive.
    end_time_ms = (now_ms // interval_ms) * interval_ms - 1
    start_time_ms = end_time_ms - days * 86_400_000
    start_time_ms = (start_time_ms // interval_ms) * interval_ms

    rows: list[list[Any]] = []
    cursor = start_time_ms
    session = requests.Session()
    session.headers.update({"User-Agent": "market-data-pipeline-learning-project/0.1"})

    while cursor <= end_time_ms:
        page = _get_page(
            session=session,
            symbol=symbol,
            interval=interval,
            start_time_ms=cursor,
            end_time_ms=end_time_ms,
            limit=1000,
        )
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + interval_ms
        if next_cursor <= cursor:
            raise RuntimeError("Pagination did not advance; stopping to prevent an infinite loop.")
        cursor = next_cursor
        time.sleep(0.15)  # Be polite to the public API.

    if not rows:
        raise RuntimeError("The API returned no candles. Check connectivity, symbol, and date range.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = RAW_DIR / f"{symbol.upper()}_{interval}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": "Binance Spot public REST API",
        "endpoint": KLINES_PATH,
        "symbol": symbol.upper(),
        "interval": interval,
        "requested_days": days,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(rows),
        "records": rows,
    }
    output_path.write_text(json.dumps(metadata), encoding="utf-8")
    return output_path, len(rows)


def load_raw_klines(raw_path: Path) -> pd.DataFrame:
    """Parse preserved raw JSON into a typed OHLCV DataFrame."""
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records found in {raw_path}")

    frame = pd.DataFrame(records, columns=RAW_KLINE_COLUMNS)
    frame["open_time"] = pd.to_datetime(frame.pop("open_time_ms"), unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame.pop("close_time_ms"), unit="ms", utc=True)

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["number_of_trades"] = frame["number_of_trades"].astype("Int64")
    frame = frame.drop(columns=["ignore"])
    frame["symbol"] = payload.get("symbol", "UNKNOWN")
    frame["interval"] = payload.get("interval", "UNKNOWN")
    return frame
