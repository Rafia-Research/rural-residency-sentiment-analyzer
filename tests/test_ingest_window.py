"""Fast tests for backfill and incremental timestamp boundaries."""

from datetime import datetime, timezone

import pandas as pd

import ingest


def sample_rows():
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-12-31T23:59:59Z",
            "2024-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
        ], utc=True),
        "value": ["too_old", "boundary", "middle", "new"],
    })


def test_backfill_uses_configured_month_window(monkeypatch):
    monkeypatch.setattr(ingest, "BACKFILL_MONTHS", 24)
    result = ingest.filter_by_time_window(
        sample_rows(),
        now=pd.Timestamp("2026-01-01T00:00:00Z"),
    )
    assert result["value"].tolist() == ["boundary", "middle", "new"]


def test_incremental_uses_strict_last_run_watermark():
    result = ingest.filter_by_time_window(
        sample_rows(),
        since_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert result["value"].tolist() == ["new"]
