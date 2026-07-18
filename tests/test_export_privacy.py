"""Fast tests for the privacy boundary at the export layer."""

import pandas as pd
import pytest

import export


def make_processed_row(record_id="t3_sensitive"):
    return pd.DataFrame({
        "id": [record_id],
        "parent_id": [None],
        "author": ["searchable_username"],
        "reddit_url": ["https://reddit.com/r/example/comments/sensitive"],
        "text": ["Email jane@example.com"],
        "original_text": ["Email jane@example.com"],
        "redacted_text": ["Email [REDACTED]"],
        "redacted_clean_text": ["email redacted"],
        "timestamp": [pd.Timestamp("2026-01-15T12:30:00Z")],
        "subreddit": ["Residency"],
        "type": ["Post"],
        "score": [4],
        "sentiment_score": [-0.8],
        "sentiment_label": ["Negative"],
        "sentiment_confidence": [0.8],
        "sentiment_uncertain": [False],
        "pii_detected": [True],
        "pii_types": ["EMAIL_ADDRESS"],
        "pii_count": [1],
    })


def test_safe_export_excludes_direct_identifiers_and_raw_text():
    safe = export.build_safe_export_dataframe(make_processed_row())

    assert {"id", "parent_id", "author", "reddit_url", "text", "original_text", "timestamp"}.isdisjoint(safe.columns)
    assert safe.loc[0, "record_id"] != "t3_sensitive"
    assert len(safe.loc[0, "record_id"]) == 20
    assert "jane@example.com" not in safe.to_csv(index=False)
    assert safe.loc[0, "redacted_text"] == "Email [REDACTED]"
    assert safe.loc[0, "year_month"] == "2026-01"


def test_export_refuses_unredacted_rows():
    unsafe = make_processed_row().drop(columns=["redacted_text"])
    with pytest.raises(ValueError, match="PII redaction"):
        export.build_safe_export_dataframe(unsafe)


def test_incremental_merge_preserves_history_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "OUTPUT_DIR", tmp_path)
    old = export.build_safe_export_dataframe(make_processed_row("old_record"))
    old.to_csv(tmp_path / "reddit_sentiment.csv", index=False)

    merged = export.merge_incremental_history(make_processed_row("new_record"))
    assert len(merged) == 2

    replacement = make_processed_row("old_record")
    replacement["score"] = 99
    merged = export.merge_incremental_history(replacement)
    assert len(merged) == 1
    assert merged.loc[0, "score"] == 99


def test_audit_log_uses_pseudonymous_record_id():
    audit = pd.DataFrame({
        "post_id": ["t3_sensitive"],
        "entity_type": ["EMAIL_ADDRESS"],
        "start_position": [6],
        "end_position": [22],
        "confidence": [0.95],
        "timestamp": ["2026-01-15T12:31:00"],
    })
    safe = export.build_safe_audit_dataframe(audit)
    assert "post_id" not in safe.columns
    assert safe.loc[0, "record_id"] == export.pseudonymize_id("t3_sensitive")
