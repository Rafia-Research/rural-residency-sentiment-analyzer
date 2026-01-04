# tests/test_pii.py
"""
Unit tests for PII detection module.
Run with: pytest tests/test_pii.py -v
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pii import detect_and_redact_pii, is_whitelisted

class TestPIIWhitelist:
    """Test suite for whitelist functionality."""
    
    def test_tv_characters_whitelisted(self):
        """TV character names should be whitelisted."""
        assert is_whitelisted("Dr. House") == True
        assert is_whitelisted("Grey's Anatomy") == True
        assert is_whitelisted("Meredith Grey") == True or is_whitelisted("meredith") == True
    
    def test_real_names_not_whitelisted(self):
        """Real-sounding names should not be whitelisted."""
        assert is_whitelisted("Dr. John Smith") == False
        assert is_whitelisted("Sarah Johnson") == False


class TestPIIDetection:
    """Test suite for PII detection and redaction."""
    
    def test_phone_number_detected(self):
        """Phone numbers should be detected and redacted."""
        df = pd.DataFrame({
            'id': ['test1'],
            'original_text': ["Call me at 555-123-4567 for more info about the program."]
        })
        result, audit = detect_and_redact_pii(df)
        
        assert "[REDACTED]" in result['redacted_text'].iloc[0]
        assert "555-123-4567" not in result['redacted_text'].iloc[0]
        assert result['pii_detected'].iloc[0] == True
    
    def test_email_detected(self):
        """Email addresses should be detected and redacted."""
        df = pd.DataFrame({
            'id': ['test2'],
            'original_text': ["Email me at doctor.smith@hospital.com for questions"]
        })
        result, audit = detect_and_redact_pii(df)
        
        assert "doctor.smith@hospital.com" not in result['redacted_text'].iloc[0]
        assert result['pii_detected'].iloc[0] == True
    
    def test_name_detected(self):
        """Personal names should be detected and redacted."""
        df = pd.DataFrame({
            'id': ['test3'],
            'original_text': ["Dr. John Smith at Springfield Hospital was my attending."]
        })
        result, audit = detect_and_redact_pii(df)
        
        # Presidio is probabilistic, might not always catch generic names without context
        # But "John Smith" is strong.
        # "Dr. John Smith" -> might extract "John Smith" as PERSON
        # Check that original is not present
        assert "John Smith" not in result['redacted_text'].iloc[0]
        assert result['pii_detected'].iloc[0] == True
    
    def test_medical_terms_not_redacted(self):
        """Common medical terms should NOT be redacted."""
        df = pd.DataFrame({
            'id': ['test4'],
            'original_text': ["The residency program focuses on family medicine and internal medicine."]
        })
        result, audit = detect_and_redact_pii(df)
        
        redacted = result['redacted_text'].iloc[0].lower()
        assert "residency" in redacted
        assert "family medicine" in redacted
        assert "internal medicine" in redacted
    
    def test_whitelist_prevents_redaction(self):
        """Whitelisted terms should not be redacted."""
        df = pd.DataFrame({
            'id': ['test5'],
            'original_text': ["It's not like Grey's Anatomy at all. Dr. House would never survive here."]
        })
        result, audit = detect_and_redact_pii(df)
        
        assert "Grey's Anatomy" in result['redacted_text'].iloc[0] or "grey" in result['redacted_text'].iloc[0].lower()
    
    def test_subreddit_names_preserved(self):
        """Subreddit references should not be redacted."""
        df = pd.DataFrame({
            'id': ['test6'],
            'original_text': ["I posted this on r/Residency and r/medicalschool yesterday."]
        })
        result, audit = detect_and_redact_pii(df)
        
        # Presidio might not catch r/Residency as PII anyway, but let's ensure
        assert "r/Residency" in result['redacted_text'].iloc[0] or "r/residency" in result['redacted_text'].iloc[0].lower()
    
    def test_audit_log_created(self):
        """Audit log should track all redactions."""
        df = pd.DataFrame({
            'id': ['post123'],
            'original_text': ["Contact John Doe at john.doe@email.com or 555-1234"]
        })
        result, audit = detect_and_redact_pii(df)
        
        assert len(audit) > 0
        assert 'post_id' in audit.columns
        assert 'entity_type' in audit.columns
        assert 'confidence' in audit.columns
        assert audit['post_id'].iloc[0] == 'post123'
    
    def test_pii_types_column_format(self):
        """PII types should be listed in comma-separated format."""
        df = pd.DataFrame({
            'id': ['test7'],
            'original_text': ["Call John at 555-1234 or email john@email.com"]
        })
        result, audit = detect_and_redact_pii(df)
        
        pii_types = result['pii_types'].iloc[0]
        assert isinstance(pii_types, str)
        assert len(pii_types) > 0
    
    def test_pii_count_column(self):
        """PII count should accurately reflect number of entities."""
        df = pd.DataFrame({
            'id': ['test8'],
            'original_text': ["Call John Smith at 555-1234 or email john@email.com"]
        })
        result, audit = detect_and_redact_pii(df)
        
        assert 'pii_count' in result.columns
        assert result['pii_count'].iloc[0] >= 2
    
    def test_empty_text_handling(self):
        """Empty text should not crash."""
        df = pd.DataFrame({
            'id': ['test9', 'test10'],
            'original_text': ["", None]
        })
        result, audit = detect_and_redact_pii(df)
        
        # Preprocessing usually converts None->"", but logic handles it
        # If passed directly, None might crash redact_text if not robust
        # Check redact_text logic: "if not text or not isinstance(text, str): return "", []"
        # It is robust.
        
        assert len(result) == 2
        assert 'redacted_text' in result.columns
    
    def test_no_pii_text_unchanged(self):
        """Text without PII should pass through unchanged."""
        original = "Rural programs are challenging but rewarding for those who choose them."
        df = pd.DataFrame({
            'id': ['test11'],
            'original_text': [original]
        })
        result, audit = detect_and_redact_pii(df)
        
        assert result['redacted_text'].iloc[0] == original
        assert result['pii_detected'].iloc[0] == False
        assert result['pii_count'].iloc[0] == 0
    
    def test_location_detection(self):
        """Specific location mentions should be detected.
        
        Note: Presidio's LOCATION entity detects city/country names, not 
        street addresses (which would require a custom recognizer).
        """
        df = pd.DataFrame({
            'id': ['test12'],
            'original_text': ["I moved from New York City to work in rural Oregon."]
        })
        result, audit = detect_and_redact_pii(df)
        
        # Presidio should detect at least one location (New York City or Oregon)
        assert result['pii_detected'].iloc[0] == True
        assert "LOCATION" in result['pii_types'].iloc[0]
