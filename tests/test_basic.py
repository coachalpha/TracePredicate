"""
Basic tests for TracePredicate system.

This module contains basic integration tests to verify the system components
are working together correctly.
"""

import pytest
from datetime import datetime

from trace_predicate.config.settings import get_settings
from trace_predicate.data_collection.fda_510k_scraper import FDA510KScraper, Device510K
from trace_predicate.database import init_database, get_db_session
from trace_predicate.database.operations import DeviceOperations


class TestSettings:
    """Test configuration settings."""
    
    def test_load_settings(self):
        """Test that settings load correctly."""
        settings = get_settings()
        assert settings.app.app_name == "TracePredicate"
        assert settings.app.app_version == "0.1.0"
    
    def test_validate_settings(self):
        """Test settings validation."""
        settings = get_settings()
        warnings = settings.validate_settings()
        # Should return a list (may contain warnings)
        assert isinstance(warnings, list)
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        settings = get_settings()
        url = settings.database.database_url
        assert "postgresql://" in url
        assert settings.database.database in url


class TestDataCollection:
    """Test data collection components."""
    
    def test_device_510k_creation(self):
        """Test Device510K object creation."""
        device = Device510K(
            k_number="K123456",
            device_name="Test Device",
            applicant="Test Company",
            approval_date=datetime.now(),
            device_code="KWA",
            product_code="KWA",
            predicate_devices=["K111111", "K222222"],
            decision="Substantially Equivalent"
        )
        
        assert device.k_number == "K123456"
        assert device.device_name == "Test Device"
        assert len(device.predicate_devices) == 2
        assert device.technical_parameters == {}
    
    def test_fda_scraper_initialization(self):
        """Test FDA scraper can be initialized."""
        scraper = FDA510KScraper()
        assert scraper.session_timeout == 30
        assert scraper.max_concurrent == 5
        assert scraper.BASE_URL


class TestDatabase:
    """Test database operations."""
    
    @pytest.fixture(scope="class")
    def db_session(self):
        """Provide a database session for testing."""
        # Note: In real tests, you'd use a test database
        with get_db_session() as session:
            yield session
    
    def test_device_operations(self, db_session):
        """Test basic device operations."""
        # Create a test device
        device = DeviceOperations.create_device(
            db_session,
            k_number="K999999",
            device_name="Test Hip Implant",
            applicant="Test Medical Corp",
            device_code="KWA"
        )
        
        assert device is not None
        assert device.k_number == "K999999"
        assert device.device_name == "Test Hip Implant"
        
        # Retrieve the device
        retrieved = DeviceOperations.get_device_by_k_number(db_session, "K999999")
        assert retrieved is not None
        assert retrieved.k_number == "K999999"
        
        # Clean up (remove test device)
        db_session.delete(device)
        db_session.commit()


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from trace_predicate.cli import app
        assert app is not None
    
    def test_api_import(self):
        """Test that API module can be imported."""
        from trace_predicate.api import app
        assert app is not None


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_system_imports(self):
        """Test that all major system components can be imported."""
        # Data collection
        from trace_predicate.data_collection import FDA510KScraper, MAUDEInterface, FDARecallScraper
        
        # Database
        from trace_predicate.database import get_db_session, DeviceOperations
        
        # Settings
        from trace_predicate.config.settings import get_settings
        
        # CLI and API
        from trace_predicate.cli import app as cli_app
        from trace_predicate.api import app as api_app
        
        assert all([
            FDA510KScraper, MAUDEInterface, FDARecallScraper,
            get_db_session, DeviceOperations,
            get_settings,
            cli_app, api_app
        ])


def test_placeholder():
    """Placeholder test to ensure pytest runs."""
    assert True


if __name__ == "__main__":
    pytest.main([__file__])