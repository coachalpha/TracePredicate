"""Database package for TracePredicate."""

from .connection import DatabaseConfig, DatabaseManager, get_database_manager, get_db_session, init_database
from .models import (
    AdverseEvent, AnalysisRun, Base, Device, DeviceClass, DeviceRecallAssociation,
    EventType, HFEClassification, LDIScore, NetworkNode, PredicateRelationship,
    Recall, RecallClass, RecallStatus, ValidationResult
)
from .operations import (
    AdverseEventOperations, AnalysisOperations, DeviceOperations, 
    LDIOperations, PredicateOperations, get_database_statistics
)

__all__ = [
    # Connection management
    "DatabaseConfig",
    "DatabaseManager", 
    "get_database_manager",
    "get_db_session",
    "init_database",
    
    # Models
    "Base",
    "Device",
    "PredicateRelationship", 
    "AdverseEvent",
    "Recall",
    "DeviceRecallAssociation",
    "LDIScore",
    "NetworkNode",
    "AnalysisRun",
    "ValidationResult",
    
    # Enums
    "DeviceClass",
    "EventType",
    "HFEClassification",
    "RecallClass", 
    "RecallStatus",
    
    # Operations
    "DeviceOperations",
    "PredicateOperations",
    "AdverseEventOperations", 
    "LDIOperations",
    "AnalysisOperations",
    "get_database_statistics",
]