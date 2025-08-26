"""
Database models for TracePredicate system.

This module defines SQLAlchemy models for storing FDA 510(k) devices,
MAUDE adverse events, recalls, LDI scores, and network analysis results.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, 
    JSON, String, Text, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DeviceClass(PyEnum):
    """FDA device class enumeration."""
    CLASS_I = "Class I"
    CLASS_II = "Class II"
    CLASS_III = "Class III"


class EventType(PyEnum):
    """MAUDE event type enumeration."""
    DEATH = "D"
    INJURY = "I"
    MALFUNCTION = "M"
    OTHER = "O"


class HFEClassification(PyEnum):
    """Human Factors Engineering classification."""
    DEVICE_FAILURE = "device_failure"
    CLEAR_USE_ERROR = "clear_use_error"
    DESIGN_INDUCED_USE_ERROR = "design_induced_use_error"
    UNCLEAR = "unclear"


class RecallClass(PyEnum):
    """FDA recall class enumeration."""
    CLASS_I = "Class I"
    CLASS_II = "Class II"
    CLASS_III = "Class III"


class RecallStatus(PyEnum):
    """Recall status enumeration."""
    ONGOING = "Ongoing"
    COMPLETED = "Completed"
    TERMINATED = "Terminated"


class Device(Base):
    """510(k) cleared medical devices."""
    __tablename__ = 'devices'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    k_number = Column(String(10), unique=True, nullable=False, index=True)
    
    # Device information
    device_name = Column(String(500), nullable=False)
    applicant = Column(String(200), nullable=False)
    approval_date = Column(DateTime, nullable=True, index=True)
    device_code = Column(String(10), nullable=True, index=True)  # e.g., 'KWA'
    product_code = Column(String(10), nullable=True)
    device_class = Column(Enum(DeviceClass), nullable=True)
    decision = Column(String(50), nullable=True)
    
    # URLs and content
    summary_url = Column(String(500), nullable=True)
    pdf_content = Column(Text, nullable=True)
    
    # Technical parameters (JSON field)
    technical_parameters = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    predicate_relationships_as_child = relationship(
        "PredicateRelationship", 
        foreign_keys="PredicateRelationship.child_device_id",
        back_populates="child_device"
    )
    predicate_relationships_as_parent = relationship(
        "PredicateRelationship",
        foreign_keys="PredicateRelationship.parent_device_id", 
        back_populates="parent_device"
    )
    adverse_events = relationship("AdverseEvent", back_populates="device")
    recalls = relationship("DeviceRecallAssociation", back_populates="device")
    ldi_scores = relationship("LDIScore", foreign_keys="LDIScore.device_id", back_populates="device")
    
    def __repr__(self):
        return f"<Device(k_number='{self.k_number}', name='{self.device_name[:50]}')>"


class PredicateRelationship(Base):
    """Predicate device relationships between 510(k) clearances."""
    __tablename__ = 'predicate_relationships'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    child_device_id = Column(Integer, ForeignKey('devices.id'), nullable=False)
    parent_device_id = Column(Integer, ForeignKey('devices.id'), nullable=False)
    
    # Relationship details
    predicate_type = Column(String(50), nullable=True)  # e.g., 'primary', 'secondary'
    reference_date = Column(DateTime, nullable=True)
    confidence_score = Column(Float, nullable=True)  # How confident we are in this relationship
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    child_device = relationship("Device", foreign_keys=[child_device_id])
    parent_device = relationship("Device", foreign_keys=[parent_device_id])
    
    # Ensure unique relationships
    __table_args__ = (
        UniqueConstraint('child_device_id', 'parent_device_id', name='unique_predicate_relationship'),
    )
    
    def __repr__(self):
        return f"<PredicateRelationship(child={self.child_device_id}, parent={self.parent_device_id})>"


class AdverseEvent(Base):
    """MAUDE adverse event reports."""
    __tablename__ = 'adverse_events'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    mdr_report_key = Column(String(20), unique=True, nullable=False, index=True)
    
    # Event information
    event_date = Column(DateTime, nullable=True, index=True)
    event_type = Column(Enum(EventType), nullable=True)
    
    # Device association
    device_id = Column(Integer, ForeignKey('devices.id'), nullable=True)
    device_name = Column(String(500), nullable=True)  # As reported in MAUDE
    manufacturer = Column(String(200), nullable=True)
    brand_name = Column(String(200), nullable=True)
    generic_name = Column(String(200), nullable=True)
    product_code = Column(String(10), nullable=True, index=True)
    device_class = Column(String(10), nullable=True)
    
    # Event descriptions
    event_description = Column(Text, nullable=True)
    device_problem = Column(Text, nullable=True)
    patient_problem = Column(Text, nullable=True)
    
    # Reporter information
    reporter_occupation = Column(String(100), nullable=True)
    report_source = Column(String(100), nullable=True)
    
    # HFE Classification
    hfe_classification = Column(Enum(HFEClassification), nullable=True)
    hfe_confidence = Column(Float, nullable=True)  # Confidence in classification
    
    # Metadata
    date_received = Column(DateTime, nullable=True)
    report_number = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    device = relationship("Device", back_populates="adverse_events")
    
    def __repr__(self):
        return f"<AdverseEvent(mdr_key='{self.mdr_report_key}', type='{self.event_type}')>"


class Recall(Base):
    """FDA medical device recalls."""
    __tablename__ = 'recalls'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    recall_number = Column(String(50), unique=True, nullable=False, index=True)
    
    # Recall information
    recall_date = Column(DateTime, nullable=True, index=True)
    recall_class = Column(Enum(RecallClass), nullable=False)
    recall_status = Column(Enum(RecallStatus), nullable=True)
    
    # Product information
    product_description = Column(Text, nullable=False)
    manufacturer = Column(String(200), nullable=False)
    brand_name = Column(String(200), nullable=True)
    product_code = Column(String(10), nullable=True, index=True)
    
    # Recall details
    reason_for_recall = Column(Text, nullable=True)
    recall_initiation_date = Column(DateTime, nullable=True)
    recall_distribution_pattern = Column(Text, nullable=True)
    quantity_in_commerce = Column(Integer, nullable=True)
    quantity_recalled = Column(Integer, nullable=True)
    
    # Risk assessment
    root_cause = Column(Text, nullable=True)
    risk_to_health = Column(Text, nullable=True)
    corrective_action = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    device_associations = relationship("DeviceRecallAssociation", back_populates="recall")
    
    def __repr__(self):
        return f"<Recall(number='{self.recall_number}', class='{self.recall_class}')>"


class DeviceRecallAssociation(Base):
    """Association between devices and recalls."""
    __tablename__ = 'device_recall_associations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    device_id = Column(Integer, ForeignKey('devices.id'), nullable=False)
    recall_id = Column(Integer, ForeignKey('recalls.id'), nullable=False)
    
    # Association details
    confidence_score = Column(Float, nullable=True)  # How confident we are in this association
    association_method = Column(String(50), nullable=True)  # 'k_number', 'name_match', etc.
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    device = relationship("Device", back_populates="recalls")
    recall = relationship("Recall", back_populates="device_associations")
    
    # Ensure unique associations
    __table_args__ = (
        UniqueConstraint('device_id', 'recall_id', name='unique_device_recall'),
    )
    
    def __repr__(self):
        return f"<DeviceRecallAssociation(device={self.device_id}, recall={self.recall_id})>"


class LDIScore(Base):
    """Lineage Drift Index scores for devices."""
    __tablename__ = 'ldi_scores'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey('devices.id'), nullable=False)
    
    # LDI components
    semantic_distance = Column(Float, nullable=True)
    parameter_difference = Column(Float, nullable=True)
    chain_length = Column(Integer, nullable=True)
    
    # Final LDI score and weights used
    ldi_score = Column(Float, nullable=False)
    semantic_weight = Column(Float, nullable=False)
    parameter_weight = Column(Float, nullable=False)
    chain_weight = Column(Float, nullable=False)
    
    # Calculation metadata
    calculation_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    calculation_version = Column(String(20), nullable=True)  # Version of algorithm used
    base_predicate_id = Column(Integer, ForeignKey('devices.id'), nullable=True)  # Primary predicate used
    
    # Additional metrics
    risk_percentile = Column(Float, nullable=True)  # Percentile rank among all devices
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    
    # Relationships
    device = relationship("Device", foreign_keys=[device_id], back_populates="ldi_scores")
    base_predicate = relationship("Device", foreign_keys=[base_predicate_id])
    
    def __repr__(self):
        return f"<LDIScore(device_id={self.device_id}, score={self.ldi_score:.3f})>"


class NetworkNode(Base):
    """Network analysis nodes (devices in the predicate network)."""
    __tablename__ = 'network_nodes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey('devices.id'), nullable=False, unique=True)
    
    # Network centrality metrics
    degree_centrality = Column(Float, nullable=True)
    betweenness_centrality = Column(Float, nullable=True)
    closeness_centrality = Column(Float, nullable=True)
    eigenvector_centrality = Column(Float, nullable=True)
    pagerank = Column(Float, nullable=True)
    
    # Network position
    generation_level = Column(Integer, nullable=True)  # Distance from root nodes
    is_root_node = Column(Boolean, default=False)
    is_leaf_node = Column(Boolean, default=False)
    
    # Risk propagation metrics
    risk_influence_score = Column(Float, nullable=True)
    downstream_device_count = Column(Integer, nullable=True)
    
    # Calculation metadata
    calculation_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    network_version = Column(String(20), nullable=True)
    
    # Relationships
    device = relationship("Device")
    
    def __repr__(self):
        return f"<NetworkNode(device_id={self.device_id}, generation={self.generation_level})>"


class AnalysisRun(Base):
    """Track analysis runs and their parameters."""
    __tablename__ = 'analysis_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Run identification
    run_name = Column(String(100), nullable=False)
    run_type = Column(String(50), nullable=False)  # 'ldi_calculation', 'network_analysis', etc.
    
    # Run parameters (stored as JSON)
    parameters = Column(JSON, nullable=True)
    
    # Results summary
    devices_processed = Column(Integer, nullable=True)
    success_count = Column(Integer, nullable=True)
    error_count = Column(Integer, nullable=True)
    
    # Timing
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Status and results
    status = Column(String(20), nullable=False, default='running')  # 'running', 'completed', 'failed'
    error_message = Column(Text, nullable=True)
    results_summary = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<AnalysisRun(name='{self.run_name}', type='{self.run_type}', status='{self.status}')>"


class ValidationResult(Base):
    """Results from LDI validation against real-world outcomes."""
    __tablename__ = 'validation_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Validation run info
    validation_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    validation_type = Column(String(50), nullable=False)  # 'adverse_events', 'recalls', 'expert_review'
    
    # Dataset info
    device_count = Column(Integer, nullable=False)
    outcome_count = Column(Integer, nullable=False)  # Number of adverse events/recalls
    
    # Statistical results
    correlation_coefficient = Column(Float, nullable=True)  # Spearman's rho
    correlation_p_value = Column(Float, nullable=True)
    r_squared = Column(Float, nullable=True)
    
    # Model performance metrics
    auc_score = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Threshold analysis
    optimal_threshold = Column(Float, nullable=True)
    sensitivity_at_threshold = Column(Float, nullable=True)
    specificity_at_threshold = Column(Float, nullable=True)
    
    # Additional metrics
    validation_notes = Column(Text, nullable=True)
    expert_agreement_rate = Column(Float, nullable=True)  # For expert validation
    
    def __repr__(self):
        return f"<ValidationResult(type='{self.validation_type}', correlation={self.correlation_coefficient:.3f})>"