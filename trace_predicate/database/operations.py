"""
Database operations and utility functions for TracePredicate.

This module provides high-level database operations for storing and
retrieving FDA 510(k) devices, adverse events, recalls, and analysis results.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import and_, or_, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from .connection import get_db_session
from .models import (
    AdverseEvent, AnalysisRun, Device, DeviceRecallAssociation, 
    HFEClassification, LDIScore, NetworkNode, PredicateRelationship,
    Recall, ValidationResult
)

logger = logging.getLogger(__name__)


class DeviceOperations:
    """Operations for managing 510(k) devices."""
    
    @staticmethod
    def create_device(
        session: Session,
        k_number: str,
        device_name: str,
        applicant: str,
        approval_date: Optional[datetime] = None,
        device_code: Optional[str] = None,
        **kwargs
    ) -> Device:
        """Create a new device record."""
        device = Device(
            k_number=k_number,
            device_name=device_name,
            applicant=applicant,
            approval_date=approval_date,
            device_code=device_code,
            **kwargs
        )
        
        try:
            session.add(device)
            session.flush()  # Get the ID
            logger.debug(f"Created device: {k_number}")
            return device
        except IntegrityError as e:
            session.rollback()
            logger.warning(f"Device {k_number} already exists: {e}")
            # Return existing device
            return session.query(Device).filter(Device.k_number == k_number).first()
    
    @staticmethod
    def get_device_by_k_number(session: Session, k_number: str) -> Optional[Device]:
        """Get device by K-number."""
        return session.query(Device).filter(Device.k_number == k_number).first()
    
    @staticmethod
    def get_devices_by_code(session: Session, device_code: str) -> List[Device]:
        """Get all devices with a specific device code."""
        return session.query(Device).filter(Device.device_code == device_code).all()
    
    @staticmethod
    def update_device_parameters(session: Session, device_id: int, parameters: Dict) -> bool:
        """Update technical parameters for a device."""
        try:
            device = session.query(Device).get(device_id)
            if device:
                device.technical_parameters = parameters
                device.updated_at = datetime.utcnow()
                logger.debug(f"Updated parameters for device {device.k_number}")
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating device parameters: {e}")
            return False
    
    @staticmethod
    def bulk_create_devices(session: Session, device_data: List[Dict]) -> Tuple[int, int]:
        """
        Bulk create devices from a list of dictionaries.
        
        Returns:
            Tuple of (created_count, skipped_count)
        """
        created_count = 0
        skipped_count = 0
        
        for data in device_data:
            try:
                # Check if device already exists
                existing = session.query(Device).filter(
                    Device.k_number == data['k_number']
                ).first()
                
                if existing:
                    skipped_count += 1
                    continue
                
                device = Device(**data)
                session.add(device)
                created_count += 1
                
            except Exception as e:
                logger.warning(f"Error creating device {data.get('k_number', 'UNKNOWN')}: {e}")
                skipped_count += 1
        
        try:
            session.commit()
            logger.info(f"Bulk created {created_count} devices, skipped {skipped_count}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error in bulk device creation: {e}")
            return 0, len(device_data)
        
        return created_count, skipped_count


class PredicateOperations:
    """Operations for managing predicate relationships."""
    
    @staticmethod
    def create_predicate_relationship(
        session: Session,
        child_k_number: str,
        parent_k_number: str,
        predicate_type: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> Optional[PredicateRelationship]:
        """Create a predicate relationship between devices."""
        
        # Get device IDs
        child_device = DeviceOperations.get_device_by_k_number(session, child_k_number)
        parent_device = DeviceOperations.get_device_by_k_number(session, parent_k_number)
        
        if not child_device or not parent_device:
            logger.warning(f"Cannot create predicate relationship: missing devices {child_k_number} -> {parent_k_number}")
            return None
        
        try:
            relationship = PredicateRelationship(
                child_device_id=child_device.id,
                parent_device_id=parent_device.id,
                predicate_type=predicate_type,
                confidence_score=confidence_score
            )
            
            session.add(relationship)
            session.flush()
            logger.debug(f"Created predicate relationship: {child_k_number} -> {parent_k_number}")
            return relationship
            
        except IntegrityError:
            logger.debug(f"Predicate relationship already exists: {child_k_number} -> {parent_k_number}")
            return None
    
    @staticmethod
    def get_device_predicates(session: Session, k_number: str) -> List[Device]:
        """Get all predicate devices for a given device."""
        device = DeviceOperations.get_device_by_k_number(session, k_number)
        if not device:
            return []
        
        predicates = session.query(Device).join(
            PredicateRelationship,
            Device.id == PredicateRelationship.parent_device_id
        ).filter(
            PredicateRelationship.child_device_id == device.id
        ).all()
        
        return predicates
    
    @staticmethod
    def get_device_descendants(session: Session, k_number: str) -> List[Device]:
        """Get all devices that use this device as a predicate."""
        device = DeviceOperations.get_device_by_k_number(session, k_number)
        if not device:
            return []
        
        descendants = session.query(Device).join(
            PredicateRelationship,
            Device.id == PredicateRelationship.child_device_id
        ).filter(
            PredicateRelationship.parent_device_id == device.id
        ).all()
        
        return descendants
    
    @staticmethod
    def get_predicate_chain(session: Session, k_number: str, max_depth: int = 10) -> List[List[str]]:
        """
        Get the full predicate chain(s) for a device.
        
        Returns:
            List of chains, where each chain is a list of K-numbers from root to target device
        """
        device = DeviceOperations.get_device_by_k_number(session, k_number)
        if not device:
            return []
        
        def _build_chains(device_id: int, current_chain: List[str], depth: int) -> List[List[str]]:
            if depth > max_depth:
                return [current_chain]
            
            # Get direct predicates
            predicates = session.query(PredicateRelationship).filter(
                PredicateRelationship.child_device_id == device_id
            ).all()
            
            if not predicates:
                # This is a root device
                return [current_chain]
            
            all_chains = []
            for pred_rel in predicates:
                parent_device = session.query(Device).get(pred_rel.parent_device_id)
                if parent_device:
                    new_chain = [parent_device.k_number] + current_chain
                    parent_chains = _build_chains(parent_device.id, new_chain, depth + 1)
                    all_chains.extend(parent_chains)
            
            return all_chains
        
        return _build_chains(device.id, [k_number], 0)


class AdverseEventOperations:
    """Operations for managing MAUDE adverse events."""
    
    @staticmethod
    def create_adverse_event(session: Session, **kwargs) -> AdverseEvent:
        """Create a new adverse event record."""
        event = AdverseEvent(**kwargs)
        
        try:
            session.add(event)
            session.flush()
            logger.debug(f"Created adverse event: {event.mdr_report_key}")
            return event
        except IntegrityError:
            session.rollback()
            logger.debug(f"Adverse event already exists: {kwargs.get('mdr_report_key')}")
            return session.query(AdverseEvent).filter(
                AdverseEvent.mdr_report_key == kwargs.get('mdr_report_key')
            ).first()
    
    @staticmethod
    def get_events_for_device(session: Session, k_number: str) -> List[AdverseEvent]:
        """Get all adverse events associated with a device."""
        device = DeviceOperations.get_device_by_k_number(session, k_number)
        if not device:
            return []
        
        return session.query(AdverseEvent).filter(
            AdverseEvent.device_id == device.id
        ).all()
    
    @staticmethod
    def get_events_by_product_code(session: Session, product_code: str) -> List[AdverseEvent]:
        """Get all adverse events for a product code."""
        return session.query(AdverseEvent).filter(
            AdverseEvent.product_code == product_code
        ).all()
    
    @staticmethod
    def update_hfe_classification(
        session: Session,
        mdr_report_key: str,
        classification: HFEClassification,
        confidence: float
    ) -> bool:
        """Update HFE classification for an adverse event."""
        try:
            event = session.query(AdverseEvent).filter(
                AdverseEvent.mdr_report_key == mdr_report_key
            ).first()
            
            if event:
                event.hfe_classification = classification
                event.hfe_confidence = confidence
                event.updated_at = datetime.utcnow()
                logger.debug(f"Updated HFE classification for {mdr_report_key}")
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating HFE classification: {e}")
            return False


class LDIOperations:
    """Operations for managing LDI scores."""
    
    @staticmethod
    def create_ldi_score(
        session: Session,
        device_k_number: str,
        ldi_score: float,
        semantic_distance: Optional[float] = None,
        parameter_difference: Optional[float] = None,
        chain_length: Optional[int] = None,
        **kwargs
    ) -> Optional[LDIScore]:
        """Create a new LDI score record."""
        device = DeviceOperations.get_device_by_k_number(session, device_k_number)
        if not device:
            logger.warning(f"Cannot create LDI score: device {device_k_number} not found")
            return None
        
        try:
            score = LDIScore(
                device_id=device.id,
                ldi_score=ldi_score,
                semantic_distance=semantic_distance,
                parameter_difference=parameter_difference,
                chain_length=chain_length,
                **kwargs
            )
            
            session.add(score)
            session.flush()
            logger.debug(f"Created LDI score for {device_k_number}: {ldi_score:.3f}")
            return score
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating LDI score: {e}")
            return None
    
    @staticmethod
    def get_latest_ldi_score(session: Session, k_number: str) -> Optional[LDIScore]:
        """Get the most recent LDI score for a device."""
        device = DeviceOperations.get_device_by_k_number(session, k_number)
        if not device:
            return None
        
        return session.query(LDIScore).filter(
            LDIScore.device_id == device.id
        ).order_by(LDIScore.calculation_date.desc()).first()
    
    @staticmethod
    def get_ldi_scores_by_range(
        session: Session,
        min_score: float,
        max_score: float,
        device_code: Optional[str] = None
    ) -> List[Tuple[Device, LDIScore]]:
        """Get devices with LDI scores in a specific range."""
        query = session.query(Device, LDIScore).join(
            LDIScore, Device.id == LDIScore.device_id
        ).filter(
            and_(
                LDIScore.ldi_score >= min_score,
                LDIScore.ldi_score <= max_score
            )
        )
        
        if device_code:
            query = query.filter(Device.device_code == device_code)
        
        return query.all()
    
    @staticmethod
    def calculate_ldi_percentiles(session: Session, device_code: Optional[str] = None) -> Dict[float, float]:
        """Calculate LDI score percentiles for a device family."""
        query = session.query(LDIScore.ldi_score).join(Device)
        
        if device_code:
            query = query.filter(Device.device_code == device_code)
        
        # Get latest scores for each device
        subquery = session.query(
            LDIScore.device_id,
            LDIScore.ldi_score,
            LDIScore.calculation_date
        ).order_by(
            LDIScore.device_id,
            LDIScore.calculation_date.desc()
        ).distinct(LDIScore.device_id).subquery()
        
        scores = session.query(subquery.c.ldi_score).all()
        score_values = [score[0] for score in scores if score[0] is not None]
        
        if not score_values:
            return {}
        
        score_values.sort()
        n = len(score_values)
        
        percentiles = {}
        for p in [10, 25, 50, 75, 90, 95, 99]:
            idx = int((p / 100) * (n - 1))
            percentiles[p] = score_values[idx]
        
        return percentiles


class AnalysisOperations:
    """Operations for managing analysis runs and validation."""
    
    @staticmethod
    def create_analysis_run(
        session: Session,
        run_name: str,
        run_type: str,
        parameters: Optional[Dict] = None
    ) -> AnalysisRun:
        """Create a new analysis run record."""
        run = AnalysisRun(
            run_name=run_name,
            run_type=run_type,
            parameters=parameters or {},
            start_time=datetime.utcnow(),
            status='running'
        )
        
        session.add(run)
        session.flush()
        logger.info(f"Created analysis run: {run_name} ({run_type})")
        return run
    
    @staticmethod
    def complete_analysis_run(
        session: Session,
        run_id: int,
        success_count: int,
        error_count: int,
        results_summary: Optional[Dict] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Mark an analysis run as completed."""
        try:
            run = session.query(AnalysisRun).get(run_id)
            if run:
                run.end_time = datetime.utcnow()
                run.duration_seconds = (run.end_time - run.start_time).total_seconds()
                run.success_count = success_count
                run.error_count = error_count
                run.results_summary = results_summary
                run.status = 'completed' if error_count == 0 else 'failed'
                if error_message:
                    run.error_message = error_message
                
                logger.info(f"Completed analysis run {run.run_name}: {success_count} success, {error_count} errors")
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error completing analysis run: {e}")
            return False
    
    @staticmethod
    def create_validation_result(
        session: Session,
        validation_type: str,
        device_count: int,
        outcome_count: int,
        correlation_coefficient: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """Create a validation result record."""
        result = ValidationResult(
            validation_type=validation_type,
            device_count=device_count,
            outcome_count=outcome_count,
            correlation_coefficient=correlation_coefficient,
            **kwargs
        )
        
        session.add(result)
        session.flush()
        logger.info(f"Created validation result: {validation_type}")
        return result


def get_database_statistics() -> Dict[str, int]:
    """Get basic database statistics."""
    with get_db_session() as session:
        stats = {}
        
        # Count records in each main table
        stats['devices'] = session.query(Device).count()
        stats['predicate_relationships'] = session.query(PredicateRelationship).count()
        stats['adverse_events'] = session.query(AdverseEvent).count()
        stats['recalls'] = session.query(Recall).count()
        stats['ldi_scores'] = session.query(LDIScore).count()
        stats['network_nodes'] = session.query(NetworkNode).count()
        stats['analysis_runs'] = session.query(AnalysisRun).count()
        stats['validation_results'] = session.query(ValidationResult).count()
        
        logger.info(f"Database statistics: {stats}")
        return stats


def main():
    """Test database operations."""
    logging.basicConfig(level=logging.INFO)
    
    # Test basic operations
    with get_db_session() as session:
        # Create a test device
        device = DeviceOperations.create_device(
            session,
            k_number="K123456",
            device_name="Test Hip Implant",
            applicant="Test Company",
            device_code="KWA"
        )
        
        print(f"Created device: {device}")
        
        # Get database statistics
        stats = get_database_statistics()
        print(f"Database statistics: {stats}")


if __name__ == "__main__":
    main()