"""
Lineage Drift Index (LDI) Calculator

This module implements the core LDI calculation engine that combines
semantic distance, parameter difference, and chain length metrics
to produce a comprehensive risk assessment score.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from .semantic_calculator import SemanticDistanceCalculator
from .parameter_calculator import ParameterDifferenceCalculator  
from .chain_calculator import ChainLengthCalculator
from ..database.models import Device, LDIScore
from ..database.operations import DeviceOperations, LDIOperations

logger = logging.getLogger(__name__)


class LDICalculator:
    """Main LDI calculation engine combining all three components."""
    
    def __init__(
        self,
        semantic_weight: float = 0.4,
        parameter_weight: float = 0.4,
        chain_weight: float = 0.2,
        semantic_method: str = "hybrid",
        normalization_method: str = "standard",
        max_chain_depth: int = 15
    ):
        """
        Initialize LDI calculator with component weights.
        
        Args:
            semantic_weight: Weight for semantic distance component
            parameter_weight: Weight for parameter difference component  
            chain_weight: Weight for chain length component
            semantic_method: Method for semantic calculation
            normalization_method: Parameter normalization method
            max_chain_depth: Maximum chain traversal depth
        """
        # Validate weights
        total_weight = semantic_weight + parameter_weight + chain_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.semantic_weight = semantic_weight
        self.parameter_weight = parameter_weight
        self.chain_weight = chain_weight
        
        # Initialize component calculators
        self.semantic_calculator = SemanticDistanceCalculator(model_type=semantic_method)
        self.parameter_calculator = ParameterDifferenceCalculator(normalization_method=normalization_method)
        self.chain_calculator = ChainLengthCalculator(max_depth=max_chain_depth)
        
        # Cache for efficiency
        self._predicate_graph_built = False
        
        logger.info(f"Initialized LDI calculator with weights: semantic={semantic_weight}, parameter={parameter_weight}, chain={chain_weight}")
    
    def calculate_ldi_score(
        self,
        session: Session,
        device_k_number: str,
        force_recalculate: bool = False
    ) -> Tuple[float, Dict[str, any]]:
        """
        Calculate LDI score for a specific device.
        
        Args:
            session: Database session
            device_k_number: K-number of the device
            force_recalculate: Whether to force recalculation if score exists
            
        Returns:
            Tuple of (ldi_score, detailed_metrics)
        """
        logger.info(f"Calculating LDI score for device: {device_k_number}")
        
        # Check if score already exists
        if not force_recalculate:
            existing_score = LDIOperations.get_latest_ldi_score(session, device_k_number)
            if existing_score:
                logger.info(f"Using existing LDI score: {existing_score.ldi_score:.3f}")
                return existing_score.ldi_score, self._extract_metrics_from_score(existing_score)
        
        # Get device information
        device = DeviceOperations.get_device_by_k_number(session, device_k_number)
        if not device:
            raise ValueError(f"Device not found: {device_k_number}")
        
        # Get predicate devices
        predicates = DeviceOperations.get_device_predicates(session, device_k_number)
        if not predicates:
            logger.warning(f"No predicates found for device {device_k_number}, using minimal LDI score")
            return 0.1, {'reason': 'no_predicates', 'components': {}}
        
        # For multiple predicates, use the primary one (first) or average
        primary_predicate = predicates[0]
        
        # Build predicate graph if not already built
        if not self._predicate_graph_built:
            self.chain_calculator.build_predicate_graph(session, device.device_code)
            self._predicate_graph_built = True
        
        # Calculate semantic distance
        semantic_distance, semantic_metrics = self._calculate_semantic_component(
            device, primary_predicate
        )
        
        # Calculate parameter difference
        parameter_difference, parameter_metrics = self._calculate_parameter_component(
            device, primary_predicate
        )
        
        # Calculate chain length
        chain_length, chain_metrics = self._calculate_chain_component(
            device_k_number
        )
        
        # Combine components into final LDI score
        ldi_score = (
            self.semantic_weight * semantic_distance +
            self.parameter_weight * parameter_difference +
            self.chain_weight * chain_length
        )
        
        # Prepare detailed metrics
        detailed_metrics = {
            'ldi_score': ldi_score,
            'components': {
                'semantic_distance': semantic_distance,
                'parameter_difference': parameter_difference,
                'chain_length': chain_length
            },
            'weights': {
                'semantic_weight': self.semantic_weight,
                'parameter_weight': self.parameter_weight,
                'chain_weight': self.chain_weight
            },
            'detailed_metrics': {
                'semantic': semantic_metrics,
                'parameter': parameter_metrics,
                'chain': chain_metrics
            },
            'primary_predicate': primary_predicate.k_number,
            'total_predicates': len(predicates)
        }
        
        logger.info(f"LDI calculation complete: {ldi_score:.3f}")
        return ldi_score, detailed_metrics
    
    def _calculate_semantic_component(self, device: Device, predicate: Device) -> Tuple[float, Dict]:
        """Calculate semantic distance component."""
        device_desc = self._get_device_description(device)
        predicate_desc = self._get_device_description(predicate)
        
        distance, metrics = self.semantic_calculator.calculate_semantic_distance(
            device_desc, predicate_desc
        )
        
        return distance, metrics
    
    def _calculate_parameter_component(self, device: Device, predicate: Device) -> Tuple[float, Dict]:
        """Calculate parameter difference component."""
        device_params = self.parameter_calculator.extract_parameters(
            self._get_device_description(device),
            device.technical_parameters or {}
        )
        
        predicate_params = self.parameter_calculator.extract_parameters(
            self._get_device_description(predicate),
            predicate.technical_parameters or {}
        )
        
        # Normalize parameters together
        normalized_params = self.parameter_calculator.normalize_parameters([device_params, predicate_params])
        
        if len(normalized_params) >= 2:
            difference, metrics = self.parameter_calculator.calculate_parameter_difference(
                normalized_params[0], normalized_params[1]
            )
        else:
            difference, metrics = 0.0, {}
        
        return difference, metrics
    
    def _calculate_chain_component(self, device_k_number: str) -> Tuple[float, Dict]:
        """Calculate chain length component."""
        chain_length, metrics = self.chain_calculator.calculate_chain_metrics(device_k_number)
        
        # Normalize chain length to 0-1 range using sigmoid function
        normalized_chain = self._normalize_chain_length(chain_length)
        
        return normalized_chain, metrics
    
    def _normalize_chain_length(self, chain_length: float) -> float:
        """
        Normalize chain length to 0-1 range using sigmoid transformation.
        
        Args:
            chain_length: Raw chain length value
            
        Returns:
            Normalized chain length (0-1)
        """
        # Use sigmoid function to normalize: f(x) = 1 / (1 + e^(-k*(x-x0)))
        # Where k controls steepness and x0 is the midpoint
        k = 0.5  # Steepness parameter
        x0 = 5.0  # Midpoint (chain length that gives 0.5 score)
        
        normalized = 1.0 / (1.0 + np.exp(-k * (chain_length - x0)))
        return normalized
    
    def _get_device_description(self, device: Device) -> str:
        """Get comprehensive device description for analysis."""
        description_parts = []
        
        if device.device_name:
            description_parts.append(device.device_name)
        
        # Add technical parameters as text
        if device.technical_parameters:
            for key, value in device.technical_parameters.items():
                if value:
                    description_parts.append(f"{key}: {value}")
        
        return " ".join(description_parts)
    
    def _extract_metrics_from_score(self, ldi_score: LDIScore) -> Dict[str, any]:
        """Extract metrics from existing LDI score record."""
        return {
            'ldi_score': ldi_score.ldi_score,
            'components': {
                'semantic_distance': ldi_score.semantic_distance,
                'parameter_difference': ldi_score.parameter_difference,
                'chain_length': ldi_score.chain_length
            },
            'weights': {
                'semantic_weight': ldi_score.semantic_weight,
                'parameter_weight': ldi_score.parameter_weight,
                'chain_weight': ldi_score.chain_weight
            },
            'calculation_date': ldi_score.calculation_date,
            'calculation_version': ldi_score.calculation_version
        }
    
    def store_ldi_score(
        self,
        session: Session,
        device_k_number: str,
        ldi_score: float,
        detailed_metrics: Dict[str, any]
    ) -> LDIScore:
        """
        Store LDI score in database.
        
        Args:
            session: Database session
            device_k_number: K-number of the device
            ldi_score: Calculated LDI score
            detailed_metrics: Detailed calculation metrics
            
        Returns:
            Created LDIScore object
        """
        components = detailed_metrics.get('components', {})
        
        score_record = LDIOperations.create_ldi_score(
            session,
            device_k_number=device_k_number,
            ldi_score=ldi_score,
            semantic_distance=components.get('semantic_distance'),
            parameter_difference=components.get('parameter_difference'),
            chain_length=components.get('chain_length'),
            semantic_weight=self.semantic_weight,
            parameter_weight=self.parameter_weight,
            chain_weight=self.chain_weight,
            calculation_version="1.0"
        )
        
        logger.info(f"Stored LDI score for device {device_k_number}: {ldi_score:.3f}")
        return score_record
    
    def batch_calculate_ldi_scores(
        self,
        session: Session,
        device_k_numbers: List[str],
        store_results: bool = True,
        force_recalculate: bool = False
    ) -> List[Tuple[str, float, Dict[str, any]]]:
        """
        Calculate LDI scores for multiple devices.
        
        Args:
            session: Database session
            device_k_numbers: List of K-numbers
            store_results: Whether to store results in database
            force_recalculate: Whether to force recalculation
            
        Returns:
            List of (k_number, ldi_score, metrics) tuples
        """
        logger.info(f"Batch calculating LDI scores for {len(device_k_numbers)} devices")
        
        results = []
        
        # Build predicate graph once for efficiency
        if not self._predicate_graph_built:
            # Get device code from first device
            first_device = DeviceOperations.get_device_by_k_number(session, device_k_numbers[0])
            if first_device:
                self.chain_calculator.build_predicate_graph(session, first_device.device_code)
                self._predicate_graph_built = True
        
        for k_number in device_k_numbers:
            try:
                ldi_score, metrics = self.calculate_ldi_score(session, k_number, force_recalculate)
                
                if store_results:
                    self.store_ldi_score(session, k_number, ldi_score, metrics)
                
                results.append((k_number, ldi_score, metrics))
                
            except Exception as e:
                logger.error(f"Error calculating LDI for device {k_number}: {e}")
                # Store error result
                results.append((k_number, 0.0, {'error': str(e)}))
        
        logger.info(f"Completed batch LDI calculation: {len(results)} results")
        return results
    
    def analyze_ldi_distribution(self, ldi_scores: List[float]) -> Dict[str, any]:
        """
        Analyze the distribution of LDI scores.
        
        Args:
            ldi_scores: List of LDI scores
            
        Returns:
            Distribution analysis metrics
        """
        if not ldi_scores:
            return {}
        
        scores = np.array(ldi_scores)
        
        analysis = {
            'count': len(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'percentiles': {
                '10': np.percentile(scores, 10),
                '25': np.percentile(scores, 25),
                '75': np.percentile(scores, 75),
                '90': np.percentile(scores, 90),
                '95': np.percentile(scores, 95),
                '99': np.percentile(scores, 99)
            },
            'risk_categories': {
                'low_risk': np.sum(scores < 0.3) / len(scores),
                'medium_risk': np.sum((scores >= 0.3) & (scores < 0.7)) / len(scores),
                'high_risk': np.sum(scores >= 0.7) / len(scores)
            }
        }
        
        return analysis


def main():
    """Test LDI calculation with sample data."""
    # This would normally connect to a database
    # For testing, we'll create a mock calculator
    
    calculator = LDICalculator(
        semantic_weight=0.4,
        parameter_weight=0.4,
        chain_weight=0.2
    )
    
    print("LDI Calculator initialized")
    print(f"Weights: Semantic={calculator.semantic_weight}, Parameter={calculator.parameter_weight}, Chain={calculator.chain_weight}")
    
    # Test distribution analysis
    test_scores = [0.1, 0.25, 0.4, 0.35, 0.6, 0.75, 0.8, 0.3, 0.45, 0.9]
    distribution = calculator.analyze_ldi_distribution(test_scores)
    
    print("\nLDI Score Distribution Analysis:")
    print(f"Mean: {distribution['mean']:.3f}")
    print(f"Median: {distribution['median']:.3f}")
    print(f"Standard Deviation: {distribution['std']:.3f}")
    print(f"Risk Categories:")
    print(f"  Low Risk (<0.3): {distribution['risk_categories']['low_risk']:.1%}")
    print(f"  Medium Risk (0.3-0.7): {distribution['risk_categories']['medium_risk']:.1%}")
    print(f"  High Risk (>0.7): {distribution['risk_categories']['high_risk']:.1%}")


if __name__ == "__main__":
    main()