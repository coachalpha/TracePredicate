"""
Weight Optimization System for LDI Components

This module implements mathematical optimization to find optimal weights
for LDI components that maximize correlation with real-world risk outcomes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy import optimize
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy.orm import Session

from ..database.operations import LDIOperations, AdverseEventOperations
from ..ldi_engine.ldi_calculator import LDICalculator

logger = logging.getLogger(__name__)


class WeightOptimizer:
    """Optimize LDI component weights using real-world outcome data."""
    
    def __init__(
        self,
        optimization_method: str = "scipy",
        correlation_method: str = "spearman",
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Initialize weight optimizer.
        
        Args:
            optimization_method: Optimization method ('scipy', 'grid_search', 'genetic')
            correlation_method: Correlation method ('spearman', 'pearson')
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.optimization_method = optimization_method
        self.correlation_method = correlation_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Optimization history
        self.optimization_history = []
        self.best_weights = None
        self.best_correlation = -1.0
        
        logger.info(f"Initialized weight optimizer: method={optimization_method}, correlation={correlation_method}")
    
    def objective_function(
        self,
        weights: np.ndarray,
        component_matrix: np.ndarray,
        outcome_vector: np.ndarray,
        regularization_lambda: float = 0.01
    ) -> float:
        """
        Objective function to minimize (negative correlation + regularization).
        
        Args:
            weights: Array of weights [semantic_weight, parameter_weight, chain_weight]
            component_matrix: Matrix of component scores (n_devices x 3)
            outcome_vector: Vector of outcome values (n_devices,)
            regularization_lambda: L2 regularization parameter
            
        Returns:
            Objective value to minimize
        """
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate LDI scores using current weights
        ldi_scores = np.dot(component_matrix, weights)
        
        # Calculate correlation with outcomes
        if self.correlation_method == "spearman":
            correlation, p_value = spearmanr(ldi_scores, outcome_vector)
        else:  # pearson
            correlation, p_value = pearsonr(ldi_scores, outcome_vector)
        
        # Handle NaN correlations
        if np.isnan(correlation):
            correlation = 0.0
        
        # Objective: maximize correlation (minimize negative correlation)
        # Add L2 regularization to prefer balanced weights
        objective = -correlation + regularization_lambda * np.sum(weights ** 2)
        
        # Store optimization history
        self.optimization_history.append({
            'weights': weights.copy(),
            'correlation': correlation,
            'p_value': p_value,
            'objective': objective
        })
        
        return objective
    
    def optimize_weights_scipy(
        self,
        component_matrix: np.ndarray,
        outcome_vector: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
        regularization_lambda: float = 0.01
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Optimize weights using scipy optimization.
        
        Args:
            component_matrix: Matrix of component scores (n_devices x 3)
            outcome_vector: Vector of outcome values
            initial_weights: Initial weight values
            regularization_lambda: Regularization parameter
            
        Returns:
            Tuple of (optimal_weights, optimization_results)
        """
        if initial_weights is None:
            initial_weights = np.array([0.33, 0.33, 0.34])  # Approximately equal weights
        
        # Define constraints: weights must sum to 1 and be non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
        ]
        
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # Each weight between 0 and 1
        
        # Reset optimization history
        self.optimization_history = []
        
        # Run optimization
        result = optimize.minimize(
            fun=self.objective_function,
            x0=initial_weights,
            args=(component_matrix, outcome_vector, regularization_lambda),
            method='SLSQP',  # Sequential Least SQuares Programming
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        # Normalize final weights
        optimal_weights = result.x / np.sum(result.x)
        
        # Calculate final metrics
        final_ldi_scores = np.dot(component_matrix, optimal_weights)
        if self.correlation_method == "spearman":
            final_correlation, final_p_value = spearmanr(final_ldi_scores, outcome_vector)
        else:
            final_correlation, final_p_value = pearsonr(final_ldi_scores, outcome_vector)
        
        optimization_results = {
            'optimal_weights': optimal_weights,
            'final_correlation': final_correlation,
            'final_p_value': final_p_value,
            'optimization_success': result.success,
            'iterations': result.nit,
            'optimization_message': result.message,
            'optimization_history': self.optimization_history,
            'initial_weights': initial_weights,
            'regularization_lambda': regularization_lambda
        }
        
        # Update best weights
        self.best_weights = optimal_weights
        self.best_correlation = final_correlation
        
        logger.info(f"Scipy optimization completed: correlation={final_correlation:.4f}, weights={optimal_weights}")
        
        return optimal_weights, optimization_results
    
    def optimize_weights_grid_search(
        self,
        component_matrix: np.ndarray,
        outcome_vector: np.ndarray,
        grid_resolution: int = 20
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Optimize weights using grid search.
        
        Args:
            component_matrix: Matrix of component scores
            outcome_vector: Vector of outcome values
            grid_resolution: Number of points per dimension
            
        Returns:
            Tuple of (optimal_weights, optimization_results)
        """
        logger.info(f"Starting grid search optimization with resolution {grid_resolution}")
        
        best_correlation = -1.0
        best_weights = None
        grid_results = []
        
        # Generate weight combinations that sum to 1
        step_size = 1.0 / (grid_resolution - 1)
        
        for i in range(grid_resolution):
            w1 = i * step_size
            for j in range(grid_resolution - i):
                w2 = j * step_size
                w3 = 1.0 - w1 - w2
                
                if w3 < 0 or w3 > 1:
                    continue
                
                weights = np.array([w1, w2, w3])
                
                # Calculate LDI scores
                ldi_scores = np.dot(component_matrix, weights)
                
                # Calculate correlation
                if self.correlation_method == "spearman":
                    correlation, p_value = spearmanr(ldi_scores, outcome_vector)
                else:
                    correlation, p_value = pearsonr(ldi_scores, outcome_vector)
                
                if np.isnan(correlation):
                    correlation = 0.0
                
                grid_results.append({
                    'weights': weights,
                    'correlation': correlation,
                    'p_value': p_value
                })
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_weights = weights
        
        optimization_results = {
            'optimal_weights': best_weights,
            'final_correlation': best_correlation,
            'grid_results': grid_results,
            'grid_resolution': grid_resolution,
            'total_combinations': len(grid_results)
        }
        
        self.best_weights = best_weights
        self.best_correlation = best_correlation
        
        logger.info(f"Grid search completed: correlation={best_correlation:.4f}, weights={best_weights}")
        
        return best_weights, optimization_results
    
    def cross_validate_weights(
        self,
        component_matrix: np.ndarray,
        outcome_vector: np.ndarray,
        n_folds: int = 5,
        random_state: int = 42
    ) -> Dict[str, any]:
        """
        Cross-validate weight optimization to assess stability.
        
        Args:
            component_matrix: Matrix of component scores
            outcome_vector: Vector of outcome values
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {n_folds}-fold cross-validation")
        
        np.random.seed(random_state)
        n_samples = len(outcome_vector)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds
        
        cv_results = {
            'fold_weights': [],
            'fold_correlations': [],
            'fold_p_values': [],
            'mean_weights': None,
            'std_weights': None,
            'mean_correlation': 0.0,
            'std_correlation': 0.0
        }
        
        for fold in range(n_folds):
            # Create train/test split
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            train_components = component_matrix[train_indices]
            train_outcomes = outcome_vector[train_indices]
            test_components = component_matrix[test_indices]
            test_outcomes = outcome_vector[test_indices]
            
            # Optimize weights on training data
            if self.optimization_method == "scipy":
                fold_weights, _ = self.optimize_weights_scipy(train_components, train_outcomes)
            else:  # grid_search
                fold_weights, _ = self.optimize_weights_grid_search(train_components, train_outcomes)
            
            # Evaluate on test data
            test_ldi_scores = np.dot(test_components, fold_weights)
            if self.correlation_method == "spearman":
                test_correlation, test_p_value = spearmanr(test_ldi_scores, test_outcomes)
            else:
                test_correlation, test_p_value = pearsonr(test_ldi_scores, test_outcomes)
            
            if np.isnan(test_correlation):
                test_correlation = 0.0
            
            cv_results['fold_weights'].append(fold_weights)
            cv_results['fold_correlations'].append(test_correlation)
            cv_results['fold_p_values'].append(test_p_value)
        
        # Calculate summary statistics
        all_weights = np.array(cv_results['fold_weights'])
        cv_results['mean_weights'] = np.mean(all_weights, axis=0)
        cv_results['std_weights'] = np.std(all_weights, axis=0)
        cv_results['mean_correlation'] = np.mean(cv_results['fold_correlations'])
        cv_results['std_correlation'] = np.std(cv_results['fold_correlations'])
        
        logger.info(f"Cross-validation completed: mean_correlation={cv_results['mean_correlation']:.4f}±{cv_results['std_correlation']:.4f}")
        
        return cv_results
    
    def sensitivity_analysis(
        self,
        component_matrix: np.ndarray,
        outcome_vector: np.ndarray,
        base_weights: np.ndarray,
        perturbation_range: float = 0.1,
        n_perturbations: int = 100
    ) -> Dict[str, any]:
        """
        Perform sensitivity analysis on optimal weights.
        
        Args:
            component_matrix: Matrix of component scores
            outcome_vector: Vector of outcome values
            base_weights: Base weights to perturb
            perturbation_range: Range of perturbation (±)
            n_perturbations: Number of perturbations to test
            
        Returns:
            Sensitivity analysis results
        """
        logger.info(f"Starting sensitivity analysis with {n_perturbations} perturbations")
        
        correlations = []
        weight_variations = []
        
        for _ in range(n_perturbations):
            # Generate random perturbations
            perturbation = np.random.uniform(-perturbation_range, perturbation_range, 3)
            perturbed_weights = base_weights + perturbation
            
            # Ensure non-negative and normalize
            perturbed_weights = np.maximum(perturbed_weights, 0.01)  # Minimum weight
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights)
            
            # Calculate correlation with perturbed weights
            ldi_scores = np.dot(component_matrix, perturbed_weights)
            if self.correlation_method == "spearman":
                correlation, _ = spearmanr(ldi_scores, outcome_vector)
            else:
                correlation, _ = pearsonr(ldi_scores, outcome_vector)
            
            if not np.isnan(correlation):
                correlations.append(correlation)
                weight_variations.append(perturbed_weights)
        
        correlations = np.array(correlations)
        weight_variations = np.array(weight_variations)
        
        # Calculate base correlation
        base_ldi_scores = np.dot(component_matrix, base_weights)
        if self.correlation_method == "spearman":
            base_correlation, _ = spearmanr(base_ldi_scores, outcome_vector)
        else:
            base_correlation, _ = pearsonr(base_ldi_scores, outcome_vector)
        
        sensitivity_results = {
            'base_weights': base_weights,
            'base_correlation': base_correlation,
            'perturbed_correlations': correlations,
            'weight_variations': weight_variations,
            'correlation_statistics': {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'min': np.min(correlations),
                'max': np.max(correlations),
                'percentiles': {
                    '5': np.percentile(correlations, 5),
                    '25': np.percentile(correlations, 25),
                    '75': np.percentile(correlations, 75),
                    '95': np.percentile(correlations, 95)
                }
            },
            'stability_measure': np.std(correlations),  # Lower is more stable
            'perturbation_range': perturbation_range,
            'n_perturbations': len(correlations)
        }
        
        logger.info(f"Sensitivity analysis completed: stability={sensitivity_results['stability_measure']:.4f}")
        
        return sensitivity_results
    
    def optimize_weights_from_database(
        self,
        session: Session,
        device_code: str = "KWA",
        outcome_type: str = "adverse_events",
        min_devices: int = 50
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Optimize weights using data from database.
        
        Args:
            session: Database session
            device_code: Device code to analyze
            outcome_type: Type of outcome ('adverse_events', 'recalls')
            min_devices: Minimum number of devices required
            
        Returns:
            Tuple of (optimal_weights, optimization_results)
        """
        logger.info(f"Optimizing weights from database: device_code={device_code}, outcome={outcome_type}")
        
        # Get LDI scores from database
        ldi_data = session.query(LDIScore).join(Device).filter(Device.device_code == device_code).all()
        
        if len(ldi_data) < min_devices:
            raise ValueError(f"Insufficient data: {len(ldi_data)} devices, minimum {min_devices} required")
        
        # Prepare component matrix
        component_matrix = []
        device_k_numbers = []
        
        for ldi_score in ldi_data:
            components = [
                ldi_score.semantic_distance or 0.0,
                ldi_score.parameter_difference or 0.0,
                ldi_score.chain_length or 0.0
            ]
            component_matrix.append(components)
            device_k_numbers.append(ldi_score.device.k_number)
        
        component_matrix = np.array(component_matrix)
        
        # Get outcome data
        if outcome_type == "adverse_events":
            outcomes = self._get_adverse_event_outcomes(session, device_k_numbers)
        else:  # recalls
            outcomes = self._get_recall_outcomes(session, device_k_numbers)
        
        outcome_vector = np.array([outcomes.get(k_num, 0.0) for k_num in device_k_numbers])
        
        # Run optimization
        if self.optimization_method == "scipy":
            optimal_weights, results = self.optimize_weights_scipy(component_matrix, outcome_vector)
        else:
            optimal_weights, results = self.optimize_weights_grid_search(component_matrix, outcome_vector)
        
        # Add database-specific information
        results['device_code'] = device_code
        results['outcome_type'] = outcome_type
        results['n_devices'] = len(ldi_data)
        results['device_k_numbers'] = device_k_numbers
        
        return optimal_weights, results
    
    def _get_adverse_event_outcomes(self, session: Session, device_k_numbers: List[str]) -> Dict[str, float]:
        """Get adverse event rates for devices."""
        outcomes = {}
        
        for k_number in device_k_numbers:
            events = AdverseEventOperations.get_events_for_device(session, k_number)
            # Normalize by time since approval or use raw count
            outcomes[k_number] = len(events)
        
        return outcomes
    
    def _get_recall_outcomes(self, session: Session, device_k_numbers: List[str]) -> Dict[str, float]:
        """Get recall indicators for devices."""
        outcomes = {}
        
        for k_number in device_k_numbers:
            # This would need to be implemented based on recall associations
            # For now, using placeholder
            outcomes[k_number] = 0.0
        
        return outcomes


def main():
    """Test weight optimization with synthetic data."""
    optimizer = WeightOptimizer(optimization_method="scipy")
    
    # Generate synthetic data
    np.random.seed(42)
    n_devices = 100
    
    # Component scores (semantic, parameter, chain)
    component_matrix = np.random.rand(n_devices, 3)
    
    # Synthetic outcome that correlates with weighted combination
    true_weights = np.array([0.5, 0.3, 0.2])
    true_ldi_scores = np.dot(component_matrix, true_weights)
    outcome_vector = true_ldi_scores + 0.1 * np.random.randn(n_devices)
    
    # Optimize weights
    optimal_weights, results = optimizer.optimize_weights_scipy(component_matrix, outcome_vector)
    
    print("Weight Optimization Results:")
    print(f"True weights: {true_weights}")
    print(f"Optimal weights: {optimal_weights}")
    print(f"Final correlation: {results['final_correlation']:.4f}")
    print(f"Optimization success: {results['optimization_success']}")
    
    # Cross-validation
    cv_results = optimizer.cross_validate_weights(component_matrix, outcome_vector)
    print(f"\nCross-validation:")
    print(f"Mean correlation: {cv_results['mean_correlation']:.4f}±{cv_results['std_correlation']:.4f}")
    print(f"Mean weights: {cv_results['mean_weights']}")
    
    # Sensitivity analysis
    sensitivity = optimizer.sensitivity_analysis(component_matrix, outcome_vector, optimal_weights)
    print(f"\nSensitivity Analysis:")
    print(f"Base correlation: {sensitivity['base_correlation']:.4f}")
    print(f"Stability measure: {sensitivity['stability_measure']:.4f}")


if __name__ == "__main__":
    main()