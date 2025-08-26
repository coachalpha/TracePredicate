"""
Parameter Difference Calculator for LDI Engine

This module implements technical parameter difference calculations to measure
quantitative changes between medical devices and their predicates.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class ParameterDifferenceCalculator:
    """Calculate technical parameter differences between medical devices."""
    
    def __init__(self, normalization_method: str = "standard"):
        """
        Initialize parameter difference calculator.
        
        Args:
            normalization_method: Method for normalizing parameters 
                                 ('standard', 'minmax', 'robust')
        """
        self.normalization_method = normalization_method
        self.scaler = self._create_scaler()
        
        # Medical device parameter categories and their importance weights
        self.parameter_weights = {
            'dimension': 1.0,      # Size, diameter, length
            'material': 0.8,       # Material composition
            'coating': 0.7,        # Surface treatments
            'design': 0.9,         # Structural design features
            'mechanical': 1.0,     # Mechanical properties
            'biocompatibility': 0.6  # Bio-compatibility measures
        }
        
        logger.info(f"Initialized parameter calculator with {normalization_method} normalization")
    
    def _create_scaler(self):
        """Create the appropriate scaler based on normalization method."""
        if self.normalization_method == "standard":
            return StandardScaler()
        elif self.normalization_method == "minmax":
            return MinMaxScaler()
        elif self.normalization_method == "robust":
            return RobustScaler()
        else:
            logger.warning(f"Unknown normalization method: {self.normalization_method}, using standard")
            return StandardScaler()
    
    def extract_parameters(self, device_description: str, technical_specs: Dict = None) -> Dict[str, Any]:
        """
        Extract technical parameters from device description and specifications.
        
        Args:
            device_description: Text description of the device
            technical_specs: Dictionary of technical specifications
            
        Returns:
            Dictionary of extracted parameters organized by category
        """
        parameters = {
            'dimension': {},
            'material': {},
            'coating': {},
            'design': {},
            'mechanical': {},
            'biocompatibility': {}
        }
        
        # Extract from text description
        if device_description:
            text_params = self._extract_from_text(device_description)
            for category, params in text_params.items():
                parameters[category].update(params)
        
        # Add technical specifications
        if technical_specs:
            spec_params = self._categorize_specs(technical_specs)
            for category, params in spec_params.items():
                parameters[category].update(params)
        
        return parameters
    
    def _extract_from_text(self, text: str) -> Dict[str, Dict[str, float]]:
        """Extract parameters from device description text."""
        text = text.lower()
        parameters = {
            'dimension': {},
            'material': {},
            'coating': {},
            'design': {},
            'mechanical': {},
            'biocompatibility': {}
        }
        
        # Dimension extraction
        # Extract diameters, lengths, sizes
        diameter_matches = re.findall(r'(\d+(?:\.\d+)?)\s*mm.*?diameter', text)
        if diameter_matches:
            parameters['dimension']['diameter_mm'] = float(diameter_matches[0])
        
        length_matches = re.findall(r'(\d+(?:\.\d+)?)\s*mm.*?length', text)
        if length_matches:
            parameters['dimension']['length_mm'] = float(length_matches[0])
        
        # Size categories (convert to numeric)
        size_mapping = {'small': 1, 'medium': 2, 'large': 3, 'extra-large': 4}
        for size_name, size_value in size_mapping.items():
            if size_name in text:
                parameters['dimension']['size_category'] = size_value
                break
        
        # Material composition (binary indicators)
        materials = {
            'titanium': ['titanium', 'ti-6al-4v', 'titanium alloy'],
            'ceramic': ['ceramic', 'alumina', 'zirconia', 'biolox'],
            'polyethylene': ['polyethylene', 'pe', 'uhmwpe'],
            'cobalt_chrome': ['cobalt chrome', 'cocr', 'cobalt chromium'],
            'stainless_steel': ['stainless steel', 'ss', '316l']
        }
        
        for material, terms in materials.items():
            parameters['material'][f'{material}_present'] = float(any(term in text for term in terms))
        
        # Coating features
        coatings = {
            'porous_coating': ['porous coating', 'porous', 'osteointegration'],
            'hydroxyapatite': ['hydroxyapatite', 'ha coating', 'calcium phosphate'],
            'plasma_spray': ['plasma spray', 'plasma sprayed'],
            'anodized': ['anodized', 'anodization']
        }
        
        for coating, terms in coatings.items():
            parameters['coating'][coating] = float(any(term in text for term in terms))
        
        # Design features
        design_features = {
            'modular_design': ['modular', 'modular design', 'interchangeable'],
            'cemented': ['cemented', 'bone cement'],
            'cementless': ['cementless', 'press-fit'],
            'anatomical': ['anatomical', 'anatomically designed'],
            'straight_stem': ['straight stem', 'straight'],
            'curved_stem': ['curved stem', 'curved']
        }
        
        for feature, terms in design_features.items():
            parameters['design'][feature] = float(any(term in text for term in terms))
        
        # Mechanical properties (if mentioned)
        mechanical_patterns = {
            'tensile_strength': r'tensile strength[:\s]*(\d+(?:\.\d+)?)',
            'yield_strength': r'yield strength[:\s]*(\d+(?:\.\d+)?)',
            'elastic_modulus': r'elastic modulus[:\s]*(\d+(?:\.\d+)?)',
            'hardness': r'hardness[:\s]*(\d+(?:\.\d+)?)'
        }
        
        for prop, pattern in mechanical_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                parameters['mechanical'][prop] = float(matches[0])
        
        # Biocompatibility indicators
        bio_terms = {
            'biocompatible': ['biocompatible', 'biocompatibility'],
            'osseointegration': ['osseointegration', 'bone integration'],
            'wear_resistant': ['wear resistant', 'low wear'],
            'corrosion_resistant': ['corrosion resistant', 'corrosion resistance']
        }
        
        for term, indicators in bio_terms.items():
            parameters['biocompatibility'][term] = float(any(ind in text for ind in indicators))
        
        return parameters
    
    def _categorize_specs(self, specs: Dict) -> Dict[str, Dict[str, float]]:
        """Categorize technical specifications into parameter groups."""
        categorized = {
            'dimension': {},
            'material': {},
            'coating': {},
            'design': {},
            'mechanical': {},
            'biocompatibility': {}
        }
        
        for key, value in specs.items():
            key_lower = key.lower()
            
            # Convert value to numeric if possible
            if isinstance(value, str):
                # Extract numeric values from strings
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', value)
                if numeric_match:
                    numeric_value = float(numeric_match.group(1))
                else:
                    # Binary indicator for text values
                    numeric_value = 1.0 if value.strip() else 0.0
            else:
                numeric_value = float(value) if value is not None else 0.0
            
            # Categorize based on key name
            if any(term in key_lower for term in ['diameter', 'length', 'width', 'height', 'size']):
                categorized['dimension'][key] = numeric_value
            elif any(term in key_lower for term in ['material', 'alloy', 'composition']):
                categorized['material'][key] = numeric_value
            elif any(term in key_lower for term in ['coating', 'surface', 'finish']):
                categorized['coating'][key] = numeric_value
            elif any(term in key_lower for term in ['design', 'geometry', 'shape']):
                categorized['design'][key] = numeric_value
            elif any(term in key_lower for term in ['strength', 'modulus', 'hardness', 'mechanical']):
                categorized['mechanical'][key] = numeric_value
            elif any(term in key_lower for term in ['biocompat', 'toxic', 'bio']):
                categorized['biocompatibility'][key] = numeric_value
            else:
                # Default to design category
                categorized['design'][key] = numeric_value
        
        return categorized
    
    def normalize_parameters(self, parameters_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize parameters across multiple devices.
        
        Args:
            parameters_list: List of parameter dictionaries
            
        Returns:
            List of normalized parameter dictionaries
        """
        if not parameters_list:
            return []
        
        # Collect all unique parameter keys by category
        all_keys_by_category = {}
        for category in self.parameter_weights.keys():
            all_keys_by_category[category] = set()
            for params in parameters_list:
                all_keys_by_category[category].update(params.get(category, {}).keys())
        
        # Create feature matrix for each category
        normalized_params_list = []
        
        for category, weight in self.parameter_weights.items():
            if not all_keys_by_category[category]:
                continue
            
            keys = sorted(all_keys_by_category[category])
            
            # Create matrix
            matrix = []
            for params in parameters_list:
                row = [params.get(category, {}).get(key, 0.0) for key in keys]
                matrix.append(row)
            
            matrix = np.array(matrix)
            
            # Normalize
            if matrix.size > 0 and matrix.std() > 1e-6:  # Avoid division by zero
                normalized_matrix = self.scaler.fit_transform(matrix)
            else:
                normalized_matrix = matrix
            
            # Update parameters
            for i, params in enumerate(parameters_list):
                if i >= len(normalized_params_list):
                    normalized_params_list.append({cat: {} for cat in self.parameter_weights.keys()})
                
                for j, key in enumerate(keys):
                    normalized_params_list[i][category][key] = normalized_matrix[i, j]
        
        return normalized_params_list
    
    def calculate_category_difference(
        self, 
        params1: Dict[str, float], 
        params2: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate difference between parameter categories.
        
        Args:
            params1: Parameters from first device
            params2: Parameters from second device
            
        Returns:
            Tuple of (category_difference, detailed_differences)
        """
        if not params1 and not params2:
            return 0.0, {}
        
        # Get all unique parameter keys
        all_keys = set(params1.keys()) | set(params2.keys())
        
        if not all_keys:
            return 0.0, {}
        
        detailed_diffs = {}
        squared_diffs = []
        
        for key in all_keys:
            val1 = params1.get(key, 0.0)
            val2 = params2.get(key, 0.0)
            
            # Calculate absolute difference
            diff = abs(val1 - val2)
            detailed_diffs[key] = diff
            squared_diffs.append(diff ** 2)
        
        # Calculate Euclidean distance
        euclidean_distance = np.sqrt(sum(squared_diffs)) / len(squared_diffs) if squared_diffs else 0.0
        
        return euclidean_distance, detailed_diffs
    
    def calculate_parameter_difference(
        self, 
        device_params: Dict[str, Any], 
        predicate_params: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate overall parameter difference between device and predicate.
        
        Args:
            device_params: Parameters from current device
            predicate_params: Parameters from predicate device
            
        Returns:
            Tuple of (overall_difference, detailed_metrics)
        """
        metrics = {
            'category_differences': {},
            'weighted_differences': {},
            'total_parameters_compared': 0
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Calculate differences for each category
        for category, weight in self.parameter_weights.items():
            dev_cat_params = device_params.get(category, {})
            pred_cat_params = predicate_params.get(category, {})
            
            if not dev_cat_params and not pred_cat_params:
                continue
            
            cat_diff, detailed_diffs = self.calculate_category_difference(dev_cat_params, pred_cat_params)
            
            metrics['category_differences'][category] = {
                'difference': cat_diff,
                'detailed': detailed_diffs,
                'parameter_count': len(set(dev_cat_params.keys()) | set(pred_cat_params.keys()))
            }
            
            # Apply weight
            weighted_diff = cat_diff * weight
            metrics['weighted_differences'][category] = weighted_diff
            
            weighted_sum += weighted_diff
            total_weight += weight
            metrics['total_parameters_compared'] += len(detailed_diffs)
        
        # Calculate overall difference
        overall_difference = weighted_sum / total_weight if total_weight > 0 else 0.0
        metrics['overall_difference'] = overall_difference
        metrics['normalization_method'] = self.normalization_method
        
        logger.debug(f"Calculated parameter difference: {overall_difference:.3f}")
        return overall_difference, metrics
    
    def batch_calculate_differences(
        self, 
        device_predicate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Calculate parameter differences for multiple device-predicate pairs.
        
        Args:
            device_predicate_pairs: List of (device_params, predicate_params) tuples
            
        Returns:
            List of (difference, metrics) tuples
        """
        # First normalize all parameters together
        all_params = []
        for device_params, predicate_params in device_predicate_pairs:
            all_params.extend([device_params, predicate_params])
        
        normalized_params = self.normalize_parameters(all_params)
        
        # Calculate differences using normalized parameters
        results = []
        for i in range(0, len(normalized_params), 2):
            device_params_norm = normalized_params[i]
            predicate_params_norm = normalized_params[i + 1]
            
            diff, metrics = self.calculate_parameter_difference(device_params_norm, predicate_params_norm)
            results.append((diff, metrics))
        
        logger.info(f"Calculated parameter differences for {len(device_predicate_pairs)} pairs")
        return results


def main():
    """Test parameter difference calculation."""
    calculator = ParameterDifferenceCalculator()
    
    # Test with sample device parameters
    device1_desc = """
    Modular titanium hip stem with 12mm diameter, 150mm length, 
    porous coating for osseointegration, cementless design.
    """
    
    device1_specs = {
        'stem_diameter': 12.0,
        'stem_length': 150.0,
        'material_titanium': 1.0,
        'coating_porous': 1.0,
        'design_modular': 1.0
    }
    
    predicate1_desc = """
    Standard titanium hip implant with 10mm diameter, 140mm length,
    smooth surface, cemented fixation.
    """
    
    predicate1_specs = {
        'stem_diameter': 10.0,
        'stem_length': 140.0,
        'material_titanium': 1.0,
        'coating_porous': 0.0,
        'design_modular': 0.0
    }
    
    # Extract parameters
    device1_params = calculator.extract_parameters(device1_desc, device1_specs)
    predicate1_params = calculator.extract_parameters(predicate1_desc, predicate1_specs)
    
    # Calculate difference
    difference, metrics = calculator.calculate_parameter_difference(device1_params, predicate1_params)
    
    print(f"Parameter Difference: {difference:.3f}")
    print("Category Differences:")
    for category, data in metrics['category_differences'].items():
        print(f"  {category}: {data['difference']:.3f} ({data['parameter_count']} parameters)")


if __name__ == "__main__":
    main()