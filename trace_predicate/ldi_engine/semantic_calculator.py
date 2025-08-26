"""
Semantic Distance Calculator for LDI Engine

This module implements semantic distance calculation using BioBERT embeddings
to measure the conceptual drift between medical device descriptions and
technical documentation.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Note: In production, would use transformers with BioBERT
# For this implementation, using TF-IDF as a practical approximation
logger = logging.getLogger(__name__)


class SemanticDistanceCalculator:
    """Calculate semantic distances between medical device descriptions."""
    
    def __init__(self, model_type: str = "tfidf"):
        """
        Initialize semantic distance calculator.
        
        Args:
            model_type: Type of model to use ('tfidf', 'biobert')
        """
        self.model_type = model_type
        self.vectorizer = None
        self.biobert_model = None
        self.biobert_tokenizer = None
        
        # Medical device specific vocabulary weights
        self.medical_terms = {
            'implant', 'prosthetic', 'orthopedic', 'titanium', 'ceramic', 
            'polyethylene', 'acetabular', 'femoral', 'hip', 'joint',
            'bearing', 'modular', 'cemented', 'cementless', 'porous',
            'coating', 'biocompatible', 'wear', 'friction', 'stability'
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the semantic model."""
        if self.model_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8
            )
            logger.info("Initialized TF-IDF vectorizer for semantic analysis")
            
        elif self.model_type == "biobert":
            try:
                from transformers import AutoTokenizer, AutoModel
                model_name = "dmis-lab/biobert-base-cased-v1.1"
                self.biobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.biobert_model = AutoModel.from_pretrained(model_name)
                logger.info("Initialized BioBERT model for semantic analysis")
            except ImportError:
                logger.warning("Transformers not available, falling back to TF-IDF")
                self.model_type = "tfidf"
                self._initialize_model()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess medical device text for semantic analysis.
        
        Args:
            text: Raw device description text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand common medical abbreviations
        abbreviations = {
            'thr': 'total hip replacement',
            'tha': 'total hip arthroplasty',
            'uhmwpe': 'ultra high molecular weight polyethylene',
            'cad': 'computer aided design',
            'fda': 'food and drug administration'
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        return text
    
    def extract_technical_features(self, text: str) -> Dict[str, float]:
        """
        Extract technical features from device description.
        
        Args:
            text: Device description text
            
        Returns:
            Dictionary of technical features with weights
        """
        text = self.preprocess_text(text)
        features = {}
        
        # Material composition features
        materials = {
            'titanium': ['titanium', 'ti-6al-4v', 'titanium alloy'],
            'ceramic': ['ceramic', 'alumina', 'zirconia', 'biolox'],
            'polyethylene': ['polyethylene', 'pe', 'uhmwpe', 'crosslinked'],
            'cobalt_chrome': ['cobalt chrome', 'cocr', 'cobalt chromium']
        }
        
        for material, terms in materials.items():
            features[f'material_{material}'] = sum(1 for term in terms if term in text)
        
        # Design features
        design_features = {
            'modular': ['modular', 'modular design', 'interchangeable'],
            'cemented': ['cemented', 'bone cement', 'pmma'],
            'cementless': ['cementless', 'press-fit', 'interference fit'],
            'porous': ['porous', 'porous coating', 'osteointegration'],
            'bearing': ['bearing', 'articulating', 'low friction']
        }
        
        for feature, terms in design_features.items():
            features[f'design_{feature}'] = sum(1 for term in terms if term in text)
        
        # Size and dimension indicators
        size_indicators = ['mm', 'diameter', 'length', 'size', 'small', 'medium', 'large']
        features['size_specificity'] = sum(1 for indicator in size_indicators if indicator in text)
        
        # Innovation indicators
        innovation_terms = ['new', 'improved', 'enhanced', 'advanced', 'novel', 'innovative']
        features['innovation_score'] = sum(1 for term in innovation_terms if term in text)
        
        return features
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using TF-IDF vectors.
        
        Args:
            text1: First device description
            text2: Second device description
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Preprocess texts
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        # Fit vectorizer on both texts
        corpus = [processed_text1, processed_text2]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity_matrix[0][0])
    
    def calculate_feature_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity based on extracted technical features.
        
        Args:
            text1: First device description
            text2: Second device description
            
        Returns:
            Feature similarity score (0-1)
        """
        features1 = self.extract_technical_features(text1)
        features2 = self.extract_technical_features(text2)
        
        # Get all unique feature keys
        all_features = set(features1.keys()) | set(features2.keys())
        
        if not all_features:
            return 0.0
        
        # Create feature vectors
        vector1 = np.array([features1.get(f, 0) for f in all_features])
        vector2 = np.array([features2.get(f, 0) for f in all_features])
        
        # Calculate cosine similarity
        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            return 0.0
        
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return float(similarity)
    
    def calculate_biobert_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using BioBERT embeddings.
        
        Args:
            text1: First device description
            text2: Second device description
            
        Returns:
            BioBERT similarity score (0-1)
        """
        if self.model_type != "biobert" or not self.biobert_model:
            logger.warning("BioBERT not available, using TF-IDF")
            return self.calculate_tfidf_similarity(text1, text2)
        
        try:
            import torch
            
            # Tokenize and encode texts
            inputs1 = self.biobert_tokenizer(
                self.preprocess_text(text1), 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            inputs2 = self.biobert_tokenizer(
                self.preprocess_text(text2), 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs1 = self.biobert_model(**inputs1)
                outputs2 = self.biobert_model(**inputs2)
                
                # Use [CLS] token embeddings
                embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
                embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"BioBERT calculation failed: {e}")
            return self.calculate_tfidf_similarity(text1, text2)
    
    def calculate_semantic_distance(
        self, 
        device_description: str, 
        predicate_description: str,
        method: str = "hybrid"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate semantic distance between device and its predicate.
        
        Args:
            device_description: Current device description
            predicate_description: Predicate device description
            method: Calculation method ('tfidf', 'biobert', 'hybrid')
            
        Returns:
            Tuple of (semantic_distance, detailed_metrics)
        """
        metrics = {}
        
        # Calculate different similarity measures
        if method in ["tfidf", "hybrid"]:
            tfidf_sim = self.calculate_tfidf_similarity(device_description, predicate_description)
            metrics['tfidf_similarity'] = tfidf_sim
        
        if method in ["biobert", "hybrid"]:
            biobert_sim = self.calculate_biobert_similarity(device_description, predicate_description)
            metrics['biobert_similarity'] = biobert_sim
        
        # Feature-based similarity
        feature_sim = self.calculate_feature_similarity(device_description, predicate_description)
        metrics['feature_similarity'] = feature_sim
        
        # Combine similarities based on method
        if method == "tfidf":
            combined_similarity = (metrics['tfidf_similarity'] + feature_sim) / 2
        elif method == "biobert":
            combined_similarity = (metrics['biobert_similarity'] + feature_sim) / 2
        else:  # hybrid
            # Weight BioBERT higher if available, otherwise use TF-IDF
            if 'biobert_similarity' in metrics:
                combined_similarity = (
                    0.5 * metrics['biobert_similarity'] + 
                    0.3 * metrics['tfidf_similarity'] + 
                    0.2 * feature_sim
                )
            else:
                combined_similarity = (
                    0.7 * metrics['tfidf_similarity'] + 
                    0.3 * feature_sim
                )
        
        # Convert similarity to distance (1 - similarity)
        semantic_distance = 1.0 - combined_similarity
        metrics['combined_similarity'] = combined_similarity
        metrics['semantic_distance'] = semantic_distance
        
        logger.debug(f"Calculated semantic distance: {semantic_distance:.3f}")
        return semantic_distance, metrics
    
    def batch_calculate_distances(
        self, 
        device_predicate_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[float, Dict[str, float]]]:
        """
        Calculate semantic distances for multiple device-predicate pairs.
        
        Args:
            device_predicate_pairs: List of (device_desc, predicate_desc) tuples
            
        Returns:
            List of (distance, metrics) tuples
        """
        results = []
        
        for device_desc, predicate_desc in device_predicate_pairs:
            distance, metrics = self.calculate_semantic_distance(device_desc, predicate_desc)
            results.append((distance, metrics))
        
        logger.info(f"Calculated semantic distances for {len(device_predicate_pairs)} pairs")
        return results


def main():
    """Test semantic distance calculation."""
    calculator = SemanticDistanceCalculator()
    
    # Test with sample hip implant descriptions
    device1 = """
    The XYZ Hip System is a modular total hip arthroplasty system consisting of 
    titanium alloy femoral stems, ceramic acetabular liners, and ultra-high 
    molecular weight polyethylene inserts. The system features advanced porous 
    coating for enhanced osseointegration.
    """
    
    predicate1 = """
    The ABC Hip Implant is a cementless total hip replacement system with 
    titanium femoral components and polyethylene acetabular cups. The device 
    utilizes traditional design principles for hip arthroplasty.
    """
    
    distance, metrics = calculator.calculate_semantic_distance(device1, predicate1)
    
    print(f"Semantic Distance: {distance:.3f}")
    print("Detailed Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    main()