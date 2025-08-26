"""
Human Factors Engineering (HFE) Classification for Adverse Events

This module implements HFE-based classification of adverse events to distinguish
between device-related failures and user-error related incidents, as specified
in the TracePredicate research plan.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class HFECategory(Enum):
    """Human Factors Engineering event categories."""
    DEVICE_FAILURE = "device_failure"           # Clear device malfunction
    USER_ERROR = "user_error"                   # Clear user mistake
    DESIGN_INDUCED_ERROR = "design_induced_error"  # Design flaw leading to user error
    INSUFFICIENT_INFORMATION = "insufficient_information"  # Cannot be classified


class HFEClassifier:
    """Classify adverse events using Human Factors Engineering principles."""
    
    def __init__(self):
        """Initialize HFE classifier."""
        self.classifier = None
        self.vectorizer = None
        self.is_trained = False
        
        # Define keyword patterns for each HFE category
        self.hfe_patterns = {
            HFECategory.DEVICE_FAILURE: {
                'keywords': [
                    'fracture', 'break', 'crack', 'failure', 'malfunction', 'defect',
                    'wear', 'corrosion', 'loosening', 'dislocation', 'separation',
                    'manufacturing', 'material', 'component failure', 'mechanical failure',
                    'fatigue', 'oxidation', 'delamination', 'brittleness'
                ],
                'patterns': [
                    r'device.*fail',
                    r'implant.*break',
                    r'component.*fracture',
                    r'material.*defect',
                    r'manufacturing.*error'
                ]
            },
            HFECategory.USER_ERROR: {
                'keywords': [
                    'incorrect', 'improper', 'wrong', 'inappropriate', 'misuse',
                    'error', 'mistake', 'procedure', 'technique', 'training',
                    'inexperience', 'surgeon error', 'positioning', 'alignment',
                    'overreaming', 'undertapping', 'malalignment'
                ],
                'patterns': [
                    r'surgeon.*error',
                    r'improper.*technique',
                    r'incorrect.*placement',
                    r'wrong.*size',
                    r'inadequate.*training'
                ]
            },
            HFECategory.DESIGN_INDUCED_ERROR: {
                'keywords': [
                    'confusing', 'unclear', 'ambiguous', 'misleading', 'difficult',
                    'complex', 'instruction', 'labeling', 'interface', 'design',
                    'ergonomic', 'usability', 'counter-intuitive', 'hard to use',
                    'similar appearance', 'look alike', 'sound alike'
                ],
                'patterns': [
                    r'design.*confusing',
                    r'unclear.*instruction',
                    r'difficult.*to.*use',
                    r'misleading.*label',
                    r'counter.*intuitive'
                ]
            }
        }
        
        logger.info("Initialized HFE classifier")
    
    def _extract_hfe_features(self, text: str) -> Dict[str, float]:
        """
        Extract HFE-relevant features from adverse event text.
        
        Args:
            text: Adverse event description
            
        Returns:
            Dictionary of HFE feature scores
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        features = {}
        
        for category, patterns in self.hfe_patterns.items():
            # Count keyword matches
            keyword_score = sum(1 for keyword in patterns['keywords'] 
                              if keyword in text_lower)
            
            # Count regex pattern matches
            pattern_score = sum(1 for pattern in patterns['patterns'] 
                              if re.search(pattern, text_lower))
            
            # Normalize by total possible matches
            total_possible = len(patterns['keywords']) + len(patterns['patterns'])
            features[f'{category.value}_score'] = (keyword_score + pattern_score) / total_possible
        
        # Additional linguistic features
        features['text_length'] = len(text.split())
        features['technical_terms'] = self._count_technical_terms(text_lower)
        features['temporal_indicators'] = self._count_temporal_indicators(text_lower)
        features['certainty_indicators'] = self._count_certainty_indicators(text_lower)
        
        return features
    
    def _count_technical_terms(self, text: str) -> float:
        """Count technical medical device terms."""
        technical_terms = [
            'acetabular', 'femoral', 'prosthetic', 'implant', 'component',
            'bearing', 'liner', 'stem', 'head', 'cup', 'modular',
            'cemented', 'cementless', 'porous', 'coating', 'titanium',
            'polyethylene', 'ceramic', 'cobalt', 'chrome'
        ]
        return sum(1 for term in technical_terms if term in text) / len(technical_terms)
    
    def _count_temporal_indicators(self, text: str) -> float:
        """Count temporal indicators that might suggest device failure vs user error."""
        temporal_terms = [
            'immediately', 'instantly', 'during', 'after', 'before',
            'surgery', 'implantation', 'months', 'years', 'weeks',
            'days', 'hours', 'post-operative', 'intraoperative'
        ]
        return sum(1 for term in temporal_terms if term in text) / len(temporal_terms)
    
    def _count_certainty_indicators(self, text: str) -> float:
        """Count certainty/uncertainty indicators."""
        certainty_terms = [
            'likely', 'probably', 'possible', 'suspected', 'appears',
            'seems', 'unclear', 'unknown', 'uncertain', 'definite',
            'confirmed', 'verified', 'observed', 'noted'
        ]
        return sum(1 for term in certainty_terms if term in text) / len(certainty_terms)
    
    def classify_event_rule_based(self, event_text: str) -> Tuple[HFECategory, float]:
        """
        Classify adverse event using rule-based approach.
        
        Args:
            event_text: Adverse event description text
            
        Returns:
            Tuple of (HFE category, confidence score)
        """
        features = self._extract_hfe_features(event_text)
        
        # Get scores for each category
        device_score = features.get('device_failure_score', 0)
        user_score = features.get('user_error_score', 0)
        design_score = features.get('design_induced_error_score', 0)
        
        # Determine category based on highest score
        scores = {
            HFECategory.DEVICE_FAILURE: device_score,
            HFECategory.USER_ERROR: user_score,
            HFECategory.DESIGN_INDUCED_ERROR: design_score
        }
        
        max_category = max(scores, key=scores.get)
        max_score = scores[max_category]
        
        # If all scores are very low, classify as insufficient information
        if max_score < 0.1:
            return HFECategory.INSUFFICIENT_INFORMATION, 0.5
        
        # Confidence is based on score magnitude and separation from other scores
        other_scores = [score for cat, score in scores.items() if cat != max_category]
        separation = max_score - max(other_scores) if other_scores else max_score
        confidence = min(0.9, max_score + separation)
        
        return max_category, confidence
    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare synthetic training data for HFE classification.
        
        Returns:
            DataFrame with labeled training examples
        """
        # Synthetic training examples based on real adverse event patterns
        training_data = [
            # Device failure examples
            ("Femoral stem fractured at the neck junction during normal ambulation", "device_failure"),
            ("Acetabular liner showed severe wear after 5 years", "device_failure"),
            ("Component separation occurred, head dissociated from stem", "device_failure"),
            ("Manufacturing defect caused early failure of the device", "device_failure"),
            ("Material corrosion led to implant degradation", "device_failure"),
            ("Device malfunction resulted in mechanical loosening", "device_failure"),
            ("Ceramic head fractured during patient activity", "device_failure"),
            ("Polyethylene wear debris caused inflammatory response", "device_failure"),
            
            # User error examples
            ("Surgeon selected wrong implant size during procedure", "user_error"),
            ("Improper surgical technique resulted in malalignment", "user_error"),
            ("Incorrect cementing technique led to poor fixation", "user_error"),
            ("Inadequate reaming caused loose fit of acetabular component", "user_error"),
            ("Surgeon error in component positioning", "user_error"),
            ("Wrong angle of insertion during implantation", "user_error"),
            ("Inappropriate patient selection for this device type", "user_error"),
            ("Overtightening of modular connection during surgery", "user_error"),
            
            # Design-induced error examples
            ("Confusing labeling led to wrong component selection", "design_induced_error"),
            ("Similar appearance of different sized components caused mix-up", "design_induced_error"),
            ("Unclear surgical instructions resulted in improper technique", "design_induced_error"),
            ("Counter-intuitive assembly process led to error", "design_induced_error"),
            ("Design made it difficult to achieve proper alignment", "design_induced_error"),
            ("Ambiguous markings on instrument caused confusion", "design_induced_error"),
            ("Poor ergonomic design contributed to surgical error", "design_induced_error"),
            ("Misleading packaging information led to wrong selection", "design_induced_error"),
            
            # Insufficient information examples
            ("Patient experienced pain post-surgery", "insufficient_information"),
            ("Device revision was required", "insufficient_information"),
            ("Complication occurred during recovery", "insufficient_information"),
            ("Patient reported discomfort", "insufficient_information")
        ]
        
        return pd.DataFrame(training_data, columns=['text', 'category'])
    
    def train_ml_classifier(self):
        """Train machine learning classifier for HFE categorization."""
        # Get training data
        training_df = self.prepare_training_data()
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True
            )),
            ('nb', MultinomialNB(alpha=1.0))
        ])
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            training_df['text'], training_df['category'], 
            test_size=0.2, random_state=42, stratify=training_df['category']
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Validate performance
        y_pred = self.classifier.predict(X_test)
        
        logger.info("HFE Classifier Performance:")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        self.is_trained = True
        logger.info("ML-based HFE classifier training completed")
    
    def classify_event_ml(self, event_text: str) -> Tuple[HFECategory, float]:
        """
        Classify adverse event using machine learning approach.
        
        Args:
            event_text: Adverse event description text
            
        Returns:
            Tuple of (HFE category, confidence score)
        """
        if not self.is_trained:
            logger.warning("ML classifier not trained, training now...")
            self.train_ml_classifier()
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba([event_text])[0]
        classes = self.classifier.classes_
        
        # Find best prediction
        best_idx = np.argmax(probabilities)
        best_category = HFECategory(classes[best_idx])
        confidence = probabilities[best_idx]
        
        return best_category, confidence
    
    def classify_event_hybrid(self, event_text: str) -> Tuple[HFECategory, float, Dict[str, Any]]:
        """
        Classify adverse event using hybrid rule-based + ML approach.
        
        Args:
            event_text: Adverse event description text
            
        Returns:
            Tuple of (HFE category, confidence score, detailed results)
        """
        # Get rule-based classification
        rule_category, rule_confidence = self.classify_event_rule_based(event_text)
        
        # Get ML-based classification
        ml_category, ml_confidence = self.classify_event_ml(event_text)
        
        # Extract features for analysis
        features = self._extract_hfe_features(event_text)
        
        # Combine results
        if rule_category == ml_category:
            # Both methods agree - high confidence
            final_category = rule_category
            final_confidence = min(0.95, (rule_confidence + ml_confidence) / 2 + 0.2)
        else:
            # Methods disagree - use higher confidence prediction
            if rule_confidence > ml_confidence:
                final_category = rule_category
                final_confidence = rule_confidence * 0.8  # Reduce confidence due to disagreement
            else:
                final_category = ml_category
                final_confidence = ml_confidence * 0.8
        
        detailed_results = {
            'rule_based': {'category': rule_category.value, 'confidence': rule_confidence},
            'ml_based': {'category': ml_category.value, 'confidence': ml_confidence},
            'features': features,
            'agreement': rule_category == ml_category
        }
        
        return final_category, final_confidence, detailed_results
    
    def batch_classify_events(self, events: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple adverse events in batch.
        
        Args:
            events: List of adverse event description texts
            
        Returns:
            List of classification results
        """
        results = []
        
        for i, event_text in enumerate(events):
            category, confidence, detailed = self.classify_event_hybrid(event_text)
            
            results.append({
                'event_id': i,
                'event_text': event_text,
                'hfe_category': category.value,
                'confidence': confidence,
                'detailed_analysis': detailed
            })
        
        logger.info(f"Classified {len(events)} adverse events using HFE methodology")
        return results
    
    def analyze_hfe_distribution(self, classification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze distribution of HFE classifications.
        
        Args:
            classification_results: Results from batch classification
            
        Returns:
            HFE distribution analysis
        """
        if not classification_results:
            return {}
        
        # Count categories
        category_counts = {}
        confidence_scores = []
        agreement_rate = 0
        
        for result in classification_results:
            category = result['hfe_category']
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_scores.append(result['confidence'])
            
            if result['detailed_analysis']['agreement']:
                agreement_rate += 1
        
        agreement_rate = agreement_rate / len(classification_results)
        
        # Calculate percentages
        total_events = len(classification_results)
        category_percentages = {
            category: count / total_events * 100 
            for category, count in category_counts.items()
        }
        
        analysis = {
            'total_events': total_events,
            'category_counts': category_counts,
            'category_percentages': category_percentages,
            'mean_confidence': np.mean(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'rule_ml_agreement_rate': agreement_rate * 100,
            'device_related_percentage': (
                category_percentages.get('device_failure', 0) + 
                category_percentages.get('design_induced_error', 0)
            ),
            'user_related_percentage': category_percentages.get('user_error', 0)
        }
        
        return analysis


def main():
    """Test HFE classification system."""
    classifier = HFEClassifier()
    
    # Train the ML classifier
    classifier.train_ml_classifier()
    
    # Test with sample adverse events
    test_events = [
        "Femoral stem fractured at the neck during normal walking activity",
        "Surgeon selected wrong implant size causing poor fit",
        "Confusing labeling led to mix-up of similar components",
        "Patient experienced discomfort after surgery"
    ]
    
    print("HFE Classification Results:")
    print("=" * 50)
    
    for event in test_events:
        category, confidence, detailed = classifier.classify_event_hybrid(event)
        print(f"\nEvent: {event[:50]}...")
        print(f"HFE Category: {category.value}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Rule-ML Agreement: {detailed['agreement']}")
    
    # Batch classification
    batch_results = classifier.batch_classify_events(test_events)
    distribution = classifier.analyze_hfe_distribution(batch_results)
    
    print(f"\nHFE Distribution Analysis:")
    print(f"Device-related events: {distribution['device_related_percentage']:.1f}%")
    print(f"User-related events: {distribution['user_related_percentage']:.1f}%")
    print(f"Mean confidence: {distribution['mean_confidence']:.3f}")


if __name__ == "__main__":
    main()