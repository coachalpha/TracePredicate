"""
FDA Medical Device Recall Database Scraper

This module provides functionality to scrape and analyze FDA medical device
recall data, focusing on correlating recalls with device lineages and
calculating recall risk metrics.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class RecallClass(Enum):
    """FDA recall classification."""
    CLASS_I = "Class I"      # Most serious - life-threatening
    CLASS_II = "Class II"    # May cause temporary health problems
    CLASS_III = "Class III"  # Unlikely to cause health problems
    

class RecallStatus(Enum):
    """Recall status."""
    ONGOING = "Ongoing"
    COMPLETED = "Completed"
    TERMINATED = "Terminated"
    

@dataclass
class DeviceRecall:
    """Represents an FDA medical device recall."""
    recall_number: str
    recall_date: datetime
    recall_class: RecallClass
    recall_status: RecallStatus
    
    # Product information
    product_description: str
    manufacturer: str
    brand_name: Optional[str]
    product_code: str
    
    # Recall details
    reason_for_recall: str
    recall_initiation_date: Optional[datetime]
    recall_distribution_pattern: Optional[str]
    quantity_in_commerce: Optional[int]
    quantity_recalled: Optional[int]
    
    # Risk assessment
    root_cause: Optional[str] = None
    risk_to_health: Optional[str] = None
    corrective_action: Optional[str] = None
    
    # Linkage to 510(k) devices
    associated_k_numbers: List[str] = None
    
    def __post_init__(self):
        if self.associated_k_numbers is None:
            self.associated_k_numbers = []


class FDARecallScraper:
    """Scraper for FDA medical device recall database."""
    
    BASE_URL = "https://www.fda.gov/medical-devices/medical-device-recalls"
    SEARCH_URL = "https://www.fda.gov/medical-devices/medical-device-recalls/medical-device-recalls-database"
    API_URL = "https://api.fda.gov/device/recall.json"  # FDA openFDA API
    
    def __init__(self, session_timeout: int = 30, api_key: Optional[str] = None):
        self.session_timeout = session_timeout
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TracePredicate-Research/0.1.0 (Medical Device Analysis)'
        })
    
    def search_recalls_by_product_code(
        self, 
        product_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        recall_classes: List[RecallClass] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search for device recalls by product code using FDA openFDA API.
        
        Args:
            product_code: FDA product code (e.g., 'KWA')
            start_date: Start date for recall search
            end_date: End date for recall search
            recall_classes: List of recall classes to include
            limit: Maximum number of results
            
        Returns:
            List of recall dictionaries
        """
        logger.info(f"Searching recalls for product code: {product_code}")
        
        # Build search query
        search_terms = [f'product_code:"{product_code}"']
        
        if start_date and end_date:
            date_range = f"report_date:[{start_date.strftime('%Y%m%d')}+TO+{end_date.strftime('%Y%m%d')}]"
            search_terms.append(date_range)
        
        if recall_classes:
            class_terms = '+OR+'.join([f'classification:"{cls.value}"' for cls in recall_classes])
            search_terms.append(f"({class_terms})")
        
        search_query = '+AND+'.join(search_terms)
        
        params = {
            'search': search_query,
            'limit': min(limit, 1000),  # FDA API limit
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = self.session.get(self.API_URL, params=params, timeout=self.session_timeout)
            response.raise_for_status()
            
            data = response.json()
            recalls = data.get('results', [])
            
            logger.info(f"Found {len(recalls)} recalls via API")
            return recalls
            
        except requests.RequestException as e:
            logger.error(f"Error searching recalls via API: {e}")
            # Fallback to web scraping if API fails
            return self._fallback_web_search(product_code, start_date, end_date, limit)
    
    def _fallback_web_search(
        self, 
        product_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fallback web scraping method when API is unavailable.
        
        Args:
            product_code: FDA product code
            start_date: Start date for search
            end_date: End date for search
            limit: Maximum results
            
        Returns:
            List of recall dictionaries
        """
        logger.info("Using fallback web scraping for recalls")
        
        # This would require more complex web scraping implementation
        # For now, return empty list and log warning
        logger.warning("Web scraping fallback not fully implemented")
        return []
    
    def get_recall_details(self, recall_data: Dict) -> DeviceRecall:
        """
        Convert API recall data to DeviceRecall object.
        
        Args:
            recall_data: Raw recall data from FDA API
            
        Returns:
            DeviceRecall object with processed information
        """
        try:
            # Parse recall class
            classification = recall_data.get('classification', '')
            if 'I' in classification and 'II' not in classification:
                recall_class = RecallClass.CLASS_I
            elif 'II' in classification:
                recall_class = RecallClass.CLASS_II
            elif 'III' in classification:
                recall_class = RecallClass.CLASS_III
            else:
                recall_class = RecallClass.CLASS_II  # Default
            
            # Parse dates
            recall_date = self._parse_fda_date(recall_data.get('report_date'))
            initiation_date = self._parse_fda_date(recall_data.get('recall_initiation_date'))
            
            # Parse status
            status_text = recall_data.get('status', '').lower()
            if 'ongoing' in status_text:
                status = RecallStatus.ONGOING
            elif 'completed' in status_text or 'terminated' in status_text:
                status = RecallStatus.COMPLETED
            else:
                status = RecallStatus.ONGOING  # Default
            
            # Extract K-numbers from product description
            product_desc = recall_data.get('product_description', '')
            k_numbers = self._extract_k_numbers(product_desc)
            
            recall = DeviceRecall(
                recall_number=recall_data.get('recall_number', ''),
                recall_date=recall_date,
                recall_class=recall_class,
                recall_status=status,
                product_description=product_desc,
                manufacturer=recall_data.get('recalling_firm', ''),
                brand_name=recall_data.get('brand_name'),
                product_code=recall_data.get('product_code', ''),
                reason_for_recall=recall_data.get('reason_for_recall', ''),
                recall_initiation_date=initiation_date,
                recall_distribution_pattern=recall_data.get('distribution_pattern'),
                quantity_in_commerce=self._parse_quantity(recall_data.get('quantity_in_commerce')),
                quantity_recalled=self._parse_quantity(recall_data.get('quantity_recalled')),
                root_cause=recall_data.get('root_cause_description'),
                risk_to_health=recall_data.get('classification'),
                corrective_action=recall_data.get('action'),
                associated_k_numbers=k_numbers
            )
            
            return recall
            
        except Exception as e:
            logger.error(f"Error processing recall data: {e}")
            # Return minimal recall object
            return DeviceRecall(
                recall_number=recall_data.get('recall_number', 'UNKNOWN'),
                recall_date=datetime.now(),
                recall_class=RecallClass.CLASS_II,
                recall_status=RecallStatus.ONGOING,
                product_description=recall_data.get('product_description', ''),
                manufacturer=recall_data.get('recalling_firm', ''),
                brand_name=None,
                product_code=recall_data.get('product_code', ''),
                reason_for_recall=recall_data.get('reason_for_recall', ''),
                recall_initiation_date=None,
                recall_distribution_pattern=None,
                quantity_in_commerce=None,
                quantity_recalled=None
            )
    
    def _parse_fda_date(self, date_string: str) -> Optional[datetime]:
        """Parse FDA API date format (YYYYMMDD)."""
        if not date_string:
            return None
            
        try:
            # FDA API uses YYYYMMDD format
            return datetime.strptime(date_string, '%Y%m%d')
        except ValueError:
            try:
                # Try alternative formats
                return datetime.strptime(date_string, '%Y-%m-%d')
            except ValueError:
                logger.debug(f"Could not parse FDA date: {date_string}")
                return None
    
    def _extract_k_numbers(self, text: str) -> List[str]:
        """Extract K-numbers from text."""
        if not text:
            return []
            
        # Pattern to match K-numbers (K followed by 6 digits)
        k_pattern = re.compile(r'K(\d{6})', re.IGNORECASE)
        matches = k_pattern.findall(text)
        
        return [f"K{match}" for match in matches]
    
    def _parse_quantity(self, quantity_str: str) -> Optional[int]:
        """Parse quantity strings that may contain numbers and text."""
        if not quantity_str:
            return None
            
        # Extract numbers from string
        numbers = re.findall(r'\d+', quantity_str.replace(',', ''))
        if numbers:
            try:
                return int(numbers[0])  # Take first number found
            except ValueError:
                pass
                
        return None
    
    def analyze_recall_patterns(self, recalls: List[DeviceRecall]) -> Dict[str, any]:
        """
        Analyze patterns in recall data.
        
        Args:
            recalls: List of DeviceRecall objects
            
        Returns:
            Dictionary with analysis results
        """
        if not recalls:
            return {}
        
        # Calculate recall metrics
        total_recalls = len(recalls)
        class_counts = {cls: 0 for cls in RecallClass}
        root_causes = {}
        manufacturer_counts = {}
        yearly_counts = {}
        
        for recall in recalls:
            # Count by class
            class_counts[recall.recall_class] += 1
            
            # Count root causes
            if recall.root_cause:
                root_causes[recall.root_cause] = root_causes.get(recall.root_cause, 0) + 1
            
            # Count by manufacturer
            manufacturer_counts[recall.manufacturer] = manufacturer_counts.get(recall.manufacturer, 0) + 1
            
            # Count by year
            if recall.recall_date:
                year = recall.recall_date.year
                yearly_counts[year] = yearly_counts.get(year, 0) + 1
        
        # Calculate recall rates and severity
        class_i_rate = class_counts[RecallClass.CLASS_I] / total_recalls
        average_quantity_recalled = sum(
            r.quantity_recalled for r in recalls 
            if r.quantity_recalled is not None
        ) / len([r for r in recalls if r.quantity_recalled is not None]) if any(r.quantity_recalled for r in recalls) else 0
        
        analysis = {
            'total_recalls': total_recalls,
            'recall_by_class': {cls.value: count for cls, count in class_counts.items()},
            'class_i_rate': class_i_rate,
            'top_root_causes': sorted(root_causes.items(), key=lambda x: x[1], reverse=True)[:10],
            'top_manufacturers': sorted(manufacturer_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'recalls_by_year': dict(sorted(yearly_counts.items())),
            'average_quantity_recalled': average_quantity_recalled,
            'k_numbers_found': sum(len(r.associated_k_numbers) for r in recalls)
        }
        
        logger.info(f"Recall analysis completed: {total_recalls} recalls analyzed")
        return analysis
    
    def match_recalls_to_devices(self, recalls: List[DeviceRecall], device_k_numbers: Set[str]) -> Dict[str, List[DeviceRecall]]:
        """
        Match recalls to specific 510(k) devices.
        
        Args:
            recalls: List of recall objects
            device_k_numbers: Set of K-numbers to match against
            
        Returns:
            Dictionary mapping K-numbers to associated recalls
        """
        device_recalls = {k_num: [] for k_num in device_k_numbers}
        
        for recall in recalls:
            for k_number in recall.associated_k_numbers:
                if k_number in device_k_numbers:
                    device_recalls[k_number].append(recall)
        
        # Also try fuzzy matching based on product names/manufacturers
        # This would require more sophisticated matching logic
        
        matched_count = sum(len(recalls) for recalls in device_recalls.values())
        logger.info(f"Matched {matched_count} recalls to {len(device_k_numbers)} devices")
        
        return device_recalls
    
    def calculate_recall_risk_score(self, recalls: List[DeviceRecall]) -> float:
        """
        Calculate a risk score based on recall history.
        
        Args:
            recalls: List of recalls for a device or device family
            
        Returns:
            Risk score (0-1, higher is riskier)
        """
        if not recalls:
            return 0.0
        
        risk_score = 0.0
        
        # Weight by recall class
        class_weights = {
            RecallClass.CLASS_I: 1.0,
            RecallClass.CLASS_II: 0.6,
            RecallClass.CLASS_III: 0.3
        }
        
        for recall in recalls:
            weight = class_weights[recall.recall_class]
            
            # Adjust for recency (more recent recalls are riskier)
            if recall.recall_date:
                years_ago = (datetime.now() - recall.recall_date).days / 365.25
                recency_factor = max(0.1, 1.0 - (years_ago / 10))  # Decay over 10 years
                weight *= recency_factor
            
            # Adjust for quantity recalled
            if recall.quantity_recalled and recall.quantity_in_commerce:
                if recall.quantity_in_commerce > 0:
                    recall_rate = recall.quantity_recalled / recall.quantity_in_commerce
                    weight *= (1 + recall_rate)  # Higher recall rate increases risk
            
            risk_score += weight
        
        # Normalize to 0-1 range
        # Assume 5 Class I recalls in recent years would be maximum risk
        max_possible_score = 5.0
        normalized_score = min(1.0, risk_score / max_possible_score)
        
        return normalized_score
    
    def export_to_dataframe(self, recalls: List[DeviceRecall]) -> pd.DataFrame:
        """Export recalls to pandas DataFrame."""
        data = []
        
        for recall in recalls:
            k_numbers_str = "|".join(recall.associated_k_numbers) if recall.associated_k_numbers else ""
            
            data.append({
                'recall_number': recall.recall_number,
                'recall_date': recall.recall_date,
                'recall_class': recall.recall_class.value,
                'recall_status': recall.recall_status.value,
                'product_description': recall.product_description,
                'manufacturer': recall.manufacturer,
                'brand_name': recall.brand_name,
                'product_code': recall.product_code,
                'reason_for_recall': recall.reason_for_recall,
                'recall_initiation_date': recall.recall_initiation_date,
                'distribution_pattern': recall.recall_distribution_pattern,
                'quantity_in_commerce': recall.quantity_in_commerce,
                'quantity_recalled': recall.quantity_recalled,
                'root_cause': recall.root_cause,
                'risk_to_health': recall.risk_to_health,
                'corrective_action': recall.corrective_action,
                'associated_k_numbers': k_numbers_str,
                'k_number_count': len(recall.associated_k_numbers)
            })
        
        return pd.DataFrame(data)


def main():
    """Example usage of FDA recall scraper."""
    scraper = FDARecallScraper()
    
    # Search for hip implant recalls
    recall_data = scraper.search_recalls_by_product_code("KWA", limit=50)
    
    # Process into DeviceRecall objects
    recalls = [scraper.get_recall_details(data) for data in recall_data]
    
    # Analyze patterns
    analysis = scraper.analyze_recall_patterns(recalls)
    print(f"Found {len(recalls)} recalls")
    print(f"Class I recalls: {analysis.get('recall_by_class', {}).get('Class I', 0)}")
    
    # Export to DataFrame
    df = scraper.export_to_dataframe(recalls)
    if not df.empty:
        print(df[['recall_number', 'recall_class', 'manufacturer']].head())


if __name__ == "__main__":
    main()