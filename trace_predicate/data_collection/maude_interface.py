"""
MAUDE (Manufacturer and User Facility Device Experience) Database Interface

This module provides functionality to query and analyze FDA MAUDE adverse
event reports, with special focus on Human Factors Engineering (HFE)
classification for medical device incidents.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class EventType(Enum):
    """MAUDE event types."""
    DEATH = "D"
    INJURY = "I" 
    MALFUNCTION = "M"
    OTHER = "O"


class HFEClassification(Enum):
    """Human Factors Engineering classification for device incidents."""
    DEVICE_FAILURE = "device_failure"           # Clear device/system failure
    CLEAR_USE_ERROR = "clear_use_error"         # Obvious user error
    DESIGN_INDUCED_USE_ERROR = "design_induced_use_error"  # Design-related use error
    UNCLEAR = "unclear"                         # Cannot determine root cause
    

@dataclass
class AdverseEvent:
    """Represents a MAUDE adverse event report."""
    mdr_report_key: str
    event_date: Optional[datetime]
    event_type: EventType
    device_name: str
    manufacturer: str
    brand_name: Optional[str]
    generic_name: Optional[str]
    product_code: str
    device_class: Optional[str]
    
    # Event details
    event_description: str
    device_problem: Optional[str]
    patient_problem: Optional[str]
    
    # Reporter information
    reporter_occupation: Optional[str]
    report_source: Optional[str]
    
    # HFE Classification (will be populated by analysis)
    hfe_classification: Optional[HFEClassification] = None
    hfe_confidence: Optional[float] = None
    
    # Additional metadata
    date_received: Optional[datetime] = None
    report_number: Optional[str] = None


class MAUDEInterface:
    """Interface for querying FDA MAUDE database."""
    
    # MAUDE database URLs
    BASE_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/search.cfm"
    DOWNLOAD_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/TextSearch.cfm"
    
    # HFE classification keywords
    HFE_KEYWORDS = {
        HFEClassification.DEVICE_FAILURE: [
            'broke', 'cracked', 'fractured', 'failed', 'malfunction', 'defective',
            'manufacturing defect', 'material failure', 'component failure',
            'device error', 'system failure', 'mechanical failure', 'broken'
        ],
        HFEClassification.CLEAR_USE_ERROR: [
            'user error', 'incorrect use', 'misuse', 'wrong procedure',
            'operator error', 'improper installation', 'incorrect technique',
            'user did not follow', 'failed to follow instructions'
        ],
        HFEClassification.DESIGN_INDUCED_USE_ERROR: [
            'confusing', 'unclear instructions', 'poor visibility',
            'difficult to use', 'misleading', 'counterintuitive',
            'design flaw', 'poor design', 'confusing interface',
            'inadequate labeling', 'similar appearance'
        ]
    }
    
    def __init__(self, session_timeout: int = 30):
        self.session_timeout = session_timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TracePredicate-Research/0.1.0 (Medical Device Analysis)'
        })
    
    def search_events_by_product_code(
        self, 
        product_code: str, 
        start_date: datetime = None,
        end_date: datetime = None,
        event_types: List[EventType] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for adverse events by product code.
        
        Args:
            product_code: FDA product code (e.g., 'KWA')
            start_date: Start date for search
            end_date: End date for search
            event_types: List of event types to include
            limit: Maximum number of results
            
        Returns:
            List of adverse event dictionaries
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*5)  # Last 5 years
        if end_date is None:
            end_date = datetime.now()
        if event_types is None:
            event_types = [EventType.DEATH, EventType.INJURY, EventType.MALFUNCTION]
            
        logger.info(f"Searching MAUDE events for product code: {product_code}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Prepare search parameters
        search_params = {
            'ProductCode': product_code,
            'StartDate': start_date.strftime('%m/%d/%Y'),
            'EndDate': end_date.strftime('%m/%d/%Y'),
            'EventType': ''.join([et.value for et in event_types]),
            'MaxRecords': str(limit) if limit else '10000',
            'search': 'Search'
        }
        
        try:
            response = self.session.post(self.BASE_URL, data=search_params, timeout=self.session_timeout)
            response.raise_for_status()
            
            events = self._parse_search_results(response.text)
            logger.info(f"Found {len(events)} adverse events")
            return events
            
        except requests.RequestException as e:
            logger.error(f"Error searching MAUDE database: {e}")
            return []
    
    def search_events_by_device_name(
        self, 
        device_names: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for adverse events by device name(s).
        
        Args:
            device_names: List of device names to search for
            start_date: Start date for search  
            end_date: End date for search
            limit: Maximum number of results
            
        Returns:
            List of adverse event dictionaries
        """
        all_events = []
        
        for device_name in device_names:
            logger.info(f"Searching events for device: {device_name}")
            
            search_params = {
                'DeviceName': device_name,
                'StartDate': start_date.strftime('%m/%d/%Y') if start_date else '',
                'EndDate': end_date.strftime('%m/%d/%Y') if end_date else '',
                'MaxRecords': str(limit) if limit else '1000',
                'search': 'Search'
            }
            
            try:
                response = self.session.post(self.BASE_URL, data=search_params, timeout=self.session_timeout)
                response.raise_for_status()
                
                events = self._parse_search_results(response.text)
                all_events.extend(events)
                
            except requests.RequestException as e:
                logger.error(f"Error searching for device {device_name}: {e}")
                continue
        
        # Remove duplicates based on MDR report key
        seen_keys = set()
        unique_events = []
        for event in all_events:
            if event['mdr_report_key'] not in seen_keys:
                seen_keys.add(event['mdr_report_key'])
                unique_events.append(event)
        
        logger.info(f"Found {len(unique_events)} unique adverse events")
        return unique_events
    
    def _parse_search_results(self, html_content: str) -> List[Dict]:
        """Parse MAUDE search results."""
        soup = BeautifulSoup(html_content, 'html.parser')
        events = []
        
        # Find results table
        tables = soup.find_all('table')
        if not tables:
            return events
            
        # Look for the main results table
        results_table = None
        for table in tables:
            headers = table.find_all('th')
            if headers and any('MDR Report Key' in th.get_text() for th in headers):
                results_table = table
                break
        
        if not results_table:
            logger.warning("No results table found")
            return events
        
        rows = results_table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 8:  # Minimum expected columns
                try:
                    event_data = {
                        'mdr_report_key': cols[0].get_text(strip=True),
                        'event_date': self._parse_date(cols[1].get_text(strip=True)),
                        'event_type': cols[2].get_text(strip=True),
                        'device_name': cols[3].get_text(strip=True),
                        'manufacturer': cols[4].get_text(strip=True),
                        'brand_name': cols[5].get_text(strip=True) if len(cols) > 5 else None,
                        'generic_name': cols[6].get_text(strip=True) if len(cols) > 6 else None,
                        'product_code': cols[7].get_text(strip=True) if len(cols) > 7 else None,
                    }
                    
                    events.append(event_data)
                    
                except Exception as e:
                    logger.warning(f"Error parsing event row: {e}")
                    continue
        
        return events
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse date string from MAUDE database."""
        if not date_string or date_string.lower() in ['n/a', 'unknown', '']:
            return None
            
        # Common MAUDE date formats
        formats = ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%B %d, %Y', '%m/%d/%y']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue
                
        logger.debug(f"Could not parse date: {date_string}")
        return None
    
    def get_event_details(self, mdr_report_key: str) -> Optional[AdverseEvent]:
        """
        Get detailed information for a specific adverse event.
        
        Args:
            mdr_report_key: Unique MDR report identifier
            
        Returns:
            AdverseEvent object with full details
        """
        detail_url = f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/detail.cfm?mdrfoi__id={mdr_report_key}"
        
        try:
            response = self.session.get(detail_url, timeout=self.session_timeout)
            response.raise_for_status()
            
            return self._parse_event_details(response.text, mdr_report_key)
            
        except requests.RequestException as e:
            logger.error(f"Error getting event details for {mdr_report_key}: {e}")
            return None
    
    def _parse_event_details(self, html_content: str, mdr_report_key: str) -> Optional[AdverseEvent]:
        """Parse detailed event information."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract various fields from the detail page
        # This is a simplified parser - the actual MAUDE detail page structure may vary
        
        try:
            # Find all text areas and input fields
            event_description = ""
            device_problem = ""
            patient_problem = ""
            
            # Look for description sections
            text_areas = soup.find_all(['textarea', 'td'])
            for area in text_areas:
                text = area.get_text(strip=True)
                if len(text) > 50:  # Likely a description
                    if 'event' in text.lower() or 'description' in text.lower():
                        event_description = text
                    elif 'device' in text.lower() and 'problem' in text.lower():
                        device_problem = text
                    elif 'patient' in text.lower() and 'problem' in text.lower():
                        patient_problem = text
            
            # Create basic event object
            # Note: This is simplified - actual parsing would need more sophisticated field extraction
            event = AdverseEvent(
                mdr_report_key=mdr_report_key,
                event_date=None,  # Would need to parse from detail page
                event_type=EventType.MALFUNCTION,  # Default, would parse from page
                device_name="",
                manufacturer="",
                brand_name=None,
                generic_name=None,
                product_code="",
                device_class=None,
                event_description=event_description,
                device_problem=device_problem,
                patient_problem=patient_problem,
                reporter_occupation=None,
                report_source=None
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error parsing event details: {e}")
            return None
    
    def classify_hfe_events(self, events: List[AdverseEvent]) -> List[AdverseEvent]:
        """
        Classify adverse events using Human Factors Engineering approach.
        
        Args:
            events: List of AdverseEvent objects
            
        Returns:
            List of events with HFE classification
        """
        logger.info(f"Classifying {len(events)} events using HFE approach")
        
        classified_events = []
        
        for event in events:
            # Combine all available text for analysis
            text_to_analyze = " ".join([
                event.event_description or "",
                event.device_problem or "",
                event.patient_problem or ""
            ]).lower()
            
            # Score each classification category
            scores = {}
            for classification, keywords in self.HFE_KEYWORDS.items():
                score = 0
                for keyword in keywords:
                    if keyword.lower() in text_to_analyze:
                        score += 1
                scores[classification] = score
            
            # Assign classification based on highest score
            if max(scores.values()) == 0:
                event.hfe_classification = HFEClassification.UNCLEAR
                event.hfe_confidence = 0.0
            else:
                best_classification = max(scores, key=scores.get)
                event.hfe_classification = best_classification
                event.hfe_confidence = scores[best_classification] / len(self.HFE_KEYWORDS[best_classification])
            
            classified_events.append(event)
        
        # Log classification results
        classification_counts = {}
        for event in classified_events:
            classification_counts[event.hfe_classification] = classification_counts.get(event.hfe_classification, 0) + 1
        
        logger.info("HFE Classification results:")
        for classification, count in classification_counts.items():
            logger.info(f"  {classification.value}: {count}")
        
        return classified_events
    
    def calculate_event_rates(self, events: List[AdverseEvent], device_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate adverse event rates per device.
        
        Args:
            events: List of classified events
            device_counts: Dictionary of device names to number deployed/sold
            
        Returns:
            Dictionary of device names to event rates
        """
        device_events = {}
        
        # Count events per device
        for event in events:
            device_name = event.device_name
            if device_name not in device_events:
                device_events[device_name] = 0
            device_events[device_name] += 1
        
        # Calculate rates
        event_rates = {}
        for device_name, event_count in device_events.items():
            if device_name in device_counts and device_counts[device_name] > 0:
                rate = (event_count / device_counts[device_name]) * 1000  # Events per 1000 devices
                event_rates[device_name] = rate
        
        return event_rates
    
    def export_to_dataframe(self, events: List[AdverseEvent]) -> pd.DataFrame:
        """Export events to pandas DataFrame for analysis."""
        data = []
        
        for event in events:
            data.append({
                'mdr_report_key': event.mdr_report_key,
                'event_date': event.event_date,
                'event_type': event.event_type.value if event.event_type else None,
                'device_name': event.device_name,
                'manufacturer': event.manufacturer,
                'brand_name': event.brand_name,
                'generic_name': event.generic_name,
                'product_code': event.product_code,
                'device_class': event.device_class,
                'event_description': event.event_description,
                'device_problem': event.device_problem,
                'patient_problem': event.patient_problem,
                'hfe_classification': event.hfe_classification.value if event.hfe_classification else None,
                'hfe_confidence': event.hfe_confidence,
                'reporter_occupation': event.reporter_occupation,
                'report_source': event.report_source,
                'date_received': event.date_received
            })
        
        return pd.DataFrame(data)


def main():
    """Example usage of MAUDE interface."""
    maude = MAUDEInterface()
    
    # Search for hip implant adverse events
    events_data = maude.search_events_by_product_code("KWA", limit=100)
    print(f"Found {len(events_data)} adverse events")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(events_data)
    if not df.empty:
        print(df[['mdr_report_key', 'device_name', 'event_type']].head())
    

if __name__ == "__main__":
    main()