"""
FDA 510(k) Database Scraper

This module provides functionality to scrape and parse FDA 510(k) clearance
documents, focusing on hip implants (KWA device code) and extracting
predicate device relationships.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pdfplumber
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class Device510K:
    """Represents a 510(k) device clearance."""
    k_number: str
    device_name: str
    applicant: str
    approval_date: datetime
    device_code: str
    product_code: str
    predicate_devices: List[str]
    decision: str
    summary_url: Optional[str] = None
    pdf_content: Optional[str] = None
    technical_parameters: Dict[str, str] = None
    
    def __post_init__(self):
        if self.technical_parameters is None:
            self.technical_parameters = {}


class FDA510KScraper:
    """Scraper for FDA 510(k) clearance database."""
    
    BASE_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"
    SEARCH_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"
    
    def __init__(self, session_timeout: int = 30, max_concurrent: int = 5):
        self.session_timeout = session_timeout
        self.max_concurrent = max_concurrent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TracePredicate-Research/0.1.0 (Medical Device Analysis)'
        })
    
    def search_devices_by_code(self, device_code: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Search for devices by product code (e.g., 'KWA' for hip implants).
        
        Args:
            device_code: FDA product code (e.g., 'KWA')
            limit: Maximum number of results to return
            
        Returns:
            List of device information dictionaries
        """
        logger.info(f"Searching for devices with code: {device_code}")
        
        search_params = {
            'start': '1',
            'sortby': '1',
            'showrecords': '100',  # Max per page
            'ProductCode': device_code,
            'DeviceClass': '',
            'search': 'Search'
        }
        
        all_devices = []
        page = 1
        
        while True:
            search_params['start'] = str((page - 1) * 100 + 1)
            
            try:
                response = self.session.post(self.SEARCH_URL, data=search_params, timeout=self.session_timeout)
                response.raise_for_status()
                
                devices_on_page = self._parse_search_results(response.text)
                
                if not devices_on_page:
                    logger.info(f"No more results found on page {page}")
                    break
                
                all_devices.extend(devices_on_page)
                logger.info(f"Found {len(devices_on_page)} devices on page {page}")
                
                if limit and len(all_devices) >= limit:
                    all_devices = all_devices[:limit]
                    break
                
                page += 1
                
            except requests.RequestException as e:
                logger.error(f"Error during search on page {page}: {e}")
                break
        
        logger.info(f"Total devices found: {len(all_devices)}")
        return all_devices
    
    def _parse_search_results(self, html_content: str) -> List[Dict]:
        """Parse search results from FDA 510(k) search page."""
        soup = BeautifulSoup(html_content, 'html.parser')
        devices = []
        
        # Find the results table
        tables = soup.find_all('table', {'class': 'PMNTable'})
        if not tables:
            return devices
            
        # Skip header row
        rows = tables[0].find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 6:
                try:
                    k_number = cols[0].get_text(strip=True)
                    device_name = cols[1].get_text(strip=True)
                    applicant = cols[2].get_text(strip=True)
                    
                    # Parse approval date
                    date_text = cols[3].get_text(strip=True)
                    approval_date = self._parse_date(date_text)
                    
                    decision = cols[4].get_text(strip=True)
                    product_code = cols[5].get_text(strip=True) if len(cols) > 5 else ""
                    
                    # Extract summary URL if available
                    summary_url = None
                    link = cols[0].find('a')
                    if link and link.get('href'):
                        summary_url = urljoin(self.BASE_URL, link['href'])
                    
                    devices.append({
                        'k_number': k_number,
                        'device_name': device_name,
                        'applicant': applicant,
                        'approval_date': approval_date,
                        'decision': decision,
                        'product_code': product_code,
                        'summary_url': summary_url
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
        
        return devices
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse date string in various formats."""
        if not date_string or date_string.lower() in ['n/a', 'pending']:
            return None
            
        # Common FDA date formats
        formats = ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%B %d, %Y']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue
                
        logger.warning(f"Could not parse date: {date_string}")
        return None
    
    async def get_device_details(self, device_info: Dict) -> Device510K:
        """
        Get detailed information for a specific device including predicate analysis.
        
        Args:
            device_info: Basic device information from search results
            
        Returns:
            Device510K object with full details
        """
        device = Device510K(
            k_number=device_info['k_number'],
            device_name=device_info['device_name'],
            applicant=device_info['applicant'],
            approval_date=device_info['approval_date'],
            device_code=device_info['product_code'],
            product_code=device_info['product_code'],
            predicate_devices=[],
            decision=device_info['decision'],
            summary_url=device_info.get('summary_url')
        )
        
        # Get summary page content and extract predicates
        if device.summary_url:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.session_timeout)) as session:
                    async with session.get(device.summary_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            device.predicate_devices = self._extract_predicate_devices(content)
                            
                            # Try to get PDF content if available
                            pdf_url = self._extract_pdf_url(content)
                            if pdf_url:
                                device.pdf_content = await self._download_and_parse_pdf(session, pdf_url)
                                
            except Exception as e:
                logger.error(f"Error getting details for {device.k_number}: {e}")
        
        return device
    
    def _extract_predicate_devices(self, html_content: str) -> List[str]:
        """Extract predicate device K-numbers from summary page."""
        soup = BeautifulSoup(html_content, 'html.parser')
        predicates = []
        
        # Look for K-numbers in various sections
        text = soup.get_text()
        
        # Pattern to match K-numbers (K followed by 6 digits)
        k_pattern = re.compile(r'K(\d{6})', re.IGNORECASE)
        matches = k_pattern.findall(text)
        
        # Also look for "predicate" sections specifically
        predicate_sections = soup.find_all(text=re.compile(r'predicate', re.IGNORECASE))
        
        for section in predicate_sections:
            parent = section.parent
            if parent:
                section_text = parent.get_text()
                section_matches = k_pattern.findall(section_text)
                matches.extend(section_matches)
        
        # Clean and deduplicate
        for match in set(matches):
            k_number = f"K{match}"
            predicates.append(k_number)
        
        logger.debug(f"Found predicate devices: {predicates}")
        return predicates
    
    def _extract_pdf_url(self, html_content: str) -> Optional[str]:
        """Extract PDF document URL from summary page."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for PDF links
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
        
        if pdf_links:
            pdf_url = urljoin(self.BASE_URL, pdf_links[0]['href'])
            return pdf_url
            
        return None
    
    async def _download_and_parse_pdf(self, session: aiohttp.ClientSession, pdf_url: str) -> Optional[str]:
        """Download and parse PDF content."""
        try:
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    pdf_content = await response.read()
                    
                    # Save temporarily and parse
                    temp_path = Path(f"/tmp/temp_510k.pdf")
                    temp_path.write_bytes(pdf_content)
                    
                    # Try pdfplumber first, fallback to PyPDF2
                    text_content = self._parse_pdf_with_pdfplumber(temp_path)
                    if not text_content:
                        text_content = self._parse_pdf_with_pypdf2(temp_path)
                    
                    # Clean up
                    temp_path.unlink(missing_ok=True)
                    
                    return text_content
                    
        except Exception as e:
            logger.error(f"Error downloading/parsing PDF {pdf_url}: {e}")
            
        return None
    
    def _parse_pdf_with_pdfplumber(self, pdf_path: Path) -> Optional[str]:
        """Parse PDF using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text.strip()
        except Exception as e:
            logger.debug(f"pdfplumber parsing failed: {e}")
            return None
    
    def _parse_pdf_with_pypdf2(self, pdf_path: Path) -> Optional[str]:
        """Parse PDF using PyPDF2 as fallback."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text.strip()
        except Exception as e:
            logger.debug(f"PyPDF2 parsing failed: {e}")
            return None
    
    async def collect_device_family(self, device_code: str, limit: Optional[int] = None) -> List[Device510K]:
        """
        Collect a family of devices with full details.
        
        Args:
            device_code: FDA product code
            limit: Maximum number of devices to collect
            
        Returns:
            List of Device510K objects with full details
        """
        logger.info(f"Collecting device family for code: {device_code}")
        
        # First get basic device list
        device_list = self.search_devices_by_code(device_code, limit)
        
        # Then get detailed information for each device
        devices = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def get_device_with_semaphore(device_info):
            async with semaphore:
                return await self.get_device_details(device_info)
        
        tasks = [get_device_with_semaphore(device_info) for device_info in device_list]
        devices = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_devices = [device for device in devices if isinstance(device, Device510K)]
        
        logger.info(f"Successfully collected {len(valid_devices)} devices")
        return valid_devices
    
    def export_to_dataframe(self, devices: List[Device510K]) -> pd.DataFrame:
        """Export devices to pandas DataFrame for analysis."""
        data = []
        
        for device in devices:
            predicate_str = "|".join(device.predicate_devices) if device.predicate_devices else ""
            
            data.append({
                'k_number': device.k_number,
                'device_name': device.device_name,
                'applicant': device.applicant,
                'approval_date': device.approval_date,
                'device_code': device.device_code,
                'product_code': device.product_code,
                'decision': device.decision,
                'predicate_devices': predicate_str,
                'predicate_count': len(device.predicate_devices),
                'summary_url': device.summary_url,
                'has_pdf_content': device.pdf_content is not None
            })
        
        return pd.DataFrame(data)


async def main():
    """Example usage of the FDA 510(k) scraper."""
    scraper = FDA510KScraper()
    
    # Collect hip implant devices (KWA code)
    devices = await scraper.collect_device_family("KWA", limit=10)
    
    # Export to DataFrame
    df = scraper.export_to_dataframe(devices)
    print(f"Collected {len(df)} devices")
    print(df[['k_number', 'device_name', 'predicate_count']].head())


if __name__ == "__main__":
    asyncio.run(main())