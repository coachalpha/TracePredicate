#!/usr/bin/env python3
"""
Complete Existing Categories Data Collection
============================================

Enhance data collection for our existing 13 categories by:
1. Completing incomplete collections (especially those stopped at 26,000)
2. Getting 50k+ records for infusion pumps (FRN) instead of 26,000
3. Ensuring all manageable categories have complete data collection
"""

import requests
import json
import gzip
from pathlib import Path
import logging
import time
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteExistingCollector:
    """Complete data collection for existing categories"""
    
    def __init__(self):
        self.cache_dir = Path("data/real_fda_dataset")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path("results/complete_collection")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.fda.gov"
        self.rate_limit_delay = 1.5
        self.api_limit = 1000
        self.max_retries = 3
        
        # Existing categories that need completion
        self.existing_categories = {
            # Orthopedic Devices - Some may be incomplete
            'KWA': 'Hip Prostheses (Orthopedic)',            # Had 26,000, should have ~137,000
            'KWP': 'Knee Prostheses (Orthopedic)',           # Had 13,964, check for more
            'KWF': 'Shoulder Prostheses (Orthopedic)',       # Had 218, likely complete
            'HRS': 'Bone Plates/Screws (Orthopedic)',        # Had 26,000, should have 36,619
            'HWC': 'Bone Drill (Orthopedic)',                # Had 26,000, check total available
            
            # Imaging Devices - Likely complete
            'LNH': 'MRI Systems (Radiology)',                # Had 4,188, check completeness
            'IYE': 'Ultrasound Systems (Radiology)',         # Had 3,022, check completeness
            'JAK': 'X-ray Systems (Radiology)',              # Had 5,028, check completeness
            
            # Life-Critical Support Devices - PRIORITY
            'FRN': 'Infusion Pumps (Critical Care)',         # Had 26,000, get 50,000+ (1.77M available)
            'BTO': 'Ventilators (Critical Care)',            # Had 6,360, check completeness
            
            # Other Categories
            'DQO': 'Catheters (Cardiovascular)',             # Had 19,308, check completeness
            'ETA': 'Hearing Aids (ENT)',                     # Had 92, likely complete
            'IOL': 'Intraocular Lenses (Ophthalmic)'         # Had 155, likely complete
        }
        
        # Special handling for FRN - get 50,000 records instead of full 1.77M
        self.special_limits = {
            'FRN': 50000  # Statistically sufficient sample of infusion pumps
        }

    def check_current_collection_status(self):
        """Check what we currently have collected"""
        
        logger.info("📊 Checking current collection status...")
        
        current_summary = {}
        
        for product_code, category_name in self.existing_categories.items():
            data_file = self.cache_dir / f"{product_code}_complete_data.json.gz"
            
            if data_file.exists():
                try:
                    with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    current_summary[product_code] = {
                        'category_name': category_name,
                        '510k_count': len(data.get('510k_data', [])),
                        'maude_count': len(data.get('maude_data', [])),
                        'recall_count': len(data.get('recall_data', [])),
                        'total_count': (len(data.get('510k_data', [])) + 
                                      len(data.get('maude_data', [])) + 
                                      len(data.get('recall_data', []))),
                        'collection_timestamp': data.get('collection_timestamp', 'unknown')
                    }
                    
                    logger.info(f"  {product_code}: {current_summary[product_code]['total_count']:,} records")
                    
                except Exception as e:
                    logger.error(f"  {product_code}: Error reading file - {e}")
                    current_summary[product_code] = {'status': 'error'}
            else:
                logger.info(f"  {product_code}: No data file found")
                current_summary[product_code] = {'status': 'missing'}
        
        return current_summary

    def check_api_availability(self, product_code: str, endpoint: str, search_param: str) -> int:
        """Check how many records are available in the API for a category"""
        
        try:
            url = f"{self.base_url}/{endpoint}"
            params = {
                'search': search_param,
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('meta', {}).get('results', {}).get('total', 0)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error checking API availability for {product_code}: {e}")
            return 0

    def fetch_enhanced_paginated_data(self, endpoint: str, search_param: str, data_type: str, 
                                     max_records: int = None) -> List[Dict]:
        """Enhanced pagination with optional record limits"""
        
        all_results = []
        skip = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        total_available = None
        
        logger.info(f"  🔄 Collecting {data_type} data with enhanced pagination...")
        if max_records:
            logger.info(f"    🎯 Target: {max_records:,} records (limited for statistical efficiency)")
        
        while consecutive_errors < max_consecutive_errors:
            try:
                url = f"{self.base_url}/{endpoint}"
                params = {
                    'search': search_param,
                    'limit': self.api_limit,
                    'skip': skip
                }
                
                response = requests.get(url, params=params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if total_available is None:
                        total_available = data['meta']['results']['total']
                        target_records = min(total_available, max_records) if max_records else total_available
                        logger.info(f"    📊 Available: {total_available:,}, Target: {target_records:,}")
                        
                        if total_available == 0:
                            logger.info(f"    ℹ️  No {data_type} data available")
                            break
                    
                    if not results:
                        logger.info(f"    ✅ Retrieved all available {data_type} data")
                        break
                    
                    all_results.extend(results)
                    consecutive_errors = 0
                    
                    collected_count = len(all_results)
                    target_records = min(total_available, max_records) if max_records else total_available
                    progress = (collected_count / target_records) * 100
                    logger.info(f"    📈 Progress: {collected_count:,}/{target_records:,} ({progress:.1f}%)")
                    
                    # Check if we've reached our target
                    if max_records and collected_count >= max_records:
                        logger.info(f"    🎯 Reached target of {max_records:,} records")
                        break
                    
                    if collected_count >= total_available:
                        logger.info(f"    🎉 Collected all {total_available:,} available records")
                        break
                    
                    skip += self.api_limit
                
                elif response.status_code == 404:
                    logger.info(f"    ℹ️  No {data_type} data available (404)")
                    break
                else:
                    consecutive_errors += 1
                    logger.warning(f"    ⚠️  API error {response.status_code} (attempt {consecutive_errors}/{max_consecutive_errors})")
                    
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"    ❌ Error: {e} (attempt {consecutive_errors}/{max_consecutive_errors})")
                if consecutive_errors < max_consecutive_errors:
                    time.sleep(self.rate_limit_delay * 2)
                
        logger.info(f"  ✅ Final {data_type} collection: {len(all_results):,} records")
        return all_results

    def enhance_category_collection(self, product_code: str, category_name: str, force_recollect: bool = False):
        """Enhance data collection for a single category"""
        
        logger.info(f"\n🔍 Enhancing collection: {product_code} ({category_name})")
        
        # Check if we should skip this category
        data_file = self.cache_dir / f"{product_code}_complete_data.json.gz"
        if data_file.exists() and not force_recollect:
            logger.info(f"  ⏭️  Skipping {product_code} - already exists (use force_recollect=True to override)")
            return None
        
        # Check API availability first
        maude_available = self.check_api_availability(
            product_code, 'device/event.json', 
            f'device.device_report_product_code:{product_code}'
        )
        
        logger.info(f"  📊 API Status - MAUDE: {maude_available:,} records available")
        
        category_data = {
            'product_code': product_code,
            'category_name': category_name,
            'collection_timestamp': datetime.now().isoformat(),
            'enhanced_collection': True,
            '510k_data': [],
            'maude_data': [],
            'recall_data': []
        }
        
        # 1. Collect 510(k) data
        logger.info(f"📋 Collecting 510(k) data...")
        category_data['510k_data'] = self.fetch_enhanced_paginated_data(
            endpoint='device/510k.json',
            search_param=f'product_code:{product_code}',
            data_type='510(k)'
        )
        
        # 2. Collect MAUDE data (with special limits if specified)
        logger.info(f"⚠️  Collecting MAUDE data...")
        max_maude = self.special_limits.get(product_code)
        category_data['maude_data'] = self.fetch_enhanced_paginated_data(
            endpoint='device/event.json',
            search_param=f'device.device_report_product_code:{product_code}',
            data_type='MAUDE',
            max_records=max_maude
        )
        
        # 3. Collect Recall data
        logger.info(f"🚨 Collecting Recall data...")
        category_data['recall_data'] = self.fetch_enhanced_paginated_data(
            endpoint='device/recall.json',
            search_param=f'product_code:{product_code}',
            data_type='Recall'
        )
        
        # Save enhanced data
        with gzip.open(data_file, 'wt', encoding='utf-8') as f:
            json.dump(category_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved enhanced data: {data_file}")
        
        return {
            'product_code': product_code,
            'category_name': category_name,
            '510k_count': len(category_data['510k_data']),
            'maude_count': len(category_data['maude_data']),
            'recall_count': len(category_data['recall_data']),
            'total_count': (len(category_data['510k_data']) + 
                          len(category_data['maude_data']) + 
                          len(category_data['recall_data'])),
            'collection_timestamp': category_data['collection_timestamp']
        }

    def run_complete_enhancement(self, priority_categories: List[str] = None):
        """Run complete enhancement for existing categories"""
        
        logger.info("🚀 Starting Complete Data Collection Enhancement")
        logger.info("🎯 Target: Complete all existing categories with enhanced collection")
        
        # First check current status
        current_status = self.check_current_collection_status()
        
        # Determine which categories to enhance
        if priority_categories:
            categories_to_process = {k: v for k, v in self.existing_categories.items() 
                                   if k in priority_categories}
        else:
            # Focus on categories that are incomplete or missing
            categories_to_process = {}
            for code, name in self.existing_categories.items():
                status = current_status.get(code, {})
                if (status.get('status') == 'missing' or 
                    status.get('maude_count', 0) == 26000 or  # Likely incomplete
                    code in ['FRN']):  # Special handling for infusion pumps
                    categories_to_process[code] = name
        
        logger.info(f"📊 Processing {len(categories_to_process)} categories needing enhancement")
        
        enhancement_summary = {
            'enhancement_timestamp': datetime.now().isoformat(),
            'categories_processed': 0,
            'successful_enhancements': 0,
            'total_510k_records': 0,
            'total_maude_records': 0,
            'total_recall_records': 0,
            'category_summaries': {}
        }
        
        for product_code, category_name in categories_to_process.items():
            try:
                enhancement_summary['categories_processed'] += 1
                
                result = self.enhance_category_collection(product_code, category_name, force_recollect=True)
                
                if result:
                    enhancement_summary['successful_enhancements'] += 1
                    enhancement_summary['total_510k_records'] += result['510k_count']
                    enhancement_summary['total_maude_records'] += result['maude_count']
                    enhancement_summary['total_recall_records'] += result['recall_count']
                    enhancement_summary['category_summaries'][product_code] = result
                    
                    logger.info(f"✅ Enhanced {product_code}: {result['total_count']:,} total records")
                
            except Exception as e:
                logger.error(f"❌ Failed to enhance {product_code}: {e}")
                continue
        
        # Save enhancement summary
        grand_total = (enhancement_summary['total_510k_records'] + 
                      enhancement_summary['total_maude_records'] + 
                      enhancement_summary['total_recall_records'])
        
        enhancement_summary['grand_total_records'] = grand_total
        
        summary_file = self.results_dir / "existing_categories_enhancement_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(enhancement_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🎉 ENHANCEMENT COMPLETED!")
        logger.info(f"📊 Enhanced records: {grand_total:,}")
        logger.info(f"📁 Summary saved: {summary_file}")
        
        return enhancement_summary

if __name__ == "__main__":
    collector = CompleteExistingCollector()
    
    print("🎯 TracePredicate Complete Existing Categories Enhancement")
    print("=========================================================")
    print()
    print("🚀 Starting automatic enhancement of incomplete categories...")
    print("   Priority: FRN (50k records), KWA, HRS (complete collection)")
    
    # Focus on priority categories that need enhancement
    priority_codes = ['FRN', 'KWA', 'HRS', 'HWC']  # Most important incomplete ones
    
    result = collector.run_complete_enhancement(priority_categories=priority_codes)
    
    print(f"\n✅ Enhancement completed: {result['grand_total_records']:,} records enhanced!")