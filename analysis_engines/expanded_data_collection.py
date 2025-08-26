#!/usr/bin/env python3
"""
Expanded Data Collection: High-Priority FDA Product Categories
==============================================================

Collects data for strategically selected high-value product categories:
- NIK: Pacemaker Pulse Generator (158,700 records)
- DTK: Coronary Stent (37,978 records)  
- MHX: Implantable Defibrillator (27,431 records)
- FDS: Endoscope (36,876 records)
- LZO: Surgical Robot (29,813 records)

This expands our dataset from 167,307 to ~400,000+ records.
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

class ExpandedDataCollector:
    """Collect high-priority additional FDA device categories"""
    
    def __init__(self):
        self.cache_dir = Path("data/real_fda_dataset")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path("results/expanded_dataset")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.fda.gov"
        self.rate_limit_delay = 1.5
        self.api_limit = 1000
        self.max_retries = 3
        
        # High-priority expansion categories
        self.expansion_categories = {
            # TIER 1: Life-Critical Devices (Must-Add)
            'NIK': 'Pacemaker Pulse Generator (Cardiovascular)',    # 158,700 records
            'DTK': 'Coronary Stent (Cardiovascular)',               # 37,978 records  
            'MHX': 'Implantable Defibrillator (Cardiovascular)',    # 27,431 records
            
            # TIER 2: Strategic High-Value
            'FDS': 'Endoscope (Surgical)',                          # 36,876 records
            'LZO': 'Surgical Robot (Surgical)',                     # 29,813 records
            
            # TIER 3: Specialized Research (Optional)
            'GDT': 'Insulin Pump (Endocrine)',                      # 4,656 records
            'GAL': 'Breast Prosthesis (Plastic Surgery)'            # 2,357 records
        }

    def fetch_all_paginated_data(self, endpoint: str, search_param: str, data_type: str) -> List[Dict]:
        """Fetch all data using pagination to bypass 1000-record limit"""
        
        all_results = []
        skip = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        total_available = None
        
        logger.info(f"  🔄 Collecting {data_type} data with pagination...")
        
        while consecutive_errors < max_consecutive_errors:
            try:
                url = f"{self.base_url}/{endpoint}"
                params = {
                    'search': search_param,
                    'limit': self.api_limit,
                    'skip': skip
                }
                
                logger.info(f"    📡 API call: skip={skip}, limit={self.api_limit}")
                
                response = requests.get(url, params=params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if total_available is None:
                        total_available = data['meta']['results']['total']
                        logger.info(f"    📊 Found total {data_type} records: {total_available:,}")
                        
                        if total_available == 0:
                            logger.info(f"    ℹ️  No {data_type} data for this category")
                            break
                    
                    if not results:
                        logger.info(f"    ✅ Retrieved all {data_type} data")
                        break
                    
                    all_results.extend(results)
                    consecutive_errors = 0
                    
                    collected_count = len(all_results)
                    progress = (collected_count / total_available) * 100 if total_available > 0 else 100
                    logger.info(f"    📈 Progress: {collected_count:,}/{total_available:,} ({progress:.1f}%)")
                    
                    skip += self.api_limit
                    
                    if collected_count >= total_available:
                        logger.info(f"    🎉 Completed collection of all {total_available:,} {data_type} records")
                        break
                
                elif response.status_code == 404:
                    logger.info(f"    ℹ️  API returned 404, no {data_type} data for this category")
                    break
                else:
                    consecutive_errors += 1
                    logger.warning(f"    ⚠️  API error {response.status_code} (attempt {consecutive_errors}/{max_consecutive_errors})")
                    
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"    ❌ Unexpected error: {e} (attempt {consecutive_errors}/{max_consecutive_errors})")
                if consecutive_errors < max_consecutive_errors:
                    time.sleep(self.rate_limit_delay * 2)
                
        logger.info(f"  ✅ Final {data_type} collection: {len(all_results):,} records")
        return all_results

    def collect_category_data(self, product_code: str, category_name: str) -> Dict:
        """Collect complete dataset for a single category"""
        
        logger.info(f"\n🔍 Starting data collection: {product_code} ({category_name})")
        
        category_data = {
            'product_code': product_code,
            'category_name': category_name,
            'collection_timestamp': datetime.now().isoformat(),
            '510k_data': [],
            'maude_data': [],
            'recall_data': []
        }
        
        # 1. Collect 510(k) data
        logger.info(f"📋 Collecting 510(k) data...")
        category_data['510k_data'] = self.fetch_all_paginated_data(
            endpoint='device/510k.json',
            search_param=f'product_code:{product_code}',
            data_type='510(k)'
        )
        
        # 2. Collect MAUDE data
        logger.info(f"⚠️  Collecting MAUDE data...")
        category_data['maude_data'] = self.fetch_all_paginated_data(
            endpoint='device/event.json',
            search_param=f'device.device_report_product_code:{product_code}',
            data_type='MAUDE'
        )
        
        # 3. Collect Recall data
        logger.info(f"🚨 Collecting Recall data...")
        category_data['recall_data'] = self.fetch_all_paginated_data(
            endpoint='device/recall.json',
            search_param=f'product_code:{product_code}',
            data_type='Recall'
        )
        
        # Save category data
        output_file = self.cache_dir / f"{product_code}_complete_data.json.gz"
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(category_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved complete data: {output_file}")
        
        return {
            'product_code': product_code,
            'category_name': category_name,
            '510k_count': len(category_data['510k_data']),
            'maude_count': len(category_data['maude_data']),
            'recall_count': len(category_data['recall_data']),
            'total_count': len(category_data['510k_data']) + len(category_data['maude_data']) + len(category_data['recall_data']),
            'collection_timestamp': category_data['collection_timestamp']
        }

    def run_expansion_collection(self, tier_level: int = 1):
        """Run the expanded data collection
        
        Args:
            tier_level: 1=Tier 1 only, 2=Tier 1+2, 3=All tiers
        """
        
        logger.info("🚀 Starting Expanded FDA Data Collection")
        logger.info(f"🎯 Target: Collect Tier {tier_level} high-priority device categories")
        
        collection_summary = {
            'expansion_timestamp': datetime.now().isoformat(),
            'tier_level': tier_level,
            'categories_processed': 0,
            'successful_categories': 0,
            'total_510k_records': 0,
            'total_maude_records': 0,
            'total_recall_records': 0,
            'category_summaries': {}
        }
        
        # Select categories based on tier level
        if tier_level == 1:
            categories = {k: v for k, v in list(self.expansion_categories.items())[:3]}  # NIK, DTK, MHX
        elif tier_level == 2:
            # For Tier 2, only collect the NEW categories (FDS, LZO) not already collected in Tier 1
            all_tier2 = {k: v for k, v in list(self.expansion_categories.items())[:5]}
            tier1_codes = set(list(self.expansion_categories.keys())[:3])
            categories = {k: v for k, v in all_tier2.items() if k not in tier1_codes}
        else:
            categories = self.expansion_categories  # All categories
        
        logger.info(f"📊 Selected {len(categories)} categories for collection")
        
        for product_code, category_name in categories.items():
            try:
                collection_summary['categories_processed'] += 1
                
                # Check if already collected
                existing_file = self.cache_dir / f"{product_code}_complete_data.json.gz"
                if existing_file.exists():
                    logger.info(f"⏭️  Skipping {product_code} - already collected")
                    continue
                
                category_result = self.collect_category_data(product_code, category_name)
                
                collection_summary['successful_categories'] += 1
                collection_summary['total_510k_records'] += category_result['510k_count']
                collection_summary['total_maude_records'] += category_result['maude_count']
                collection_summary['total_recall_records'] += category_result['recall_count']
                collection_summary['category_summaries'][product_code] = category_result
                
                logger.info(f"✅ Completed {product_code}: {category_result['total_count']:,} total records")
                
            except Exception as e:
                logger.error(f"❌ Failed to collect {product_code}: {e}")
                continue
        
        # Save expansion summary
        grand_total = (collection_summary['total_510k_records'] + 
                      collection_summary['total_maude_records'] + 
                      collection_summary['total_recall_records'])
        
        collection_summary['grand_total_records'] = grand_total
        
        summary_file = self.results_dir / f"expansion_tier{tier_level}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🎉 EXPANSION COLLECTION COMPLETED!")
        logger.info(f"📊 Total new records: {grand_total:,}")
        logger.info(f"📁 Summary saved: {summary_file}")
        
        return collection_summary

if __name__ == "__main__":
    collector = ExpandedDataCollector()
    
    print("🎯 TracePredicate Expanded Data Collection")
    print("==========================================")
    print()
    print("Tier 1 (Must-Add): NIK, DTK, MHX (~224,000 records)")
    print("Tier 2 (Strategic): + FDS, LZO (~291,000 records)")  
    print("Tier 3 (Complete): + GDT, GAL (~298,000 records)")
    print()
    print("🎆 Starting automatic Tier 1 collection (high-priority life-critical devices)")
    
    # Start with Tier 1 - most critical devices
    tier = 1
    
    result = collector.run_expansion_collection(tier_level=tier)
    
    print(f"\n✅ Tier 1 Collection completed: {result['grand_total_records']:,} new records!")
    print("🎯 Ready to update analysis pipeline with expanded dataset")
    
    # Also run Tier 2 for strategic categories
    print("\n🚀 Proceeding with Tier 2 collection (strategic categories)...")
    result2 = collector.run_expansion_collection(tier_level=2)
    print(f"\n✅ Tier 2 Collection completed: {result2['grand_total_records']:,} total new records!")