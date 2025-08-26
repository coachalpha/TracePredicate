#!/usr/bin/env python3
"""
Maximum Collection Strategy: Optimal Additional Categories
=========================================================

Collect the highest-value additional FDA device categories to maximize
our dataset within API constraints. Focus on categories that provide
maximum research value with good data volumes.
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

class MaximumCollectionStrategy:
    """Collect maximum value additional categories within API constraints"""
    
    def __init__(self):
        self.cache_dir = Path("data/real_fda_dataset")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path("results/maximum_collection")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.fda.gov"
        self.rate_limit_delay = 1.5
        self.api_limit = 1000
        self.max_retries = 3
        
        # Maximum value additional categories (ordered by priority)
        self.maximum_value_categories = {
            # TIER 1: Life-Critical Devices (Will hit 26k limit)
            'MHX': {
                'name': 'Implantable Defibrillator (Cardiovascular)',
                'available_records': 27431,
                'expected_collection': 26000,
                'strategic_value': 5,  # 1-5 scale
                'priority': 1,
                'rationale': 'Life-critical cardiac device, completes cardiovascular trio with NIK/DTK'
            },
            'KDI': {
                'name': 'Artificial Heart (Cardiovascular)',
                'available_records': 40826,
                'expected_collection': 26000,
                'strategic_value': 5,
                'priority': 2,
                'rationale': 'Ultimate life-critical device, highest regulatory oversight'
            },
            
            # TIER 2: High-Volume Strategic (Will hit 26k limit)
            'FDS': {
                'name': 'Endoscope (Surgical)',
                'available_records': 36876,
                'expected_collection': 26000,
                'strategic_value': 4,
                'priority': 3,
                'rationale': 'FDA priority area (infection control), high clinical utilization'
            },
            'LZO': {
                'name': 'Surgical Robot (Surgical)',
                'available_records': 29813,
                'expected_collection': 26000,
                'strategic_value': 4,
                'priority': 4,
                'rationale': 'Emerging technology, evolving regulatory pathways'
            },
            'DYB': {
                'name': 'Blood Pressure Monitor (Cardiovascular)',
                'available_records': 35046,
                'expected_collection': 26000,
                'strategic_value': 3,
                'priority': 5,
                'rationale': 'High-volume diagnostic device, population health impact'
            },
            
            # TIER 3: Strategic Complete Collection (Full data possible)
            'MMI': {
                'name': 'Mammography System (Radiology)',
                'available_records': 18054,
                'expected_collection': 18054,
                'strategic_value': 4,
                'priority': 6,
                'rationale': 'Cancer screening, women\'s health priority'
            },
            
            # TIER 4: Research Value (Full collection)
            'LWJ': {
                'name': 'CT Scanner (Radiology)',
                'available_records': 4124,
                'expected_collection': 4124,
                'strategic_value': 3,
                'priority': 7,
                'rationale': 'Advanced imaging, completes radiology portfolio'
            },
            'MCH': {
                'name': 'Cochlear Implant (ENT)',
                'available_records': 468,
                'expected_collection': 468,
                'strategic_value': 3,
                'priority': 8,
                'rationale': 'Specialized implantable, quality of life device'
            }
        }

    def fetch_optimized_paginated_data(self, endpoint: str, search_param: str, data_type: str, 
                                     max_records: int = None) -> List[Dict]:
        """Optimized pagination with API limit awareness"""
        
        all_results = []
        skip = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        total_available = None
        
        logger.info(f"  🔄 Collecting {data_type} data (optimized for API limits)...")
        if max_records:
            logger.info(f"    🎯 Target: {max_records:,} records")
        
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
                    
                    # Log progress every 5,000 records or significant milestones
                    if collected_count % 5000 == 0 or progress >= 99:
                        logger.info(f"    📈 Progress: {collected_count:,}/{target_records:,} ({progress:.1f}%)")
                    
                    # Check API limit conditions
                    if collected_count >= 26000 and collected_count % 1000 == 0:
                        logger.info(f"    🚨 Approaching potential API limit at {collected_count:,} records")
                    
                    # Check if we've reached our target
                    if max_records and collected_count >= max_records:
                        logger.info(f"    🎯 Reached target of {max_records:,} records")
                        break
                    
                    if collected_count >= total_available:
                        logger.info(f"    🎉 Collected all {total_available:,} available records")
                        break
                    
                    skip += self.api_limit
                
                elif response.status_code == 400:
                    # This is the expected API limit - log and break
                    logger.info(f"    🚨 Hit FDA API limit at {len(all_results):,} records (expected ~26k limit)")
                    break
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

    def collect_optimal_category(self, product_code: str, category_info: Dict) -> Dict:
        """Collect data for a single optimal category"""
        
        category_name = category_info['name']
        expected_records = category_info['expected_collection']
        
        logger.info(f"\n🔍 Collecting Priority {category_info['priority']}: {product_code} ({category_name})")
        logger.info(f"  📊 Expected: {expected_records:,} records")
        logger.info(f"  🎯 Value: {category_info['strategic_value']}/5 - {category_info['rationale']}")
        
        # Check if already exists
        data_file = self.cache_dir / f"{product_code}_complete_data.json.gz"
        if data_file.exists():
            logger.info(f"  ⏭️  Skipping {product_code} - already collected")
            return None
        
        category_data = {
            'product_code': product_code,
            'category_name': category_name,
            'collection_timestamp': datetime.now().isoformat(),
            'maximum_collection_strategy': True,
            'expected_records': expected_records,
            'strategic_value': category_info['strategic_value'],
            'priority': category_info['priority'],
            'rationale': category_info['rationale'],
            '510k_data': [],
            'maude_data': [],
            'recall_data': []
        }
        
        # 1. Collect 510(k) data
        logger.info(f"📋 Collecting 510(k) data...")
        category_data['510k_data'] = self.fetch_optimized_paginated_data(
            endpoint='device/510k.json',
            search_param=f'product_code:{product_code}',
            data_type='510(k)'
        )
        
        # 2. Collect MAUDE data (main data source)
        logger.info(f"⚠️  Collecting MAUDE data...")
        category_data['maude_data'] = self.fetch_optimized_paginated_data(
            endpoint='device/event.json',
            search_param=f'device.device_report_product_code:{product_code}',
            data_type='MAUDE'
        )
        
        # 3. Collect Recall data
        logger.info(f"🚨 Collecting Recall data...")
        category_data['recall_data'] = self.fetch_optimized_paginated_data(
            endpoint='device/recall.json',
            search_param=f'product_code:{product_code}',
            data_type='Recall'
        )
        
        # Save collected data
        with gzip.open(data_file, 'wt', encoding='utf-8') as f:
            json.dump(category_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved: {data_file}")
        
        total_collected = (len(category_data['510k_data']) + 
                          len(category_data['maude_data']) + 
                          len(category_data['recall_data']))
        
        return {
            'product_code': product_code,
            'category_name': category_name,
            'priority': category_info['priority'],
            'strategic_value': category_info['strategic_value'],
            'expected_records': expected_records,
            'actual_collected': total_collected,
            '510k_count': len(category_data['510k_data']),
            'maude_count': len(category_data['maude_data']),
            'recall_count': len(category_data['recall_data']),
            'collection_timestamp': category_data['collection_timestamp']
        }

    def run_maximum_collection(self, priority_limit: int = None):
        """Execute maximum value collection strategy"""
        
        logger.info("🚀 Starting Maximum Collection Strategy")
        logger.info("🎯 Target: Collect highest-value additional categories within API limits")
        
        # Sort categories by priority
        sorted_categories = sorted(
            self.maximum_value_categories.items(),
            key=lambda x: x[1]['priority']
        )
        
        if priority_limit:
            sorted_categories = sorted_categories[:priority_limit]
            logger.info(f"📊 Limited to top {priority_limit} priority categories")
        
        logger.info(f"📊 Processing {len(sorted_categories)} categories by priority")
        
        collection_summary = {
            'strategy_timestamp': datetime.now().isoformat(),
            'strategy_type': 'maximum_value_collection',
            'categories_processed': 0,
            'successful_collections': 0,
            'total_510k_records': 0,
            'total_maude_records': 0,
            'total_recall_records': 0,
            'expected_vs_actual': {},
            'category_results': {}
        }
        
        for product_code, category_info in sorted_categories:
            try:
                collection_summary['categories_processed'] += 1
                
                result = self.collect_optimal_category(product_code, category_info)
                
                if result:
                    collection_summary['successful_collections'] += 1
                    collection_summary['total_510k_records'] += result['510k_count']
                    collection_summary['total_maude_records'] += result['maude_count']
                    collection_summary['total_recall_records'] += result['recall_count']
                    
                    # Track expected vs actual
                    expected = result['expected_records']
                    actual = result['actual_collected']
                    collection_summary['expected_vs_actual'][product_code] = {
                        'expected': expected,
                        'actual': actual,
                        'efficiency': (actual / expected * 100) if expected > 0 else 0
                    }
                    
                    collection_summary['category_results'][product_code] = result
                    
                    logger.info(f"✅ Priority {result['priority']} completed: {product_code} - {result['actual_collected']:,} records")
                    
                    # Brief pause between categories to be nice to the API
                    time.sleep(3)
                
            except Exception as e:
                logger.error(f"❌ Failed to collect {product_code}: {e}")
                continue
        
        # Calculate totals and save summary
        grand_total = (collection_summary['total_510k_records'] + 
                      collection_summary['total_maude_records'] + 
                      collection_summary['total_recall_records'])
        
        collection_summary['grand_total_records'] = grand_total
        
        # Save strategy summary
        summary_file = self.results_dir / "maximum_collection_strategy_results.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🎉 MAXIMUM COLLECTION STRATEGY COMPLETED!")
        logger.info(f"📊 New records collected: {grand_total:,}")
        logger.info(f"📈 Categories successfully added: {collection_summary['successful_collections']}")
        logger.info(f"📁 Results summary: {summary_file}")
        
        # Display efficiency summary
        logger.info(f"\n📊 COLLECTION EFFICIENCY SUMMARY:")
        for code, efficiency_data in collection_summary['expected_vs_actual'].items():
            logger.info(f"  {code}: {efficiency_data['actual']:,}/{efficiency_data['expected']:,} ({efficiency_data['efficiency']:.1f}%)")
        
        return collection_summary

if __name__ == "__main__":
    collector = MaximumCollectionStrategy()
    
    print("🎯 TracePredicate Maximum Collection Strategy")
    print("=" * 50)
    print()
    print("🚀 Collecting highest-value additional categories:")
    print("   • MHX: Implantable Defibrillator (27k records)")
    print("   • KDI: Artificial Heart (41k → 26k records)")  
    print("   • FDS: Endoscope (37k → 26k records)")
    print("   • LZO: Surgical Robot (30k → 26k records)")
    print("   • DYB: Blood Pressure Monitor (35k → 26k records)")
    print("   • MMI: Mammography System (18k records)")
    print("   • LWJ: CT Scanner (4k records)")  
    print("   • MCH: Cochlear Implant (468 records)")
    print()
    print("🎯 Expected additional records: ~170,000")
    print("🎯 Total expected dataset: ~390,000 records")
    print()
    
    # Run maximum collection for top 5 priorities first
    result = collector.run_maximum_collection(priority_limit=5)
    
    print(f"\n✅ Phase 1 completed: {result['grand_total_records']:,} additional records!")
    print("🚀 Ready to continue with remaining categories if desired")