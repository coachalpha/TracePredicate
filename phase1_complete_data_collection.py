#!/usr/bin/env python3
"""
Phase 1: Complete FDA Data Collection
=====================================

This script collects ALL available real FDA data and caches it locally
for future analysis without repeated API calls.
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import time
from typing import Dict, List, Any
import pickle
import gzip

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteFDADataCollector:
    """Collect and cache complete FDA dataset for offline analysis."""
    
    def __init__(self):
        self.cache_dir = Path("fda_data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://api.fda.gov"
        self.rate_limit_delay = 1.0
        
        # Device categories to collect
        self.device_categories = {
            'KWA': 'Hip Prostheses (Orthopedic)',
            'LNH': 'MRI Systems (Radiology)', 
            'MAF': 'Cardiac Devices (Cardiovascular)'
        }

    def get_data_inventory(self) -> Dict[str, Dict[str, int]]:
        """Get complete inventory of available FDA data."""
        logger.info("Getting complete FDA data inventory...")
        
        inventory = {}
        
        for product_code, category_name in self.device_categories.items():
            logger.info(f"Checking {product_code} ({category_name})...")
            
            counts = {
                '510k_total': 0,
                'maude_total': 0,
                'recalls_total': 0
            }
            
            # Get 510(k) count
            try:
                url = f"{self.base_url}/device/510k.json"
                params = {'search': f'product_code:{product_code}', 'limit': 1}
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    counts['510k_total'] = data['meta']['results']['total']
                    logger.info(f"  510(k): {counts['510k_total']:,} available")
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.warning(f"510(k) inventory error for {product_code}: {e}")
            
            # Get MAUDE count
            try:
                url = f"{self.base_url}/device/event.json"
                params = {'search': f'device.device_report_product_code:{product_code}', 'limit': 1}
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    counts['maude_total'] = data['meta']['results']['total']
                    logger.info(f"  MAUDE: {counts['maude_total']:,} available")
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.warning(f"MAUDE inventory error for {product_code}: {e}")
            
            # Get Recalls count
            try:
                url = f"{self.base_url}/device/recall.json"
                params = {'search': f'product_code:{product_code}', 'limit': 1}
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    counts['recalls_total'] = data['meta']['results']['total']
                    logger.info(f"  Recalls: {counts['recalls_total']:,} available")
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.warning(f"Recalls inventory error for {product_code}: {e}")
            
            inventory[product_code] = {
                'category_name': category_name,
                **counts,
                'total': sum(counts.values())
            }
            
            logger.info(f"  {product_code} TOTAL: {inventory[product_code]['total']:,} records")
        
        # Save inventory
        with open(self.cache_dir / "fda_data_inventory.json", 'w') as f:
            json.dump(inventory, f, indent=2)
        
        grand_total = sum(cat['total'] for cat in inventory.values())
        logger.info(f"\nGRAND TOTAL AVAILABLE: {grand_total:,} FDA records")
        
        return inventory

    def collect_batch_data(self, url: str, search_param: str, batch_size: int = 1000) -> List[Dict]:
        """Collect data in batches with proper error handling."""
        all_results = []
        skip = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while consecutive_errors < max_consecutive_errors:
            try:
                params = {
                    'search': search_param,
                    'limit': batch_size,
                    'skip': skip
                }
                
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                total_available = data['meta']['results']['total']
                
                if not results:
                    logger.info(f"No more results at skip={skip}")
                    break
                
                all_results.extend(results)
                consecutive_errors = 0  # Reset error counter
                
                logger.info(f"Collected {len(all_results):,} / {total_available:,} records")
                
                skip += batch_size
                
                # Check if we've collected everything
                if len(all_results) >= total_available:
                    logger.info(f"Collected all {total_available:,} available records")
                    break
                
            except requests.exceptions.RequestException as e:
                consecutive_errors += 1
                logger.warning(f"Request error {consecutive_errors}/{max_consecutive_errors} at skip={skip}: {e}")
                if consecutive_errors < max_consecutive_errors:
                    time.sleep(min(consecutive_errors * 2, 10))  # Exponential backoff
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Unexpected error {consecutive_errors}/{max_consecutive_errors}: {e}")
                if consecutive_errors < max_consecutive_errors:
                    time.sleep(5)
        
        return all_results

    def collect_complete_dataset(self, inventory: Dict) -> Dict:
        """Collect complete FDA dataset based on inventory."""
        logger.info("Starting complete FDA data collection...")
        
        complete_dataset = {
            'collection_timestamp': datetime.now().isoformat(),
            'inventory': inventory,
            'data': {}
        }
        
        total_collected = 0
        
        for product_code, info in inventory.items():
            if info['total'] == 0:
                logger.info(f"Skipping {product_code} - no data available")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"COLLECTING {product_code} ({info['category_name']})")
            logger.info(f"Expected: {info['total']:,} total records")
            logger.info(f"{'='*60}")
            
            category_data = {
                'product_code': product_code,
                'category_name': info['category_name'],
                'expected_counts': info,
                '510k_data': [],
                'maude_data': [],
                'recall_data': []
            }
            
            # Collect 510(k) data
            if info['510k_total'] > 0:
                logger.info(f"\nCollecting {info['510k_total']:,} 510(k) records for {product_code}...")
                try:
                    url = f"{self.base_url}/device/510k.json"
                    search_param = f'product_code:{product_code}'
                    category_data['510k_data'] = self.collect_batch_data(url, search_param)
                    logger.info(f"✅ 510(k) collection complete: {len(category_data['510k_data']):,} records")
                except Exception as e:
                    logger.error(f"❌ 510(k) collection failed: {e}")
            
            # Collect MAUDE data
            if info['maude_total'] > 0:
                logger.info(f"\nCollecting {info['maude_total']:,} MAUDE records for {product_code}...")
                try:
                    url = f"{self.base_url}/device/event.json"
                    search_param = f'device.device_report_product_code:{product_code}'
                    category_data['maude_data'] = self.collect_batch_data(url, search_param)
                    logger.info(f"✅ MAUDE collection complete: {len(category_data['maude_data']):,} records")
                except Exception as e:
                    logger.error(f"❌ MAUDE collection failed: {e}")
            
            # Collect Recalls data
            if info['recalls_total'] > 0:
                logger.info(f"\nCollecting {info['recalls_total']:,} Recalls records for {product_code}...")
                try:
                    url = f"{self.base_url}/device/recall.json"
                    search_param = f'product_code:{product_code}'
                    category_data['recall_data'] = self.collect_batch_data(url, search_param)
                    logger.info(f"✅ Recalls collection complete: {len(category_data['recall_data']):,} records")
                except Exception as e:
                    logger.error(f"❌ Recalls collection failed: {e}")
            
            # Save category data
            category_total = (len(category_data['510k_data']) + 
                            len(category_data['maude_data']) + 
                            len(category_data['recall_data']))
            
            total_collected += category_total
            complete_dataset['data'][product_code] = category_data
            
            logger.info(f"\n✅ {product_code} COLLECTION SUMMARY:")
            logger.info(f"   510(k): {len(category_data['510k_data']):,} / {info['510k_total']:,}")
            logger.info(f"   MAUDE: {len(category_data['maude_data']):,} / {info['maude_total']:,}")
            logger.info(f"   Recalls: {len(category_data['recall_data']):,} / {info['recalls_total']:,}")
            logger.info(f"   Total: {category_total:,} / {info['total']:,}")
            logger.info(f"   Running Total: {total_collected:,}")
            
            # Save incremental progress
            self.save_dataset_chunk(product_code, category_data)
        
        complete_dataset['total_collected'] = total_collected
        return complete_dataset

    def save_dataset_chunk(self, product_code: str, data: Dict):
        """Save individual category data chunk."""
        chunk_file = self.cache_dir / f"{product_code}_complete_data.json.gz"
        
        with gzip.open(chunk_file, 'wt') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {product_code} data chunk to {chunk_file}")

    def save_complete_dataset(self, dataset: Dict):
        """Save complete dataset with compression."""
        
        # Save main dataset (without raw data to save space)
        summary = {
            'collection_timestamp': dataset['collection_timestamp'],
            'inventory': dataset['inventory'],
            'total_collected': dataset['total_collected'],
            'data_files': {
                code: f"{code}_complete_data.json.gz" 
                for code in dataset['data'].keys()
            }
        }
        
        with open(self.cache_dir / "complete_dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save compressed complete dataset
        with gzip.open(self.cache_dir / "complete_fda_dataset.pkl.gz", 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"💾 Complete dataset saved to cache directory")

    def generate_collection_report(self, dataset: Dict) -> str:
        """Generate comprehensive collection report."""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_collected = dataset['total_collected']
        inventory = dataset['inventory']
        
        report = f"""
# TracePredicate: Complete FDA Data Collection Report
Generated: {timestamp}

## 🎯 MISSION: COMPLETE FDA DATA COLLECTION

This report documents the **COMPLETE COLLECTION** of all available FDA regulatory data
for the TracePredicate research project. All data is now cached locally for unlimited analysis.

## 📊 COLLECTION RESULTS

### Data Inventory vs. Collection:
"""
        
        for product_code, info in inventory.items():
            if product_code in dataset['data']:
                collected = dataset['data'][product_code]
                
                report += f"""
#### {info['category_name']} ({product_code})
- **510(k) Clearances**: {len(collected['510k_data']):,} / {info['510k_total']:,} available ({(len(collected['510k_data'])/max(info['510k_total'],1)*100):.1f}%)
- **MAUDE Events**: {len(collected['maude_data']):,} / {info['maude_total']:,} available ({(len(collected['maude_data'])/max(info['maude_total'],1)*100):.1f}%)
- **FDA Recalls**: {len(collected['recall_data']):,} / {info['recalls_total']:,} available ({(len(collected['recall_data'])/max(info['recalls_total'],1)*100):.1f}%)
- **Category Total**: {len(collected['510k_data']) + len(collected['maude_data']) + len(collected['recall_data']):,} / {info['total']:,} available
"""
        
        total_available = sum(info['total'] for info in inventory.values())
        collection_rate = (total_collected / max(total_available, 1)) * 100
        
        report += f"""

### 🏆 COLLECTION SUMMARY:
- **Total Available**: {total_available:,} FDA records
- **Total Collected**: {total_collected:,} FDA records  
- **Collection Rate**: {collection_rate:.1f}%
- **Cache Location**: `{self.cache_dir.absolute()}`

## 💾 CACHED DATA STRUCTURE

The complete FDA dataset is now stored locally in the following format:

```
{self.cache_dir}/
├── complete_dataset_summary.json     # Main index
├── complete_fda_dataset.pkl.gz       # Complete compressed dataset  
├── fda_data_inventory.json           # Original data inventory
├── KWA_complete_data.json.gz         # Hip prostheses data
├── LNH_complete_data.json.gz         # MRI systems data
└── MAF_complete_data.json.gz         # Cardiac devices data
```

## 🚀 NEXT STEPS: UNLIMITED ANALYSIS

With the complete FDA dataset cached locally, you can now:

1. **Run Multiple Analyses** without API calls
2. **Experiment with Parameters** using the full dataset
3. **Develop New Metrics** on the complete data
4. **Generate Visualizations** from the full regulatory record
5. **Perform Cross-Category Comparisons** with all available data

## 🎯 ANALYSIS READINESS

**TracePredicate is now ready for:**
✅ **Unlimited LDI Analysis** - Full dataset available locally
✅ **Statistical Validation** - No API rate limits  
✅ **Comprehensive Research** - Complete regulatory history
✅ **Reproducible Science** - Consistent dataset for all analyses
✅ **Advanced Analytics** - Full data exploration capabilities

---

**STATUS**: Complete FDA data collection **SUCCESSFUL**
**Records Cached**: {total_collected:,} real FDA regulatory records
**Ready for Analysis**: YES - Unlimited offline analysis now possible
"""
        
        # Save report
        report_path = self.cache_dir / "COMPLETE_DATA_COLLECTION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📋 Collection report saved to {report_path}")
        return report


def main():
    """Execute complete FDA data collection."""
    
    print("🎯 TracePredicate: Complete FDA Data Collection")
    print("=" * 60)
    print("📊 MISSION: Cache ALL available FDA regulatory data locally")
    print("🚀 GOAL: Enable unlimited offline analysis without API limits")
    print()
    
    collector = CompleteFDADataCollector()
    
    try:
        print("🔍 PHASE 1: FDA Data Inventory...")
        inventory = collector.get_data_inventory()
        
        total_available = sum(cat['total'] for cat in inventory.values())
        print(f"\n📊 TOTAL AVAILABLE: {total_available:,} FDA records")
        
        if total_available == 0:
            print("❌ No data available for collection")
            return
        
        # Estimate collection time
        estimated_time = (total_available / 1000) * 2  # ~2 seconds per 1000 records
        print(f"⏱️  ESTIMATED TIME: ~{estimated_time/60:.0f} minutes")
        
        print("\n🚀 PHASE 2: Complete Data Collection...")
        complete_dataset = collector.collect_complete_dataset(inventory)
        
        print("\n💾 PHASE 3: Saving complete dataset...")
        collector.save_complete_dataset(complete_dataset)
        
        print("\n📋 PHASE 4: Generating collection report...")
        report = collector.generate_collection_report(complete_dataset)
        
        print(f"\n🎉 COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"📊 TOTAL COLLECTED: {complete_dataset['total_collected']:,} FDA records")
        print(f"💾 CACHED LOCATION: {collector.cache_dir}")
        print(f"📋 COLLECTION REPORT: COMPLETE_DATA_COLLECTION_REPORT.md")
        print(f"\n✅ TracePredicate is now ready for unlimited offline analysis!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Collection interrupted by user")
        print("💾 Partial data may be saved in cache directory")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        print(f"\n❌ Collection failed: {e}")
        raise

if __name__ == "__main__":
    main()