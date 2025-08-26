#!/usr/bin/env python3
"""
Collection Status Monitor
=========================

Monitor the progress of our expanded data collection and provide real-time status updates.
"""

import json
import gzip
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List

class CollectionStatusMonitor:
    """Monitor data collection progress and status"""
    
    def __init__(self):
        self.cache_dir = Path("data/real_fda_dataset")
        self.results_dir = Path("results")
        
        # All categories we're tracking
        self.all_categories = {
            # Existing categories
            'KWA': 'Hip Prostheses (Orthopedic)',
            'KWP': 'Knee Prostheses (Orthopedic)', 
            'KWF': 'Shoulder Prostheses (Orthopedic)',
            'HRS': 'Bone Plates/Screws (Orthopedic)',
            'HWC': 'Bone Drill (Orthopedic)',
            'LNH': 'MRI Systems (Radiology)',
            'IYE': 'Ultrasound Systems (Radiology)',
            'JAK': 'X-ray Systems (Radiology)',
            'FRN': 'Infusion Pumps (Critical Care)',
            'BTO': 'Ventilators (Critical Care)',
            'DQO': 'Catheters (Cardiovascular)',
            'ETA': 'Hearing Aids (ENT)',
            'IOL': 'Intraocular Lenses (Ophthalmic)',
            
            # New Tier 1 categories
            'NIK': 'Pacemaker Pulse Generator (Cardiovascular)',
            'DTK': 'Coronary Stent (Cardiovascular)',
            'MHX': 'Implantable Defibrillator (Cardiovascular)',
            
            # New Tier 2 categories
            'FDS': 'Endoscope (Surgical)',
            'LZO': 'Surgical Robot (Surgical)',
            
            # New Tier 3 categories
            'GDT': 'Insulin Pump (Endocrine)',
            'GAL': 'Breast Prosthesis (Plastic Surgery)'
        }

    def check_category_status(self, product_code: str) -> Dict:
        """Check the current status of a single category"""
        
        data_file = self.cache_dir / f"{product_code}_complete_data.json.gz"
        
        if not data_file.exists():
            return {
                'status': 'missing',
                'file_exists': False,
                'last_modified': None,
                'records': {}
            }
        
        try:
            # Get file modification time
            last_modified = datetime.fromtimestamp(data_file.stat().st_mtime)
            
            # Load and analyze data
            with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                'status': 'available',
                'file_exists': True,
                'last_modified': last_modified.isoformat(),
                'collection_timestamp': data.get('collection_timestamp', 'unknown'),
                'enhanced_collection': data.get('enhanced_collection', False),
                'records': {
                    '510k_count': len(data.get('510k_data', [])),
                    'maude_count': len(data.get('maude_data', [])),
                    'recall_count': len(data.get('recall_data', [])),
                    'total_count': (len(data.get('510k_data', [])) + 
                                  len(data.get('maude_data', [])) + 
                                  len(data.get('recall_data', [])))
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'file_exists': True,
                'error': str(e),
                'records': {}
            }

    def generate_comprehensive_status(self) -> Dict:
        """Generate comprehensive status report"""
        
        print("🔍 TracePredicate Data Collection Status Monitor")
        print("=" * 50)
        
        status_report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_categories': len(self.all_categories),
            'categories_with_data': 0,
            'total_records': 0,
            'category_details': {},
            'summary_by_type': {
                'existing_original': {'count': 0, 'records': 0},
                'existing_enhanced': {'count': 0, 'records': 0},
                'new_tier1': {'count': 0, 'records': 0},
                'new_tier2': {'count': 0, 'records': 0},
                'new_tier3': {'count': 0, 'records': 0},
                'missing': {'count': 0, 'categories': []}
            }
        }
        
        # Category type mapping
        existing_categories = ['KWA', 'KWP', 'KWF', 'HRS', 'HWC', 'LNH', 'IYE', 'JAK', 'FRN', 'BTO', 'DQO', 'ETA', 'IOL']
        tier1_categories = ['NIK', 'DTK', 'MHX']
        tier2_categories = ['FDS', 'LZO']
        tier3_categories = ['GDT', 'GAL']
        
        print(f"\n📊 Checking {len(self.all_categories)} categories...")
        print()
        
        for product_code, category_name in self.all_categories.items():
            category_status = self.check_category_status(product_code)
            status_report['category_details'][product_code] = {
                'name': category_name,
                **category_status
            }
            
            # Determine category type and update summary
            if category_status['status'] == 'available':
                status_report['categories_with_data'] += 1
                total_records = category_status['records']['total_count']
                status_report['total_records'] += total_records
                
                enhanced = category_status.get('enhanced_collection', False)
                
                if product_code in existing_categories:
                    category_type = 'existing_enhanced' if enhanced else 'existing_original'
                elif product_code in tier1_categories:
                    category_type = 'new_tier1'
                elif product_code in tier2_categories:
                    category_type = 'new_tier2'
                else:
                    category_type = 'new_tier3'
                
                status_report['summary_by_type'][category_type]['count'] += 1
                status_report['summary_by_type'][category_type]['records'] += total_records
                
                # Display status
                status_icon = "🔄" if enhanced else "📊"
                print(f"{status_icon} {product_code:<4} | {category_name:<40} | {total_records:>8,} records")
                
            else:
                status_report['summary_by_type']['missing']['count'] += 1
                status_report['summary_by_type']['missing']['categories'].append(product_code)
                print(f"❌ {product_code:<4} | {category_name:<40} | Missing")
        
        return status_report

    def display_summary(self, status_report: Dict):
        """Display executive summary of collection status"""
        
        print(f"\n" + "=" * 70)
        print("📈 COLLECTION SUMMARY")
        print("=" * 70)
        
        print(f"🎯 Total Categories: {status_report['total_categories']}")
        print(f"✅ Categories with Data: {status_report['categories_with_data']}")
        print(f"📊 Total Records: {status_report['total_records']:,}")
        
        print(f"\n📁 DATA BREAKDOWN:")
        
        summary = status_report['summary_by_type']
        
        if summary['existing_original']['count'] > 0:
            print(f"   📦 Original Existing: {summary['existing_original']['count']} categories, {summary['existing_original']['records']:,} records")
            
        if summary['existing_enhanced']['count'] > 0:
            print(f"   🔄 Enhanced Existing: {summary['existing_enhanced']['count']} categories, {summary['existing_enhanced']['records']:,} records")
            
        if summary['new_tier1']['count'] > 0:
            print(f"   ⭐ New Tier 1: {summary['new_tier1']['count']} categories, {summary['new_tier1']['records']:,} records")
            
        if summary['new_tier2']['count'] > 0:
            print(f"   🥈 New Tier 2: {summary['new_tier2']['count']} categories, {summary['new_tier2']['records']:,} records")
            
        if summary['new_tier3']['count'] > 0:
            print(f"   🥉 New Tier 3: {summary['new_tier3']['count']} categories, {summary['new_tier3']['records']:,} records")
        
        if summary['missing']['count'] > 0:
            print(f"   ❌ Missing: {summary['missing']['count']} categories ({', '.join(summary['missing']['categories'])})")
        
        # Calculate improvements
        original_total = 167307  # Our original dataset size
        current_total = status_report['total_records']
        improvement = ((current_total - original_total) / original_total * 100) if original_total > 0 else 0
        
        print(f"\n📈 PROGRESS vs ORIGINAL DATASET:")
        print(f"   📊 Original: 167,307 records")
        print(f"   🚀 Current:  {current_total:,} records")
        print(f"   📈 Growth:   +{current_total - original_total:,} records ({improvement:+.1f}%)")
        
        if current_total > original_total:
            print(f"   🎉 SUCCESS: Dataset expanded by {improvement:.1f}%!")
        
        print("=" * 70)

    def save_status_report(self, status_report: Dict):
        """Save detailed status report to file"""
        
        report_file = self.results_dir / "collection_status_report.json"
        self.results_dir.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(status_report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed status report saved: {report_file}")

    def run_status_check(self):
        """Run complete status check and display results"""
        
        status_report = self.generate_comprehensive_status()
        self.display_summary(status_report)
        self.save_status_report(status_report)
        
        return status_report

if __name__ == "__main__":
    monitor = CollectionStatusMonitor()
    monitor.run_status_check()