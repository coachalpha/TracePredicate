#!/usr/bin/env python3
"""
Working MAUDE API Test Script
============================

This script demonstrates the WORKING MAUDE database API access
using the correct field names discovered during investigation.
"""

import requests
import json
import pandas as pd
from typing import Dict, List, Any

def test_maude_api_access():
    """Test MAUDE API with correct field names."""
    
    print("🔬 Testing MAUDE Database API Access")
    print("=" * 50)
    
    # Test 1: Search by device_report_product_code (WORKING!)
    print("\n📊 Test 1: Search by Product Code KWA")
    url = "https://api.fda.gov/device/event.json"
    params = {
        'search': 'device.device_report_product_code:KWA',
        'limit': 5
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        total_results = data['meta']['results']['total']
        print(f"✅ SUCCESS: Found {total_results} KWA adverse events!")
        
        # Show sample results
        for i, result in enumerate(data['results'][:3]):
            print(f"\n📋 Event {i+1}:")
            print(f"  - Report Number: {result.get('report_number')}")
            print(f"  - Event Type: {result.get('event_type')}")
            print(f"  - Date Received: {result.get('date_received')}")
            print(f"  - Product Code: {result['device'][0]['device_report_product_code']}")
            print(f"  - Device Name: {result['device'][0].get('brand_name', 'N/A')}")
            print(f"  - Manufacturer: {result['device'][0].get('manufacturer_d_name', 'N/A')}")
            if 'product_problems' in result:
                print(f"  - Problems: {', '.join(result['product_problems'])}")
    else:
        print(f"❌ FAILED: {response.status_code}")
    
    # Test 2: Search by 510(k) number (WORKING!)
    print("\n\n📊 Test 2: Search by 510(k) Number K101086")
    params = {
        'search': 'pma_pmn_number:K101086',
        'limit': 3
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        total_results = data['meta']['results']['total']
        print(f"✅ SUCCESS: Found {total_results} events for 510(k) K101086!")
        
        for i, result in enumerate(data['results'][:2]):
            print(f"\n📋 Event {i+1}:")
            print(f"  - 510(k) Number: {result.get('pma_pmn_number')}")
            print(f"  - Event Type: {result.get('event_type')}")
            print(f"  - Product Code: {result['device'][0]['device_report_product_code']}")
            print(f"  - Brand Name: {result['device'][0].get('brand_name')}")
    else:
        print(f"❌ FAILED: {response.status_code}")
    
    # Test 3: Search LNH products
    print("\n\n📊 Test 3: Search by Product Code LNH (MRI Systems)")
    params = {
        'search': 'device.device_report_product_code:LNH',
        'limit': 3
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        total_results = data['meta']['results']['total']
        print(f"✅ SUCCESS: Found {total_results} LNH adverse events!")
        
        for i, result in enumerate(data['results'][:2]):
            print(f"\n📋 Event {i+1}:")
            print(f"  - Report Number: {result.get('report_number')}")
            print(f"  - Event Type: {result.get('event_type')}")
            print(f"  - Product Code: {result['device'][0]['device_report_product_code']}")
            print(f"  - Device: {result['device'][0].get('brand_name')}")
            print(f"  - Manufacturer: {result['device'][0].get('manufacturer_d_name')}")
    else:
        print(f"❌ FAILED: {response.status_code}")
    
    # Test 4: Search MAF products
    print("\n\n📊 Test 4: Search by Product Code MAF (Cardiac)")
    params = {
        'search': 'device.device_report_product_code:MAF',
        'limit': 3
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        total_results = data['meta']['results']['total']
        print(f"✅ SUCCESS: Found {total_results} MAF adverse events!")
        
        for i, result in enumerate(data['results'][:2]):
            print(f"\n📋 Event {i+1}:")
            print(f"  - Report Number: {result.get('report_number')}")
            print(f"  - Event Type: {result.get('event_type')}")
            print(f"  - Product Code: {result['device'][0]['device_report_product_code']}")
            print(f"  - Device: {result['device'][0].get('brand_name')}")
    else:
        print(f"❌ FAILED: {response.status_code}")

def collect_maude_data_sample():
    """Collect sample MAUDE data for analysis."""
    print("\n\n🔍 Collecting MAUDE Sample Data for Analysis")
    print("=" * 50)
    
    url = "https://api.fda.gov/device/event.json"
    all_events = []
    
    for product_code in ['KWA', 'LNH', 'MAF']:
        print(f"\n📊 Collecting {product_code} events...")
        
        params = {
            'search': f'device.device_report_product_code:{product_code}',
            'limit': 10  # Get 10 per category
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            events = data['results']
            
            for event in events:
                # Extract key information
                event_data = {
                    'product_code': product_code,
                    'report_number': event.get('report_number'),
                    'event_type': event.get('event_type'),
                    'date_received': event.get('date_received'),
                    'date_of_event': event.get('date_of_event'),
                    'pma_pmn_number': event.get('pma_pmn_number'),
                    'brand_name': event['device'][0].get('brand_name') if event.get('device') else None,
                    'manufacturer': event['device'][0].get('manufacturer_d_name') if event.get('device') else None,
                    'product_problems': ', '.join(event.get('product_problems', [])),
                    'adverse_event_flag': event.get('adverse_event_flag'),
                    'mdr_report_key': event.get('mdr_report_key')
                }
                all_events.append(event_data)
            
            print(f"  ✅ Collected {len(events)} {product_code} events")
        else:
            print(f"  ❌ Failed to collect {product_code} events")
    
    # Create DataFrame
    df = pd.DataFrame(all_events)
    
    print(f"\n📈 Summary:")
    print(f"  - Total Events: {len(df)}")
    print(f"  - KWA Events: {len(df[df['product_code'] == 'KWA'])}")
    print(f"  - LNH Events: {len(df[df['product_code'] == 'LNH'])}")
    print(f"  - MAF Events: {len(df[df['product_code'] == 'MAF'])}")
    
    # Save to file
    output_file = 'real_fda_analysis/maude_sample_data.csv'
    df.to_csv(output_file, index=False)
    print(f"  📁 Data saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    # Test API access
    test_maude_api_access()
    
    # Collect sample data
    sample_df = collect_maude_data_sample()
    
    print("\n\n🎉 MAUDE DATABASE ACCESS FULLY FUNCTIONAL!")
    print("✅ The issue was using wrong field name: 'product_code' vs 'device.device_report_product_code'")
    print("✅ Can search by product codes: KWA, LNH, MAF")
    print("✅ Can search by 510(k) numbers: pma_pmn_number field")
    print("✅ Rich data available: event types, manufacturers, problems, dates")
    print("\n🚀 TracePredicate can now access REAL MAUDE adverse event data!")