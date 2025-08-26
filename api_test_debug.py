#!/usr/bin/env python3
"""
直接测试API调用诊断MAUDE数据收集失败的确切原因
"""

import requests
import json
import time

def test_maude_api_directly():
    """直接测试MAUDE API调用"""
    print("🔍 直接测试MAUDE API调用...")
    
    base_url = "https://api.fda.gov/device/event.json"
    
    # 测试已知有大量数据的产品代码
    test_codes = ['FRN', 'KWA', 'HWC', 'HRS', 'JAK']
    
    results = {}
    
    for code in test_codes:
        print(f"\n🧪 测试 {code}:")
        
        # 测试1: 基本计数查询
        try:
            params = {
                'search': f'device.device_report_product_code:{code}',
                'limit': 1
            }
            response = requests.get(base_url, params=params, timeout=30)
            print(f"  状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                total = data['meta']['results']['total']
                print(f"  ✅ 总计数: {total:,}")
                
                if total > 0 and 'results' in data and data['results']:
                    print(f"  ✅ 成功获取样本数据")
                    sample_event = data['results'][0]
                    print(f"  📋 样本字段: {list(sample_event.keys())[:10]}")
                    
                    # 检查产品代码字段
                    if 'device' in sample_event:
                        device_info = sample_event['device'][0] if sample_event['device'] else {}
                        print(f"  🔍 设备字段: {list(device_info.keys())[:5]}")
                        product_code_found = device_info.get('device_report_product_code')
                        print(f"  🎯 产品代码匹配: {product_code_found == code}")
                else:
                    print(f"  ❌ 无样本数据返回")
            else:
                print(f"  ❌ API错误: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  错误详情: {error_data}")
                except:
                    print(f"  响应文本: {response.text[:200]}")
            
        except Exception as e:
            print(f"  ❌ 请求异常: {e}")
        
        # 测试2: 尝试获取实际数据
        try:
            params = {
                'search': f'device.device_report_product_code:{code}',
                'limit': 5
            }
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                actual_results = data.get('results', [])
                print(f"  📊 实际获取记录: {len(actual_results)}")
                
                results[code] = {
                    'total_available': data['meta']['results']['total'],
                    'sample_retrieved': len(actual_results),
                    'api_working': True,
                    'sample_data': actual_results[:2] if actual_results else []
                }
            else:
                results[code] = {
                    'total_available': 0,
                    'sample_retrieved': 0,
                    'api_working': False,
                    'error_code': response.status_code
                }
        except Exception as e:
            results[code] = {
                'api_working': False,
                'error': str(e)
            }
        
        time.sleep(1)  # 避免API限制
    
    return results

def compare_with_original_code():
    """检查原始代码中的MAUDE收集逻辑"""
    print("\n🔍 检查原始代码中的MAUDE收集逻辑...")
    
    try:
        with open("major_510k_categories_analysis.py", 'r') as f:
            content = f.read()
        
        # 查找MAUDE收集的关键代码段
        maude_collection_lines = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'maude' in line.lower() and ('collect' in line.lower() or 'url' in line.lower() or 'params' in line.lower()):
                maude_collection_lines.append(f"Line {i+1}: {line.strip()}")
        
        print("📋 MAUDE相关代码行:")
        for line in maude_collection_lines[:10]:  # 显示前10行
            print(f"  {line}")
        
        # 检查特定的逻辑问题
        potential_issues = []
        if 'strategic_maude = min(sample_size, info[\'maude_total\'])' in content:
            if 'if strategic_maude > 0:' in content:
                print("  ✅ MAUDE采样逻辑存在且有条件检查")
            else:
                potential_issues.append("❌ MAUDE采样缺少条件检查")
        else:
            potential_issues.append("❌ 未找到MAUDE采样逻辑")
            
        if 'device.device_report_product_code' in content:
            print("  ✅ 使用正确的MAUDE API字段")
        else:
            potential_issues.append("❌ 可能使用错误的MAUDE API字段")
        
        if potential_issues:
            print("🚨 潜在问题:")
            for issue in potential_issues:
                print(f"  {issue}")
                
        return potential_issues
        
    except Exception as e:
        print(f"❌ 无法读取原始代码: {e}")
        return [f"代码读取错误: {e}"]

def main():
    print("🚨 TracePredicate API测试诊断")
    print("=" * 50)
    
    # 直接测试API
    api_results = test_maude_api_directly()
    
    # 检查原始代码
    code_issues = compare_with_original_code()
    
    print("\n" + "=" * 50)
    print("📊 诊断总结")
    print("=" * 50)
    
    print("\n🎯 API测试结果:")
    for code, result in api_results.items():
        if result.get('api_working', False):
            print(f"  ✅ {code}: API工作正常，{result['total_available']:,}条数据可用")
        else:
            print(f"  ❌ {code}: API调用失败")
    
    print(f"\n🔍 代码问题: {len(code_issues)} 个")
    for issue in code_issues:
        print(f"  {issue}")
    
    # 生成修复建议
    print("\n🔧 修复建议:")
    if any(not r.get('api_working', False) for r in api_results.values()):
        print("  1. ❌ API连接有问题，需要检查网络和认证")
    else:
        print("  1. ✅ API连接正常，问题在数据收集逻辑")
    
    if code_issues:
        print("  2. 🔧 修复数据收集代码中的逻辑错误")
    else:
        print("  2. ✅ 代码逻辑看起来正常")
    
    print("  3. 🚀 重新运行修复后的数据收集")
    print("  4. ✅ 验证修复后的结果")

if __name__ == "__main__":
    main()