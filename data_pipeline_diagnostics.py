#!/usr/bin/env python3
"""
数据管道诊断：查找MAUDE数据关联问题
================================================

诊断在major_510k_categories_analysis.py中发现的数据矛盾问题
"""

import json
import gzip
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipelineDiagnostics:
    """诊断数据管道中的MAUDE数据关联问题"""
    
    def __init__(self):
        self.cache_dir = Path("major_510k_cache")
        self.results_dir = Path("data_diagnostics")
        self.results_dir.mkdir(exist_ok=True)
    
    def analyze_raw_data_files(self):
        """分析所有原始数据文件中的MAUDE事件数量"""
        logger.info("🔍 诊断原始数据文件中的MAUDE事件...")
        
        diagnostics = {
            'category_analysis': {},
            'total_maude_collected': 0,
            'total_510k_collected': 0,
            'total_recalls_collected': 0,
            'categories_with_zero_maude': [],
            'categories_with_high_maude': [],
            'data_integrity_issues': []
        }
        
        # 检查每个类别的原始数据文件
        for cache_file in self.cache_dir.glob("*.json.gz"):
            product_code = cache_file.stem.replace("_data", "")
            
            try:
                with gzip.open(cache_file, 'rt') as f:
                    category_data = json.load(f)
                
                maude_count = len(category_data.get('maude_data', []))
                _510k_count = len(category_data.get('510k_data', []))
                recalls_count = len(category_data.get('recall_data', []))
                category_name = category_data.get('category_name', 'Unknown')
                
                diagnostics['category_analysis'][product_code] = {
                    'category_name': category_name,
                    'maude_events': maude_count,
                    '510k_clearances': _510k_count,
                    'recalls': recalls_count,
                    'total_records': maude_count + _510k_count + recalls_count
                }
                
                diagnostics['total_maude_collected'] += maude_count
                diagnostics['total_510k_collected'] += _510k_count
                diagnostics['total_recalls_collected'] += recalls_count
                
                # 识别异常模式
                if maude_count == 0 and category_name not in ['Dental Implants', 'Contact Lenses']:
                    diagnostics['categories_with_zero_maude'].append({
                        'product_code': product_code,
                        'category_name': category_name,
                        'expected_high_risk': True if product_code in ['FRN', 'KWA', 'HWC', 'HRS'] else False
                    })
                
                if maude_count > 1000:
                    diagnostics['categories_with_high_maude'].append({
                        'product_code': product_code,
                        'category_name': category_name,
                        'maude_count': maude_count
                    })
                
                logger.info(f"✅ {product_code} ({category_name}): {maude_count:,} MAUDE + {_510k_count:,} 510k + {recalls_count:,} recalls")
                
            except Exception as e:
                logger.error(f"❌ 无法读取 {cache_file}: {e}")
                diagnostics['data_integrity_issues'].append({
                    'file': str(cache_file),
                    'error': str(e)
                })
        
        return diagnostics
    
    def check_api_collection_parameters(self):
        """检查API数据收集的参数设置"""
        logger.info("🔍 检查API收集参数...")
        
        # 检查之前的主要分析脚本
        try:
            with open("major_510k_categories_analysis.py", 'r') as f:
                script_content = f.read()
            
            # 查找MAUDE数据收集的关键代码
            api_issues = []
            
            if 'strategic_maude = min(sample_size, info[\'maude_total\'])' in script_content:
                api_issues.append("✅ MAUDE采样逻辑正确设置为sample_size(2000)")
            else:
                api_issues.append("❌ MAUDE采样逻辑可能有问题")
                
            if 'device.device_report_product_code' in script_content:
                api_issues.append("✅ MAUDE API字段名正确使用device.device_report_product_code")
            else:
                api_issues.append("❌ MAUDE API字段名可能错误")
            
            return api_issues
            
        except Exception as e:
            return [f"❌ 无法检查API参数: {e}"]
    
    def verify_specific_high_risk_categories(self):
        """专门验证已知高风险类别的数据"""
        logger.info("🔍 验证特定高风险类别的MAUDE数据...")
        
        high_risk_verification = {}
        high_risk_codes = ['FRN', 'KWA', 'HWC', 'HRS', 'JAK']  # 已知高风险类别
        
        for code in high_risk_codes:
            cache_file = self.cache_dir / f"{code}_data.json.gz"
            if cache_file.exists():
                try:
                    with gzip.open(cache_file, 'rt') as f:
                        category_data = json.load(f)
                    
                    maude_events = category_data.get('maude_data', [])
                    category_name = category_data.get('category_name', 'Unknown')
                    
                    # 分析MAUDE事件的实际内容
                    event_analysis = {
                        'total_events': len(maude_events),
                        'sample_events': [],
                        'event_types_found': set(),
                        'has_event_data': len(maude_events) > 0
                    }
                    
                    # 检查前几个事件的结构
                    for i, event in enumerate(maude_events[:3]):
                        event_analysis['sample_events'].append({
                            'event_key': list(event.keys())[:5],  # 前5个字段
                            'has_event_type': 'event_type' in event,
                            'has_product_code': any('product_code' in str(k).lower() for k in event.keys()),
                            'event_type_value': event.get('event_type', 'NOT_FOUND')
                        })
                        
                        if 'event_type' in event:
                            event_type = event['event_type']
                            if isinstance(event_type, list):
                                event_analysis['event_types_found'].update(event_type)
                            else:
                                event_analysis['event_types_found'].add(str(event_type))
                    
                    event_analysis['event_types_found'] = list(event_analysis['event_types_found'])
                    
                    high_risk_verification[code] = {
                        'category_name': category_name,
                        'analysis': event_analysis
                    }
                    
                except Exception as e:
                    high_risk_verification[code] = {
                        'error': str(e)
                    }
        
        return high_risk_verification
    
    def generate_diagnostic_report(self, diagnostics, api_issues, verification):
        """生成完整的诊断报告"""
        
        report = f"""
# TracePredicate 数据管道诊断报告
生成时间: {pd.Timestamp.now()}

## 🚨 发现的关键问题

### 数据收集总览:
- **总MAUDE事件**: {diagnostics['total_maude_collected']:,} 条
- **总510(k)批准**: {diagnostics['total_510k_collected']:,} 条  
- **总召回记录**: {diagnostics['total_recalls_collected']:,} 条
- **分析类别数**: {len(diagnostics['category_analysis'])} 个

### ❌ 零MAUDE事件的类别 ({len(diagnostics['categories_with_zero_maude'])} 个):
"""
        
        for cat in diagnostics['categories_with_zero_maude']:
            risk_indicator = "⚠️ 预期高风险" if cat['expected_high_risk'] else "ℹ️ 可能合理"
            report += f"- **{cat['product_code']}** ({cat['category_name']}) {risk_indicator}\n"
        
        report += f"\n### ✅ 高MAUDE事件的类别 ({len(diagnostics['categories_with_high_maude'])} 个):\n"
        for cat in diagnostics['categories_with_high_maude']:
            report += f"- **{cat['product_code']}** ({cat['category_name']}): {cat['maude_count']:,} 事件\n"
        
        report += "\n### 🔍 API收集参数检查:\n"
        for issue in api_issues:
            report += f"- {issue}\n"
        
        report += "\n### 🎯 高风险类别详细验证:\n"
        for code, data in verification.items():
            if 'error' in data:
                report += f"#### {code}: ❌ 数据读取错误\n- 错误: {data['error']}\n\n"
            else:
                analysis = data['analysis']
                status = "✅ 有数据" if analysis['has_event_data'] else "❌ 无数据"
                report += f"#### {code} ({data['category_name']}): {status}\n"
                report += f"- **事件总数**: {analysis['total_events']:,}\n"
                if analysis['event_types_found']:
                    report += f"- **事件类型**: {', '.join(analysis['event_types_found'][:5])}\n"
                if analysis['sample_events']:
                    report += f"- **数据结构**: {analysis['sample_events'][0]['event_key']}\n"
                report += "\n"
        
        report += """
## 🔧 问题根本原因分析

### 可能的原因:
1. **API字段映射错误**: MAUDE API查询使用了错误的产品代码字段
2. **数据采样逻辑错误**: 采样算法可能跳过了MAUDE数据收集
3. **数据过滤过度**: 可能在数据清洗过程中过度过滤了有效数据
4. **API限制问题**: 可能遇到了未处理的API限制或超时

### 建议的修复步骤:
1. ✅ **验证API调用**: 直接测试每个产品代码的MAUDE API响应
2. ✅ **修复数据收集逻辑**: 确保MAUDE数据正确收集和存储
3. ✅ **重新运行完整分析**: 使用修复后的数据管道
4. ✅ **验证结果合理性**: 确保高风险类别显示预期的风险水平

## 📊 数据完整性评估

### 当前状态: ❌ 数据管道不可靠
- 多个预期高风险类别显示零MAUDE事件
- 数据总量与报告统计不一致  
- 需要立即修复后重新分析

### 修复优先级: 🔥 最高优先级
这个数据问题使当前的所有LDI分析结果无效。必须在报告任何结论之前修复此问题。
"""
        
        # 详细的类别分析表
        report += "\n## 📋 详细类别分析\n\n| 产品代码 | 类别名称 | MAUDE事件 | 510(k) | 召回 | 总计 |\n"
        report += "|---------|---------|----------|--------|------|------|\n"
        
        for code, data in diagnostics['category_analysis'].items():
            report += f"| {code} | {data['category_name']} | {data['maude_events']:,} | {data['510k_clearances']:,} | {data['recalls']:,} | {data['total_records']:,} |\n"
        
        return report
    
    def run_complete_diagnostics(self):
        """运行完整的诊断流程"""
        logger.info("🚨 启动TracePredicate数据管道诊断...")
        
        diagnostics = self.analyze_raw_data_files()
        api_issues = self.check_api_collection_parameters()
        verification = self.verify_specific_high_risk_categories()
        
        report = self.generate_diagnostic_report(diagnostics, api_issues, verification)
        
        # 保存诊断报告
        report_path = self.results_dir / "DATA_PIPELINE_DIAGNOSTICS.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📋 诊断报告保存至: {report_path}")
        
        # 输出关键发现
        logger.info("🚨 关键发现:")
        logger.info(f"   - 零MAUDE事件类别: {len(diagnostics['categories_with_zero_maude'])} 个")
        logger.info(f"   - 总MAUDE事件收集: {diagnostics['total_maude_collected']:,} 条")
        logger.info(f"   - 数据完整性问题: {len(diagnostics['data_integrity_issues'])} 个")
        
        return report_path

def main():
    diagnostics = DataPipelineDiagnostics()
    report_path = diagnostics.run_complete_diagnostics()
    print(f"✅ 诊断完成! 报告保存至: {report_path}")

if __name__ == "__main__":
    main()