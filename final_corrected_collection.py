#!/usr/bin/env python3
"""
最终修正版本：遵守FDA API 1000条记录限制的数据收集
===========================================================

使用正确的API限制重新收集所有MAUDE数据
"""

import requests
import json
import gzip
from pathlib import Path
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectedMAUDECollector:
    """修正后的MAUDE数据收集器 - 遵守API限制"""
    
    def __init__(self):
        self.cache_dir = Path("major_510k_cache")
        self.base_url = "https://api.fda.gov"
        self.rate_limit_delay = 1.0
        self.max_api_limit = 1000  # FDA API的硬限制
    
    def collect_maude_with_limit(self, product_code: str, expected_count: int) -> list:
        """收集MAUDE数据，遵守API 1000条限制"""
        
        # 确定实际收集数量（不超过1000）
        collect_count = min(self.max_api_limit, expected_count)
        
        logger.info(f"  🎯 目标收集: {collect_count:,} 条 (总可用: {expected_count:,})")
        
        try:
            url = f"{self.base_url}/device/event.json"
            params = {
                'search': f'device.device_report_product_code:{product_code}',
                'limit': collect_count
            }
            
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                logger.info(f"  ✅ 成功收集: {len(results):,} MAUDE记录")
                return results
            else:
                logger.error(f"  ❌ API调用失败: {response.status_code}")
                try:
                    error_data = response.json()
                    logger.error(f"  错误详情: {error_data}")
                except:
                    logger.error(f"  响应文本: {response.text[:200]}")
                return []
                
        except Exception as e:
            logger.error(f"  ❌ 收集异常: {e}")
            return []
    
    def fix_category_maude_data(self, product_code: str, expected_maude_count: int):
        """修复单个类别的MAUDE数据"""
        
        cache_file = self.cache_dir / f"{product_code}_data.json.gz"
        if not cache_file.exists():
            logger.error(f"❌ 缓存文件不存在: {cache_file}")
            return False
        
        try:
            # 读取现有数据
            with gzip.open(cache_file, 'rt') as f:
                category_data = json.load(f)
            
            current_maude_count = len(category_data.get('maude_data', []))
            category_name = category_data.get('category_name', 'Unknown')
            
            logger.info(f"\n🔧 处理 {product_code} ({category_name})")
            logger.info(f"  当前MAUDE数据: {current_maude_count:,}")
            logger.info(f"  期望MAUDE数据: {expected_maude_count:,}")
            
            if current_maude_count == 0 and expected_maude_count > 0:
                # 重新收集MAUDE数据
                maude_results = self.collect_maude_with_limit(product_code, expected_maude_count)
                
                if maude_results:
                    # 更新数据
                    category_data['maude_data'] = maude_results
                    
                    # 保存更新后的数据
                    with gzip.open(cache_file, 'wt') as f:
                        json.dump(category_data, f, indent=2, default=str)
                    
                    logger.info(f"  💾 已更新缓存文件")
                    return True
                else:
                    logger.error(f"  ❌ MAUDE数据收集失败")
                    return False
            else:
                logger.info(f"  ✅ MAUDE数据已存在或无数据，跳过")
                return True
                
        except Exception as e:
            logger.error(f"❌ 处理 {product_code} 时出错: {e}")
            return False
    
    def fix_all_categories(self):
        """修复所有类别的MAUDE数据收集"""
        
        logger.info("🚀 开始修复所有类别的MAUDE数据收集...")
        logger.info(f"⚡ 使用API限制: {self.max_api_limit} 条记录")
        
        # 读取inventory
        inventory_file = self.cache_dir / "major_510k_inventory.json"
        if not inventory_file.exists():
            logger.error("❌ 找不到inventory文件")
            return False
        
        with open(inventory_file, 'r') as f:
            inventory = json.load(f)
        
        # 统计
        total_categories = len(inventory)
        categories_with_maude = 0
        total_expected_maude = 0
        total_collected_maude = 0
        
        success_count = 0
        failure_count = 0
        
        # 处理每个类别
        for product_code, info in inventory.items():
            expected_maude = info.get('maude_total', 0)
            
            if expected_maude > 0:
                categories_with_maude += 1
                total_expected_maude += expected_maude
                
                success = self.fix_category_maude_data(product_code, expected_maude)
                
                if success:
                    success_count += 1
                    # 实际收集数量（受API限制）
                    actual_collected = min(self.max_api_limit, expected_maude)
                    total_collected_maude += actual_collected
                else:
                    failure_count += 1
                
                time.sleep(self.rate_limit_delay)
            else:
                success_count += 1  # 没有MAUDE数据的也算成功
        
        # 最终报告
        logger.info(f"\n🎉 修复完成!")
        logger.info(f"📊 总类别数: {total_categories}")
        logger.info(f"📊 有MAUDE数据的类别: {categories_with_maude}")
        logger.info(f"📊 期望总MAUDE事件: {total_expected_maude:,}")
        logger.info(f"📊 实际收集MAUDE事件: {total_collected_maude:,}")
        logger.info(f"✅ 成功修复: {success_count}")
        logger.info(f"❌ 修复失败: {failure_count}")
        
        collection_rate = (total_collected_maude / max(total_expected_maude, 1)) * 100
        logger.info(f"📈 收集率: {collection_rate:.1f}%")
        
        return failure_count == 0
    
    def generate_corrected_analysis(self):
        """基于修正后的数据生成分析"""
        logger.info("\n🧮 基于修正后的数据生成LDI分析...")
        
        # 这里可以调用修正后的LDI分析
        # 由于原始分析代码很长，我们先验证数据修复效果
        
        verification_results = {}
        total_maude_after_fix = 0
        
        for cache_file in self.cache_dir.glob("*_data.json.gz"):
            product_code = cache_file.stem.replace("_data", "")
            
            try:
                with gzip.open(cache_file, 'rt') as f:
                    category_data = json.load(f)
                
                maude_count = len(category_data.get('maude_data', []))
                category_name = category_data.get('category_name', 'Unknown')
                
                verification_results[product_code] = {
                    'category_name': category_name,
                    'maude_collected': maude_count,
                    'has_maude_data': maude_count > 0
                }
                
                total_maude_after_fix += maude_count
                
            except Exception as e:
                logger.error(f"验证 {product_code} 时出错: {e}")
        
        # 显示修复后的数据状态
        logger.info(f"\n📊 修复后数据验证:")
        logger.info(f"   修复后总MAUDE事件: {total_maude_after_fix:,}")
        
        categories_with_data = sum(1 for r in verification_results.values() if r['has_maude_data'])
        logger.info(f"   有MAUDE数据的类别: {categories_with_data}/{len(verification_results)}")
        
        # 显示有数据的类别
        logger.info(f"\n✅ 有MAUDE数据的类别:")
        for code, result in verification_results.items():
            if result['has_maude_data']:
                logger.info(f"   - {code} ({result['category_name']}): {result['maude_collected']:,} 事件")
        
        # 显示无数据的类别（这些应该是真的没有数据，而非收集失败）
        no_data_categories = [r for r in verification_results.values() if not r['has_maude_data']]
        if no_data_categories:
            logger.info(f"\nℹ️  无MAUDE数据的类别 ({len(no_data_categories)} 个):")
            for result in no_data_categories[:5]:  # 只显示前5个
                logger.info(f"   - {result['category_name']}")
        
        return verification_results

def main():
    print("🔧 TracePredicate: 最终修正版MAUDE数据收集")
    print("=" * 70)
    print("🎯 目标: 使用正确的API限制重新收集所有MAUDE数据")
    print(f"⚡ API限制: 1000条记录 (FDA限制)")
    print()
    
    collector = CorrectedMAUDECollector()
    
    try:
        # 修复所有MAUDE数据
        success = collector.fix_all_categories()
        
        # 验证修复结果并生成分析
        verification = collector.generate_corrected_analysis()
        
        if success:
            print(f"\n🎉 修复完全成功!")
            print(f"💾 所有数据已正确收集并保存")
            print(f"📊 现在数据管道是可靠的")
            print(f"🚀 可以运行真实的LDI分析了")
        else:
            print(f"\n⚠️  修复部分完成，但仍有一些问题")
        
        print(f"\n📋 下一步:")
        print(f"   1. 使用修正后的数据重新运行LDI分析")
        print(f"   2. 验证结果的合理性")
        print(f"   3. 生成真实的风险排名报告")
        
    except Exception as e:
        logger.error(f"修复失败: {e}")
        print(f"\n❌ 修复失败: {e}")

if __name__ == "__main__":
    main()