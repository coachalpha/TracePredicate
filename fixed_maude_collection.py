#!/usr/bin/env python3
"""
修复后的MAUDE数据收集脚本
直接修复major_510k_categories_analysis.py中的MAUDE数据收集问题
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

class FixedMAUDECollector:
    """修复后的MAUDE数据收集器"""
    
    def __init__(self):
        self.cache_dir = Path("major_510k_cache")
        self.base_url = "https://api.fda.gov"
        self.rate_limit_delay = 1.0
    
    def fix_maude_data_for_category(self, product_code: str, expected_maude_count: int, sample_size: int = 2000):
        """为特定类别修复MAUDE数据收集"""
        logger.info(f"🔧 修复 {product_code} 的MAUDE数据收集...")
        
        # 读取现有的类别数据文件
        cache_file = self.cache_dir / f"{product_code}_data.json.gz"
        
        if not cache_file.exists():
            logger.error(f"❌ 缓存文件不存在: {cache_file}")
            return False
        
        try:
            with gzip.open(cache_file, 'rt') as f:
                category_data = json.load(f)
            
            current_maude_count = len(category_data.get('maude_data', []))
            logger.info(f"  当前MAUDE数据: {current_maude_count:,}")
            logger.info(f"  期望MAUDE数据: {expected_maude_count:,}")
            
            if current_maude_count == 0 and expected_maude_count > 0:
                # 重新收集MAUDE数据
                strategic_maude = min(sample_size, expected_maude_count)
                logger.info(f"  重新收集MAUDE数据: {strategic_maude:,} 条")
                
                try:
                    url = f"{self.base_url}/device/event.json"
                    params = {
                        'search': f'device.device_report_product_code:{product_code}',
                        'limit': strategic_maude
                    }
                    
                    logger.info(f"  🌐 API调用: {url}")
                    logger.info(f"  📊 参数: {params}")
                    
                    response = requests.get(url, params=params, timeout=60)
                    logger.info(f"  📡 响应状态: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        maude_results = data.get('results', [])
                        
                        logger.info(f"  ✅ 成功收集: {len(maude_results):,} MAUDE记录")
                        
                        # 更新类别数据
                        category_data['maude_data'] = maude_results
                        
                        # 保存更新后的数据
                        with gzip.open(cache_file, 'wt') as f:
                            json.dump(category_data, f, indent=2, default=str)
                        
                        logger.info(f"  💾 已更新缓存文件")
                        return True
                    else:
                        logger.error(f"  ❌ API调用失败: {response.status_code}")
                        logger.error(f"  响应内容: {response.text[:500]}")
                        return False
                        
                except Exception as e:
                    logger.error(f"  ❌ MAUDE收集异常: {e}")
                    return False
            else:
                logger.info(f"  ✅ {product_code} MAUDE数据已存在，无需修复")
                return True
                
        except Exception as e:
            logger.error(f"❌ 处理 {product_code} 时出错: {e}")
            return False
    
    def fix_all_maude_data(self):
        """修复所有类别的MAUDE数据"""
        logger.info("🚀 开始修复所有类别的MAUDE数据...")
        
        # 读取inventory以获取期望的MAUDE计数
        inventory_file = self.cache_dir / "major_510k_inventory.json"
        if not inventory_file.exists():
            logger.error("❌ 找不到inventory文件")
            return False
        
        with open(inventory_file, 'r') as f:
            inventory = json.load(f)
        
        fixed_categories = []
        failed_categories = []
        
        for product_code, info in inventory.items():
            expected_maude = info.get('maude_total', 0)
            category_name = info.get('category_name', 'Unknown')
            
            if expected_maude > 0:  # 只修复应该有MAUDE数据的类别
                logger.info(f"\n📊 处理 {product_code} ({category_name})")
                logger.info(f"   期望MAUDE事件: {expected_maude:,}")
                
                success = self.fix_maude_data_for_category(product_code, expected_maude)
                
                if success:
                    fixed_categories.append({
                        'product_code': product_code,
                        'category_name': category_name,
                        'expected_maude': expected_maude
                    })
                else:
                    failed_categories.append({
                        'product_code': product_code,
                        'category_name': category_name,
                        'expected_maude': expected_maude
                    })
                
                time.sleep(self.rate_limit_delay)
            else:
                logger.info(f"⏭️  跳过 {product_code} ({category_name}) - 无MAUDE数据")
        
        # 生成修复报告
        logger.info(f"\n🎉 MAUDE数据修复完成!")
        logger.info(f"✅ 成功修复: {len(fixed_categories)} 个类别")
        logger.info(f"❌ 修复失败: {len(failed_categories)} 个类别")
        
        if fixed_categories:
            logger.info("\n✅ 成功修复的类别:")
            for cat in fixed_categories:
                logger.info(f"  - {cat['product_code']} ({cat['category_name']}): {cat['expected_maude']:,} 期望事件")
        
        if failed_categories:
            logger.info("\n❌ 修复失败的类别:")
            for cat in failed_categories:
                logger.info(f"  - {cat['product_code']} ({cat['category_name']}): {cat['expected_maude']:,} 期望事件")
        
        return len(failed_categories) == 0
    
    def verify_fixed_data(self):
        """验证修复后的数据完整性"""
        logger.info("\n🔍 验证修复后的数据完整性...")
        
        verification_results = {}
        total_maude_collected = 0
        
        for cache_file in self.cache_dir.glob("*.json.gz"):
            if cache_file.name == "major_510k_inventory.json":
                continue
                
            product_code = cache_file.stem.replace("_data", "")
            
            try:
                with gzip.open(cache_file, 'rt') as f:
                    category_data = json.load(f)
                
                maude_count = len(category_data.get('maude_data', []))
                category_name = category_data.get('category_name', 'Unknown')
                
                verification_results[product_code] = {
                    'category_name': category_name,
                    'maude_collected': maude_count
                }
                
                total_maude_collected += maude_count
                
                if maude_count > 0:
                    logger.info(f"✅ {product_code} ({category_name}): {maude_count:,} MAUDE事件")
                else:
                    logger.info(f"ℹ️  {product_code} ({category_name}): 0 MAUDE事件")
                    
            except Exception as e:
                logger.error(f"❌ 验证 {product_code} 时出错: {e}")
        
        logger.info(f"\n📊 验证总结:")
        logger.info(f"   总MAUDE事件收集: {total_maude_collected:,}")
        logger.info(f"   处理的类别数: {len(verification_results)}")
        
        return verification_results

def main():
    print("🔧 TracePredicate: MAUDE数据修复工具")
    print("=" * 60)
    print("🎯 目标: 修复major_510k_categories_analysis.py中的MAUDE数据收集问题")
    print()
    
    collector = FixedMAUDECollector()
    
    try:
        # 修复MAUDE数据
        success = collector.fix_all_maude_data()
        
        # 验证修复结果
        verification = collector.verify_fixed_data()
        
        if success:
            print(f"\n🎉 修复完成! 所有MAUDE数据已成功收集")
            print(f"💾 数据已保存到major_510k_cache目录")
            print(f"🚀 现在可以重新运行LDI分析获得正确结果")
        else:
            print(f"\n⚠️  修复部分完成，请检查失败的类别")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  修复被用户中断")
    except Exception as e:
        logger.error(f"修复失败: {e}")
        print(f"\n❌ 修复失败: {e}")

if __name__ == "__main__":
    main()