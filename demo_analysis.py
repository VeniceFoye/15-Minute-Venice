#!/usr/bin/env python3
"""
威尼斯15分钟生活圈分析演示
==================

基于1808年威尼斯土地登记册数据和1740年POI数据的15分钟生活圈分析

运行方法：
    cd 15-Minute-Venice
    python demo_analysis.py
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.fifteen_minute_analysis import FifteenMinuteVenice
except ImportError:
    # 如果直接导入失败，尝试其他方式
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "fifteen_minute_analysis", 
        "src/fifteen_minute_analysis.py"
    )
    fifteen_min_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fifteen_min_module)
    FifteenMinuteVenice = fifteen_min_module.FifteenMinuteVenice

def main():
    """运行15分钟威尼斯生活圈分析"""
    
    print("=" * 60)
    print("威尼斯15分钟生活圈分析（1808年数据 + 1740年POI）")
    print("=" * 60)
    
    # 配置数据路径
    data_dir = Path("../../venice-data-week-data")
    
    # 检查数据文件是否存在
    required_files = [
        data_dir / "1808-sommarioni" / "venice_1808_landregister_geometries.geojson",
        data_dir / "1740-catastici" / "1740_catastici_version20250625.geojson"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("错误：以下数据文件缺失：")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请确保venice-data-week-data目录在正确位置")
        return
    
    # 创建分析器实例
    analyzer = FifteenMinuteVenice(
        data_dir=data_dir,
        cell_size=3.0,  # 3米栅格，平衡精度和计算速度
        walk_speed_kmh=3.0,  # 历史步行速度（比现代慢）
        time_limit_min=15.0,  # 15分钟时间限制
        target_crs=32633  # UTM Zone 33N适合威尼斯
    )
    
    print(f"分析参数：")
    print(f"  栅格大小：{analyzer.cell_size}米")
    print(f"  步行速度：{analyzer.walk_speed_ms * 60 / 1000:.1f} km/h")
    print(f"  时间限制：{analyzer.time_limit_min}分钟")
    print(f"  最大步行距离：{analyzer.max_distance_m:.0f}米")
    print()
    
    try:
        # 运行完整分析
        results = analyzer.run_full_analysis()
        
        print("=" * 60)
        print("分析完成！")
        print("=" * 60)
        print(f"共分析了{len(results)}栋建筑的15分钟生活圈")
        print(f"结果文件保存在：results/15min_analysis_results.csv")
        print(f"分析报告保存在：analysis_report.txt")
        print(f"可视化图表保存在：analysis_results.png")
        
        # 显示一些基本统计
        print(f"\n基本统计：")
        print(f"  平均可达面积：{results['accessible_area_m2'].mean():.0f} m²")
        print(f"  平均可达POI数：{results['total_pois'].mean():.1f}个")
        print(f"  高便利性建筑（>20个POI）：{(results['total_pois'] > 20).sum()}栋")
        
    except Exception as e:
        print(f"分析过程中出现错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 