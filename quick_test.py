#!/usr/bin/env python3
"""
威尼斯15分钟生活圈快速测试
=====================

简化版测试脚本，验证数据加载和基本分析功能
"""

import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point
import matplotlib.pyplot as plt

def quick_analysis():
    """快速分析测试"""
    
    print("威尼斯15分钟生活圈快速测试")
    print("=" * 40)
    
    # 数据路径  
    data_dir = Path("../../venice-data-week-data")
    
    # 1. 测试数据加载
    print("1. 测试数据加载...")
    
    # 1808年几何数据
    geometries_1808 = data_dir / "1808-sommarioni" / "venice_1808_landregister_geometries.geojson"
    if not geometries_1808.exists():
        print(f"错误：文件不存在 {geometries_1808}")
        return
    
    gdf_1808 = gpd.read_file(geometries_1808)
    print(f"✓ 1808年数据：{len(gdf_1808)}条记录")
    print(f"  几何类型分布：{gdf_1808['geometry_type'].value_counts().to_dict()}")
    
    # 1740年POI数据
    catastici_1740 = data_dir / "1740-catastici" / "1740_catastici_version20250625.geojson"
    if not catastici_1740.exists():
        print(f"错误：文件不存在 {catastici_1740}")
        return
    
    poi_1740 = gpd.read_file(catastici_1740)
    print(f"✓ 1740年POI：{len(poi_1740)}条记录")
    
    # 转换坐标系
    target_crs = 32633  # UTM Zone 33N
    if gdf_1808.crs.to_epsg() != target_crs:
        gdf_1808 = gdf_1808.to_crs(target_crs)
    if poi_1740.crs.to_epsg() != target_crs:
        poi_1740 = poi_1740.to_crs(target_crs)
    
    print(f"✓ 坐标系统一为：EPSG:{target_crs}")
    
    # 2. 分离数据类型
    print("\n2. 分析数据结构...")
    
    buildings = gdf_1808[gdf_1808["geometry_type"] == 'building'].copy()
    streets = gdf_1808[gdf_1808["geometry_type"].isin(["street", "sottoportico"])].copy()
    canals = gdf_1808[gdf_1808["geometry_type"] == "water"].copy()
    
    print(f"  建筑：{len(buildings)}个")
    print(f"  街道：{len(streets)}条")
    print(f"  运河：{len(canals)}条")
    
    # 过滤POI点要素
    poi_points = poi_1740[poi_1740.geometry.type == 'Point'].copy()
    print(f"  POI点：{len(poi_points)}个")
    
    # 3. 基本空间分析
    print("\n3. 基本空间分析...")
    
    # 计算包围盒
    bounds_1808 = gdf_1808.total_bounds
    bounds_1740 = poi_points.total_bounds
    print(f"  1808年数据范围：{bounds_1808}")
    print(f"  1740年POI范围：{bounds_1740}")
    
    # 4. 简化的15分钟分析示例
    print("\n4. 简化15分钟分析示例...")
    
    # 参数设置
    walk_speed_kmh = 3.0  # 历史步行速度
    time_limit_min = 15.0
    max_distance_m = (walk_speed_kmh * 1000 / 60) * time_limit_min  # 750米
    
    print(f"  步行速度：{walk_speed_kmh} km/h")
    print(f"  最大距离：{max_distance_m:.0f}米")
    
    # 随机选择几个建筑进行测试
    test_buildings = buildings.sample(min(5, len(buildings))).copy()
    
    results = []
    for idx, building in test_buildings.iterrows():
        centroid = building.geometry.centroid
        
        # 简单的欧几里得距离计算POI可达性
        distances = poi_points.geometry.distance(centroid)
        accessible_pois = poi_points[distances <= max_distance_m]
        
        # 计算可达范围内的POI统计
        poi_count = len(accessible_pois)
        poi_types = accessible_pois['standard_type'].value_counts().to_dict() if 'standard_type' in accessible_pois.columns else {}
        poi_diversity = len(poi_types)
        
        results.append({
            'building_id': idx,
            'x': centroid.x,
            'y': centroid.y,
            'accessible_pois': poi_count,
            'poi_diversity': poi_diversity,
            'poi_types': poi_types
        })
        
        print(f"  建筑 {idx}：15分钟内可达{poi_count}个POI，{poi_diversity}种类型")
    
    # 5. 基本统计
    print("\n5. 测试结果统计...")
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print(f"  平均可达POI数：{results_df['accessible_pois'].mean():.1f}")
        print(f"  平均POI类型数：{results_df['poi_diversity'].mean():.1f}")
        print(f"  最多可达POI：{results_df['accessible_pois'].max()}")
    
    # 6. 简单可视化
    print("\n6. 生成简单可视化...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 数据分布概览
    axes[0].hist(gdf_1808['geometry_type'].value_counts().values, alpha=0.7)
    axes[0].set_title('1808年数据类型分布')
    axes[0].set_xlabel('数量')
    
    # POI可达性
    if len(results_df) > 0:
        axes[1].bar(range(len(results_df)), results_df['accessible_pois'])
        axes[1].set_title('测试建筑15分钟POI可达性')
        axes[1].set_xlabel('建筑ID')
        axes[1].set_ylabel('可达POI数量')
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ 结果图表保存为：quick_test_results.png")
    
    print("\n" + "=" * 40)
    print("快速测试完成！")
    print("数据加载和基本分析功能正常")
    print("可以继续运行完整的15分钟生活圈分析")

if __name__ == "__main__":
    try:
        quick_analysis()
    except Exception as e:
        print(f"测试过程中出现错误：{e}")
        import traceback
        traceback.print_exc() 