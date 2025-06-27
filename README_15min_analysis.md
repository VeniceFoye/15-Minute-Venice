# 威尼斯15分钟生活圈分析

基于1808年威尼斯土地登记册数据和1740年POI数据的历史15分钟生活圈分析。

## 项目概述

这个项目结合了您现有的栅格地图技术，分析1808年威尼斯每栋建筑的15分钟步行可达范围，并统计其中包含的1740年POI（兴趣点）类型分布。

### 历史背景
- **1808年数据**：拿破仑时期的威尼斯土地登记册，包含详细的建筑和街道信息
- **1740年POI**：威尼斯Catastici数据，包含各种商业、宗教、公共设施的位置
- **历史步行速度**：约3 km/h（比现代步行速度略慢，考虑到历史时期的街道条件）

## 文件结构

```
15-Minute-Venice/
├── src/
│   ├── fifteen_minute_analysis.py    # 主分析类
│   ├── gridify.py                    # 栅格化工具（已有）
│   └── 1808_gridify.py              # 1808年数据处理（已有）
├── demo_analysis.py                  # 演示脚本
├── README_15min_analysis.md         # 本说明文档
└── results/                         # 分析结果输出目录
    ├── 15min_analysis_results.csv   # 详细分析结果
    ├── analysis_report.txt          # 统计报告
    └── analysis_results.png         # 可视化图表
```

## 使用方法

### 1. 环境准备

确保安装了必要的Python包：
```bash
pip install geopandas pandas numpy scipy matplotlib seaborn shapely rasterio
```

### 2. 数据准备

确保以下数据文件存在：
- `../venice-data-week-data/1808-sommarioni/venice_1808_landregister_geometries.geojson`
- `../venice-data-week-data/1740-catastici/1740_catastici_version20250625.geojson`

### 3. 运行分析

```bash
cd 15-Minute-Venice
python demo_analysis.py
```

## 分析流程

### 1. 数据加载和预处理
- 加载1808年的建筑、街道、运河数据
- 加载1740年的POI数据
- 统一坐标系为UTM Zone 33N (EPSG:32633)

### 2. 栅格化
- 将矢量数据转换为栅格地图
- 栅格大小：3米×3米（平衡精度和计算效率）
- 分类：海洋(0)、街道(1)、建筑(2)、运河(3)、庭院(4)

### 3. 可达性分析
- 为每栋建筑计算15分钟步行可达范围
- 步行速度：3 km/h（历史速度）
- 最大距离：750米
- 使用距离传播算法计算可达区域

### 4. POI分析
- 统计每个15分钟生活圈内的POI数量和类型
- 计算POI类型多样性
- 分析生活便利性等级

## 输出结果

### 1. CSV数据文件 (`results/15min_analysis_results.csv`)
包含每栋建筑的以下信息：
- `building_id`: 建筑编号
- `x`, `y`: 建筑坐标
- `accessible_area_m2`: 15分钟可达面积
- `accessible_buildings`: 可达建筑数量
- `accessible_streets`: 可达街道长度
- `total_pois`: 生活圈内POI总数
- `poi_diversity`: POI类型多样性
- `poi_counts`: 各类型POI详细计数

### 2. 统计报告 (`analysis_report.txt`)
包含整体统计信息：
- 平均可达面积和分布
- POI可达性统计
- 生活便利性分级

### 3. 可视化图表 (`analysis_results.png`)
包含四个子图：
- 15分钟可达面积分布直方图
- 生活圈内POI数量分布
- POI类型多样性分布
- 可达面积vs POI数量散点图

## 分析参数调整

您可以在`demo_analysis.py`中调整以下参数：

```python
analyzer = FifteenMinuteVenice(
    data_dir=data_dir,
    cell_size=3.0,          # 栅格大小（米）
    walk_speed_kmh=3.0,     # 步行速度（km/h）
    time_limit_min=15.0,    # 时间限制（分钟）
    target_crs=32633        # 坐标系
)
```

## 应用场景

### 历史城市规划研究
- 比较不同历史时期的城市可达性
- 分析城市形态演变对生活便利性的影响

### POI分布研究
- 研究1740年不同类型设施的空间分布模式
- 分析商业、宗教、公共设施的可达性差异

### 15分钟城市理念的历史验证
- 验证历史威尼斯是否符合现代15分钟城市理念
- 为现代城市规划提供历史参考

## 技术特点

1. **历史数据融合**：结合两个不同历史时期的数据
2. **栅格化分析**：利用您现有的高效栅格化技术
3. **可达性算法**：简化的距离传播算法，适合威尼斯网格状街道
4. **空间分析**：综合考虑地理位置和POI类型分布

## 后续扩展

1. **时间序列分析**：加入更多历史时期数据
2. **交通模式**：考虑船只交通在运河上的可达性
3. **社会经济分析**：结合人口和经济数据
4. **可视化增强**：创建交互式地图和动态可视化

## 问题排除

如果遇到导入错误，请确保：
1. Python路径设置正确
2. 所有依赖包已安装
3. 数据文件路径正确

如果分析速度较慢，可以：
1. 增大`cell_size`参数
2. 减少分析的建筑数量（用于测试）
3. 使用更高配置的计算机 