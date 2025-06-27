#!/usr/bin/env python3
"""
路径测试脚本 - 验证数据文件路径
"""

from pathlib import Path
import os

def test_paths():
    """测试不同路径配置"""
    
    print("当前工作目录：", os.getcwd())
    print("脚本位置：", __file__)
    print()
    
    # 从15-Minute-Venice目录运行时的路径
    base_paths = [
        "../venice-data-week-data",
        "../../venice-data-week-data", 
        "/Users/yilinlin/Yilin/Venice_Workshop/venice-data-week-data"
    ]
    
    for base_path in base_paths:
        print(f"测试路径：{base_path}")
        data_dir = Path(base_path)
        
        # 测试关键文件
        files_to_test = [
            data_dir / "1808-sommarioni" / "venice_1808_landregister_geometries.geojson",
            data_dir / "1740-catastici" / "1740_catastici_version20250625.geojson"
        ]
        
        all_exist = True
        for file_path in files_to_test:
            exists = file_path.exists()
            print(f"  {file_path.name}: {'✓' if exists else '✗'}")
            if not exists:
                all_exist = False
                
        if all_exist:
            print(f"  👍 路径 {base_path} 可用！")
            return str(data_dir.resolve())
        print()
    
    print("❌ 没有找到有效的数据路径")
    return None

if __name__ == "__main__":
    correct_path = test_paths()
    if correct_path:
        print(f"\n建议使用的数据路径：{correct_path}") 