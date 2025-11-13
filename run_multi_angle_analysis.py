#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多角度测量一致性分析脚本

功能:
1. 对多个不同入射角的光谱数据（如附件1和附件2）分别进行厚度拟合。
2. 收集每个角度的拟合结果（厚度 d, 不确定度 σ_d）。
3. 评估单次测量的内部可靠性（CV）。
4. 进行角度间的一致性检验（相对差异、置信区间重叠）。
5. 使用加权平均法合并一致的结果，给出一个更可靠的最终厚度值。
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import List, Dict, Any

# 确保 iterative_extremum_fitter.py 在同一目录或Python路径下
try:
    from iterative_extremum_fitter import IterativeExtremumFitter, load_spectrum_data
except ImportError:
    print("无法导入 'IterativeExtremumFitter'。")
    exit(1)

@dataclass
class AngleResult:
    """一个数据类，用于存储单次角度拟合的结果"""
    angle: float
    filepath: str
    thickness: float
    uncertainty: float
    fit_result: Dict[str, Any]  # 存储完整的拟合结果以备后续分析

def analyze_consistency(results: List[AngleResult]):
    """
    分析来自多个角度的厚度测量结果的一致性。

    Args:
        results: AngleResult 对象的列表。
    """
    if len(results) < 2:
        print("需要至少两个角度的测量结果才能进行一致性分析。")
        return

    print("\n" + "="*60)
    print(" 多角度测量一致性与合并结果分析")
    print("="*60)

    # 次测量内部可靠性 使用变异系数(CV)评估
    print("\n--- 1. 单次测量内部可靠性评估 ---")
    all_reliable = True
    for res in results:
        cv = (res.uncertainty / res.thickness) * 100 if res.thickness > 0 else float('inf')
        if cv < 5:
            reliability = "高"
        elif cv < 10:
            reliability = "中等"
        else:
            reliability = "低"
        
        print(f"角度 {res.angle}°: d = {res.thickness:.3f} ± {res.uncertainty:.3f} μm (CV = {cv:.2f}%, 可靠性: {reliability})")
        if reliability == "低":
            all_reliable = False
    
    if not all_reliable:
        print("\n警告：存在低可靠性的单次测量，最终合并结果的置信度可能下降。")

    # 将主要对前两个结果进行配对比较
    res1, res2 = results[0], results[1]
    d1, sigma1 = res1.thickness, res1.uncertainty
    d2, sigma2 = res2.thickness, res2.uncertainty

    # 角度间一致性检验 
    print("\n--- 2. 角度间一致性检验 (比较前两个结果) ---")
    
    # 相对差异
    relative_diff = np.abs(d1 - d2) / ((d1 + d2) / 2) * 100
    print(f"相对差异 (|d1-d2|/mean(d1,d2)): {relative_diff:.2f}%")
    if relative_diff > 5.0:
        print("  -> 警告：不同角度测量的厚度相对差异较大 (>5%)，结果可能不一致。")
    else:
        print("  -> 信息：厚度测量值具有良好的一致性。")

    # 置信区间重叠检验
    ci1 = (d1 - sigma1, d1 + sigma1)
    ci2 = (d2 - sigma2, d2 + sigma2)
    overlap = max(ci1[0], ci2[0]) < min(ci1[1], ci2[1])
    print(f"1-sigma 置信区间重叠: {'是' if overlap else '否'}")
    if not overlap:
        print("  -> 警告：两个测量结果的1-sigma置信区间不重叠，可能存在系统偏差。")

    # 最终推荐值 (加权平均)
    print("\n--- 3. 最终推荐值 (基于所有高/中等可靠性结果) ---")
    
    # 筛选出可靠的结果用于最终计算
    reliable_results = [r for r in results if (r.uncertainty / r.thickness * 100) < 10]
    
    if len(reliable_results) < 1:
        print("没有可靠的测量结果可用于计算最终值。")
        return

    weights = np.array([1 / (res.uncertainty**2) for res in reliable_results])
    thicknesses = np.array([res.thickness for res in reliable_results])

    # 加权平均厚度
    d_final = np.sum(weights * thicknesses) / np.sum(weights)
    
    # 合并不确定度
    sigma_final = np.sqrt(1 / np.sum(weights))
    
    # 最终变异系数 (CV)
    cv_final = (sigma_final / d_final) * 100
    if cv_final < 5:
        final_reliability = "高"
    elif cv_final < 10:
        final_reliability = "中等"
    else:
        final_reliability = "低"

    print(f"最终厚度 (加权平均): {d_final:.4f} ± {sigma_final:.4f} μm")
    print(f"最终变异系数 (CV): {cv_final:.2f}%")
    print(f"最终结果可靠性: {final_reliability}")
    print("="*60)


def main():
    """主执行函数"""
    # 定义待分析的数据文件和对应的入射角
    measurements = [
        {'file': os.path.join('data', '附件1.csv'), 'angle': 10.0},
        {'file': os.path.join('data', '附件2.csv'), 'angle': 15.0},
    ]
    
    fitter = IterativeExtremumFitter(material='sic')
    all_results: List[AngleResult] = []

    for measurement in measurements:
        filepath = measurement['file']
        angle = measurement['angle']
        
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在 {filepath}，跳过该测量。")
            continue

        print(f"\n>>> 正在处理文件: {filepath} @ {angle}° <<<")
        
        wavenumber, reflectance = load_spectrum_data(filepath)
        
        result = fitter.fit_spectrum(
            wavenumber, reflectance, angle, use_denoise=True
        )
        
        if result.get('success'):
            d = result['params']['d']
            sigma_d = result['param_uncertainties'].get('d', float('nan'))
            
            # 检查不确定度是否有效
            if np.isnan(sigma_d):
                print(f"警告: 角度 {angle}° 的不确定度计算失败，该结果将不用于最终合并。")
                continue

            all_results.append(AngleResult(
                angle=angle,
                filepath=filepath,
                thickness=d,
                uncertainty=sigma_d,
                fit_result=result
            ))
            print(f"拟合成功: d = {d:.3f} ± {sigma_d:.3f} μm")
        else:
            print(f"拟合失败: {result.get('message', '未知错误')}")

    # 当所有拟合都完成后，进行一致性分析
    if len(all_results) > 1:
        analyze_consistency(all_results)
    else:
        print("\n未能成功完成足够的拟合来进行一致性分析。")

if __name__ == '__main__':
    main()