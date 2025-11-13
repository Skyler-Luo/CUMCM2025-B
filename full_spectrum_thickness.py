#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于全谱拟合的SiC外延层厚度

物理原理：
- 小波阈值去噪预处理提高数据质量
- 使用标准三项Sellmeier公式计算束缚电子对折射率的贡献
- 结合Drude模型计算自由载流子对复折射率的影响
- 通过传输矩阵法计算单层薄膜的反射光谱
- 使用非线性拟合同时确定外延层厚度d和载流子浓度N
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

# 字体解决方案
from matplotlib.font_manager import FontProperties
import os

def find_chinese_font():
    """
    在系统中查找可用的中文字体文件，并返回一个FontProperties对象。
    """
    if os.name == 'nt':
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/simsun.ttc',
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                print(f"找到系统字体: {font_path}")
                return FontProperties(fname=font_path, size=12)
    try:
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        for font_name in font_names:
            try:
                return FontProperties(font_name, size=12)
            except Exception:
                continue
    except Exception:
        pass
    print("未能在系统中找到任何指定的中文字体，图形中的中文可能无法显示。")
    return None

# 获取中文字体
CHINESE_FONT = find_chinese_font()

# 基础matplotlib配置 - 修复数学表达式负号显示问题
def configure_matplotlib_fonts():
    """配置matplotlib字体，确保中文和数学表达式都能正确显示"""
    # 设置数学文本字体为系统默认，确保负号等符号正确显示
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'
    mpl.rcParams['mathtext.default'] = 'regular'
    
    # 确保负号正确显示
    mpl.rcParams['axes.unicode_minus'] = False
    
    # 设置全局字体大小
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 11
    
    # 提高图形质量
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    
    # 设置更美观的默认样式
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linewidth'] = 0.5
    
    # 改善科学记数法显示
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['axes.formatter.useoffset'] = False

# 应用字体配置
configure_matplotlib_fonts()
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import time

# 小波去噪相关库
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets库未安装，将跳过小波去噪功能")


@dataclass
class MaterialParams:
    """材料参数数据类"""
    # 标准三项Sellmeier系数（束缚电子贡献）
    # n²(λ) = 1 + Σ(A_i * λ² / (λ² - λ_i²))
    A1: float  # 第一项系数
    A2: float  # 第二项系数  
    A3: float  # 第三项系数
    lambda1_sq: float  # 第一项特征波长的平方（μm²）
    lambda2_sq: float  # 第二项特征波长的平方（μm²）
    lambda3_sq: float  # 第三项特征波长的平方（μm²）
    # 物理常数
    m_eff: float  # 有效质量（单位：m_e）
    eps_inf: float  # 高频介电常数


class FullSpectrumAnalyzer:
    """
    全谱拟合SiC外延层分析器
    
    该类实现了基于物理模型的厚度和载流子浓度联合测量，包括：
    - 小波阈值去噪预处理
    - 标准三项Sellmeier公式计算束缚电子折射率贡献
    - Drude模型计算自由载流子对复介电函数的影响
    - 传输矩阵法计算反射率
    - 非线性优化拟合
    - 多角度数据联合分析
    """
    
    # 物理常数
    C_LIGHT = 2.99792458e10  # 光速 cm/s
    E_CHARGE = 1.602176634e-19  # 电子电荷 C
    EPS_0 = 8.854187817e-12  # 真空介电常数 F/m
    M_ELECTRON = 9.10938356e-31  # 电子质量 kg
    
    # 材料参数库
    MATERIALS = {
        'sic': MaterialParams(
            # SiC的三项Sellmeier系数
            A1=0.20075, A2=5.54861, A3=35.65066,
            lambda1_sq=-12.07224, lambda2_sq=0.02641, lambda3_sq=1268.24708,
            m_eff=0.5, eps_inf=6.5
        ),
        'si': MaterialParams(
            # Si的三项Sellmeier系数  
            A1=10.6684293, A2=0.0030434748, A3=1.54133408,
            lambda1_sq=0.09091243, lambda2_sq=1.28764888, lambda3_sq=1218816.0,
            m_eff=0.26, eps_inf=11.7
        )
    }
    
    def __init__(self, material='sic'):
        """
        初始化分析器
        
        Parameters:
        -----------
        material : str
            材料类型，'sic' 或 'si'
        """
        if material not in self.MATERIALS:
            raise ValueError(f"不支持的材料类型: {material}")
        
        self.material_params = self.MATERIALS[material]
        self.material = material
        
        # 小波去噪参数
        self.wavelet_params = {
            'wavelet': 'db4',  # Daubechies小波
            'mode': 'symmetric',  # 边界处理模式
            'threshold_mode': 'soft'  # 软阈值
        }
        
    def calculate_complex_permittivity(self, wavenumber: np.ndarray, 
                                     carrier_density: float, 
                                     scattering_rate: float) -> np.ndarray:
        """
        计算Drude-Sellmeier复介电函数
        
        物理模型：
        ε(ω) = ε_bound(ω) + ε_drude(ω)
        
        其中：
        - ε_bound = n²(λ) = 1 + Σ(A_i*λ²/(λ²-λ_i²))  [三项Sellmeier公式，束缚电子]
        - ε_drude = -ωp²/(ω² + iγω)  [Drude项，自由载流子]
        - ωp² = Ne²/(ε₀m*)           [等离子体频率]
        
        Parameters:
        -----------
        wavenumber : array_like
            波数，单位 cm⁻¹
        carrier_density : float
            载流子浓度，单位 cm⁻³
        scattering_rate : float
            散射率，单位 s⁻¹
            
        Returns:
        --------
        epsilon : ndarray (complex)
            复介电函数
        """
        # Sellmeier项（束缚电子贡献）
        # 波数转换为波长：λ(μm) = 10000 / ν(cm⁻¹)
        wavelength = 10000.0 / wavenumber  # μm
        wavelength_sq = wavelength**2
        
        # 三项Sellmeier公式：n²(λ) = 1 + Σ(A_i*λ²/(λ²-λ_i²))
        params = self.material_params
        
        # 第一项
        denom1 = wavelength_sq - params.lambda1_sq
        denom1 = np.where(np.abs(denom1) < 1e-12, 1e-12, denom1)
        term1 = params.A1 * wavelength_sq / denom1
        
        # 第二项  
        denom2 = wavelength_sq - params.lambda2_sq
        denom2 = np.where(np.abs(denom2) < 1e-12, 1e-12, denom2)
        term2 = params.A2 * wavelength_sq / denom2
        
        # 第三项
        denom3 = wavelength_sq - params.lambda3_sq
        denom3 = np.where(np.abs(denom3) < 1e-12, 1e-12, denom3)
        term3 = params.A3 * wavelength_sq / denom3
        
        # 总的折射率平方（即介电常数的束缚电子部分）
        epsilon_bound = 1.0 + term1 + term2 + term3
        
        # 如果载流子浓度为零，仅返回束缚电子项
        if carrier_density <= 0:
            return epsilon_bound.astype(complex)
        
        # Drude项（自由载流子贡献）
        # 波数转角频率：ω = 2πcν
        omega = 2.0 * np.pi * self.C_LIGHT * wavenumber
        
        # 等离子体频率（SI单位）
        N_m3 = carrier_density * 1e6  # cm⁻³ → m⁻³
        m_star = self.material_params.m_eff * self.M_ELECTRON
        omega_p_squared = (N_m3 * self.E_CHARGE**2) / (self.EPS_0 * m_star)
        
        # Drude项：ε_D = -ωp²/(ω² + iγω)
        denom_drude = omega**2 + 1j * scattering_rate * omega
        epsilon_drude = -omega_p_squared / denom_drude
        
        # 总介电函数
        epsilon_total = epsilon_bound + epsilon_drude
        
        return epsilon_total.astype(complex)
    
    def calculate_refractive_index(self, wavenumber: np.ndarray,
                                 carrier_density: float,
                                 scattering_rate: float) -> np.ndarray:
        """
        计算复折射率 ñ = n + ik
        
        Parameters:
        -----------
        wavenumber : array_like
            波数，单位 cm⁻¹
        carrier_density : float
            载流子浓度，单位 cm⁻³
        scattering_rate : float
            散射率，单位 s⁻¹
            
        Returns:
        --------
        n_complex : ndarray (complex)
            复折射率
        """
        epsilon = self.calculate_complex_permittivity(wavenumber, carrier_density, scattering_rate)
        return np.sqrt(epsilon)
    
    def wavelet_denoise(self, reflectance: np.ndarray, 
                       wavelet: Optional[str] = None,
                       sigma: Optional[float] = None) -> np.ndarray:
        """
        小波阈值去噪处理
        
        Parameters:
        -----------
        reflectance : array_like
            反射率数据
        wavelet : str, optional
            小波类型，默认使用'db4'
        sigma : float, optional
            噪声标准差，如果为None则自动估计
            
        Returns:
        --------
        denoised_reflectance : ndarray
            去噪后的反射率数据
        """
        if not PYWT_AVAILABLE:
            warnings.warn("PyWavelets库不可用，跳过小波去噪")
            return reflectance.copy()
        
        try:
            # 使用默认参数或传入参数
            wavelet_type = wavelet or self.wavelet_params['wavelet']
            mode = self.wavelet_params['mode']
            threshold_mode = self.wavelet_params['threshold_mode']
            
            # 小波分解
            coeffs = pywt.wavedec(reflectance, wavelet_type, mode=mode)
            
            # 估计噪声标准差
            if sigma is None:
                # 使用最高频细节系数估计噪声
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # 计算阈值（软阈值）
            threshold = sigma * np.sqrt(2 * np.log(len(reflectance)))
            
            # 对细节系数进行阈值处理
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode=threshold_mode) 
                                for detail in coeffs[1:]]
            
            # 小波重构
            denoised = pywt.waverec(coeffs_thresh, wavelet_type, mode=mode)
            
            # 确保输出长度与输入相同
            if len(denoised) != len(reflectance):
                denoised = denoised[:len(reflectance)]
            
            return denoised
            
        except Exception as e:
            warnings.warn(f"小波去噪失败: {e}，返回原始数据")
            return reflectance.copy()
    
    
    def transfer_matrix_single_layer(self, wavenumber: np.ndarray,
                                   thickness: float,
                                   n_film: np.ndarray,
                                   n_substrate: np.ndarray,
                                   incident_angle: float,
                                   n_ambient: float = 1.0) -> np.ndarray:
        """
        使用传输矩阵法计算单层薄膜的反射率
        
        物理原理：
        - 基于Maxwell方程和边界条件
        - 考虑薄膜内的多次反射
        - 适用于任意复折射率
        
        Parameters:
        -----------
        wavenumber : array_like
            波数，单位 cm⁻¹
        thickness : float
            薄膜厚度，单位 μm
        n_film : array_like (complex)
            薄膜复折射率
        n_substrate : array_like (complex)
            衬底复折射率
        incident_angle : float
            入射角，单位度
        n_ambient : float
            环境介质折射率（默认空气，n=1）
            
        Returns:
        --------
        reflectance : ndarray
            反射率（实数，0-1）
        """
        # 角度转弧度
        theta0 = np.radians(incident_angle)
        
        # 波长（cm）
        wavelength = 1.0 / wavenumber
        
        # 厚度转换：μm → cm
        d_cm = thickness * 1e-4
        
        # Snell定律计算各层中的折射角
        sin_theta0 = np.sin(theta0)
        
        # 薄膜中的折射角（复数）
        sin_theta1 = n_ambient * sin_theta0 / n_film
        cos_theta1 = np.sqrt(1.0 - sin_theta1**2)
        
        # 衬底中的折射角
        sin_theta2 = n_ambient * sin_theta0 / n_substrate  
        cos_theta2 = np.sqrt(1.0 - sin_theta2**2)
        
        # 菲涅尔系数：环境-薄膜界面
        r01 = (n_ambient * np.cos(theta0) - n_film * cos_theta1) / \
              (n_ambient * np.cos(theta0) + n_film * cos_theta1)
        t01 = 2 * n_ambient * np.cos(theta0) / \
              (n_ambient * np.cos(theta0) + n_film * cos_theta1)
        
        # 菲涅尔系数：薄膜-衬底界面
        r12 = (n_film * cos_theta1 - n_substrate * cos_theta2) / \
              (n_film * cos_theta1 + n_substrate * cos_theta2)
        t12 = 2 * n_film * cos_theta1 / \
              (n_film * cos_theta1 + n_substrate * cos_theta2)
        
        # 薄膜中的相位厚度
        beta = 2 * np.pi * n_film * d_cm * cos_theta1 / wavelength
        
        # 薄膜传输矩阵元素
        M11 = np.cos(beta)
        M12 = 1j * np.sin(beta) / (n_film * cos_theta1)
        M21 = 1j * n_film * cos_theta1 * np.sin(beta)
        M22 = np.cos(beta)
        
        # 总的反射系数
        # 考虑薄膜传输矩阵和界面反射
        Y = n_substrate * cos_theta2  # 衬底导纳
        
        # 从传输矩阵计算总导纳
        Y_total = (M21 + M22 * Y) / (M11 + M12 * Y)
        
        # 总反射系数
        r_total = (n_ambient * np.cos(theta0) - Y_total) / \
                  (n_ambient * np.cos(theta0) + Y_total)
        
        # 反射率
        reflectance = np.abs(r_total)**2
        
        return reflectance
    
    def forward_model(self, wavenumber: np.ndarray, params: Dict,
                     incident_angle: float) -> np.ndarray:
        """
        前向模型：根据参数计算理论反射光谱
        
        Parameters:
        -----------
        wavenumber : array_like
            波数，单位 cm⁻¹
        params : dict
            拟合参数字典，包含：
            - 'd': 厚度 (μm)
            - 'N_epi': 外延层载流子浓度 (cm⁻³)
            - 'gamma_epi': 外延层散射率 (s⁻¹)
            - 'N_sub': 衬底载流子浓度 (cm⁻³)
            - 'gamma_sub': 衬底散射率 (s⁻¹)
        incident_angle : float
            入射角度
            
        Returns:
        --------
        reflectance : ndarray
            理论反射率
        """
        # 计算外延层复折射率
        n_epi = self.calculate_refractive_index(
            wavenumber, params['N_epi'], params['gamma_epi']
        )
        
        # 计算衬底复折射率
        n_sub = self.calculate_refractive_index(
            wavenumber, params['N_sub'], params['gamma_sub']
        )
        
        # 使用传输矩阵法计算反射率
        reflectance = self.transfer_matrix_single_layer(
            wavenumber, params['d'], n_epi, n_sub, incident_angle
        )
        
        return reflectance
    
    def objective_function(self, x: np.ndarray, wavenumber: np.ndarray,
                          reflectance_data: np.ndarray, incident_angle: float,
                          param_names: List[str], weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        目标函数：计算模型与数据的残差
        
        Parameters:
        -----------
        x : array_like
            待拟合参数数组
        wavenumber : array_like
            波数数据
        reflectance_data : array_like
            实测反射率数据
        incident_angle : float
            入射角度
        param_names : list
            参数名称列表
        weights : array_like, optional
            数据点权重
            
        Returns:
        --------
        residuals : ndarray
            残差数组
        """
        # 构建参数字典
        params = dict(zip(param_names, x))
        
        try:
            # 计算理论光谱
            model_reflectance = self.forward_model(wavenumber, params, incident_angle)
            # 计算残差
            residuals = model_reflectance - reflectance_data
            # 应用权重
            if weights is not None:
                residuals *= weights
                
            return residuals
            
        except Exception as e:
            warnings.warn(f"前向模型计算失败: {e}")
            return np.full_like(reflectance_data, 1e6)
    
    def fit_spectrum(self, wavenumber: np.ndarray, reflectance: np.ndarray,
                    incident_angle: float, initial_guess: Optional[Dict] = None,
                    bounds: Optional[Dict] = None, method: str = 'lm',
                    use_wavelet_denoise: bool = True) -> Dict:
        """
        拟合单个角度的反射光谱
        
        Parameters:
        -----------
        wavenumber : array_like
            波数，单位 cm⁻¹
        reflectance : array_like
            反射率，单位 %
        incident_angle : float
            入射角，单位度
        initial_guess : dict, optional
            初始参数猜测
        bounds : dict, optional
            参数边界
        method : str
            优化方法：'lm'(Levenberg-Marquardt) 或 'global'(全局优化)
        use_wavelet_denoise : bool
            是否使用小波去噪预处理
            
        Returns:
        --------
        result : dict
            拟合结果，包含参数值、不确定性、拟合质量等
        """
        # 数据预处理
        reflectance_original = reflectance.copy()  # 保存原始数据
        reflectance = reflectance / 100.0  # % → 小数
        
        # 小波去噪预处理
        reflectance_denoised = reflectance.copy()
        
        if use_wavelet_denoise and PYWT_AVAILABLE:
            print("应用小波阈值去噪...")
            reflectance_denoised = self.wavelet_denoise(reflectance)
            print("  小波去噪处理完成")
            
            # 使用去噪后的数据进行拟合
            reflectance = reflectance_denoised
        elif use_wavelet_denoise and not PYWT_AVAILABLE:
            print("警告: 小波去噪功能不可用，使用原始数据")
        
        # 默认初始猜测
        if initial_guess is None:
            initial_guess = {
                'd': 10.0,          # μm
                'N_epi': 1e16,      # cm⁻³
                'gamma_epi': 1e13,  # s⁻¹
                'N_sub': 1e19,      # cm⁻³
                'gamma_sub': 1e14   # s⁻¹
            }
        
        # 默认边界
        if bounds is None:
            bounds = {
                'd': (2.0, 10.0),
                'N_epi': (1e14, 1e18),
                'gamma_epi': (1e11, 1e15),
                'N_sub': (1e17, 1e21),
                'gamma_sub': (1e12, 1e16)
            }
        
        # 构建参数数组和边界
        param_names = list(initial_guess.keys())
        x0 = np.array([initial_guess[name] for name in param_names])
        lower_bounds = np.array([bounds[name][0] for name in param_names])
        upper_bounds = np.array([bounds[name][1] for name in param_names])
        
        print(f"开始拟合 (入射角 {incident_angle}°)...")
        print(f"初始参数: {dict(zip(param_names, x0))}")
        
        start_time = time.time()
        
        if method == 'global':
            # 全局优化（差分进化算法）
            def objective_wrapper(x):
                residuals = self.objective_function(
                    x, wavenumber, reflectance, incident_angle, param_names
                )
                return np.sum(residuals**2)
            
            opt_result = differential_evolution(
                objective_wrapper,
                bounds=list(zip(lower_bounds, upper_bounds)),
                seed=42,
                maxiter=500,
                popsize=20
            )
            x_opt = opt_result.x
            success = opt_result.success
            cost = opt_result.fun
        else:
            # 局部优化（Levenberg-Marquardt）
            opt_result = least_squares(
                self.objective_function,
                x0,
                args=(wavenumber, reflectance, incident_angle, param_names),
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                max_nfev=2000
            )
            x_opt = opt_result.x
            success = opt_result.success
            cost = opt_result.cost
        elapsed_time = time.time() - start_time
        # 构建结果字典
        fitted_params = dict(zip(param_names, x_opt))
        # 计算拟合质量指标
        model_reflectance = self.forward_model(wavenumber, fitted_params, incident_angle)
        residuals = model_reflectance - reflectance
        
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        r_squared = 1 - np.var(residuals) / np.var(reflectance)
        
        result = {
            'success': success,
            'params': fitted_params,
            'cost': cost,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'residuals': residuals,
            'model_reflectance': model_reflectance,
            'elapsed_time': elapsed_time,
            'method': method,
            'denoising': {
                'applied': use_wavelet_denoise and PYWT_AVAILABLE,
                'original_reflectance': reflectance_original,
                'denoised_reflectance': reflectance_denoised * 100.0 if use_wavelet_denoise else None
            }
        }
        
        print(f"拟合完成 ({elapsed_time:.1f}s)")
        print(f"成功: {success}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r_squared:.6f}")
        print(f"拟合参数:")
        for name, value in fitted_params.items():
            if name == 'd':
                print(f"  {name}: {value:.3f} μm")
            elif 'N_' in name:
                # 使用安全的科学记数法格式化，避免负号显示问题
                exp_str = f"{value:.2e}".replace('e-', 'e−').replace('e+', 'e+')
                print(f"  {name}: {exp_str} cm⁻³")
            elif 'gamma_' in name:
                exp_str = f"{value:.2e}".replace('e-', 'e−').replace('e+', 'e+')
                print(f"  {name}: {exp_str} s⁻¹")
        
        return result
    
    def bootstrap_uncertainty(self, wavenumber: np.ndarray, reflectance: np.ndarray,
                            incident_angle: float, fitted_params: Dict,
                            n_bootstrap: int = 100) -> Dict:
        """
        使用Bootstrap方法估算参数不确定性
        
        Parameters:
        -----------
        wavenumber : array_like
            波数数据
        reflectance : array_like
            反射率数据
        incident_angle : float
            入射角度
        fitted_params : dict
            拟合得到的参数
        n_bootstrap : int
            Bootstrap采样次数
            
        Returns:
        --------
        uncertainty : dict
            参数不确定性统计
        """
        print(f"开始Bootstrap不确定性分析 (N={n_bootstrap})...")
        
        # 计算残差
        model_reflectance = self.forward_model(wavenumber, fitted_params, incident_angle)
        residuals = reflectance / 100.0 - model_reflectance
        
        # 残差的标准差（作为噪声水平估计）
        noise_std = np.std(residuals)
        
        param_names = list(fitted_params.keys())
        bootstrap_params = []
        
        rng = np.random.default_rng(42)
        
        for i in range(n_bootstrap):
            # 生成带噪声的数据
            noise = rng.normal(0, noise_std, len(reflectance))
            noisy_reflectance = (model_reflectance + noise) * 100.0
            
            try:
                # 拟合带噪声的数据
                result = self.fit_spectrum(
                    wavenumber, noisy_reflectance, incident_angle, 
                    initial_guess=fitted_params, method='lm'
                )
                
                if result['success']:
                    bootstrap_params.append([result['params'][name] for name in param_names])
                    
            except Exception:
                continue
        
        if len(bootstrap_params) < 10:
            warnings.warn("Bootstrap样本数量不足，不确定性估计可能不可靠")
        
        bootstrap_params = np.array(bootstrap_params)
        
        # 计算统计量
        uncertainty = {}
        for i, name in enumerate(param_names):
            values = bootstrap_params[:, i]
            uncertainty[name] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'ci95': np.percentile(values, [2.5, 97.5])
            }
        
        print(f"Bootstrap分析完成 (有效样本: {len(bootstrap_params)}/{n_bootstrap})")
        
        return uncertainty


def load_spectrum_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载光谱数据文件
    
    Parameters:
    -----------
    filepath : str
        CSV文件路径
        
    Returns:
    --------
    wavenumber : ndarray
        波数，单位 cm⁻¹
    reflectance : ndarray
        反射率，单位 %
    """
    df = pd.read_csv(filepath)
    wavenumber = df.iloc[:, 0].values
    reflectance = df.iloc[:, 1].values
    
    # 过滤NaN值
    mask = ~(np.isnan(wavenumber) | np.isnan(reflectance))
    wavenumber = wavenumber[mask]
    reflectance = reflectance[mask]
    
    # 按波数排序
    sort_idx = np.argsort(wavenumber)
    wavenumber = wavenumber[sort_idx]
    reflectance = reflectance[sort_idx]
    
    return wavenumber, reflectance


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于全谱拟合的SiC外延层厚度和载流子浓度测量')
    parser.add_argument('file', help='CSV文件路径')
    parser.add_argument('--angle', type=float, default=10.0, help='入射角度 (默认10°)')
    parser.add_argument('--material', choices=['sic', 'si'], default='sic', help='材料类型')
    parser.add_argument('--method', choices=['lm', 'global'], default='lm', help='优化方法')
    parser.add_argument('--bootstrap', type=int, default=50, help='Bootstrap采样次数')
    parser.add_argument('--plot', action='store_true', help='显示拟合结果图')
    parser.add_argument('--range', nargs=2, type=float, help='波数范围 (最小值 最大值)')
    parser.add_argument('--no-denoise', action='store_true', help='禁用小波去噪预处理')
    parser.add_argument('--wavelet', default='db4', help='小波类型 (默认db4)')
    
    # 图片保存相关参数
    parser.add_argument('--save-figures', action='store_true', help='保存拟合结果图片')
    parser.add_argument('--output-dir', default='data/figures', help='图片输出目录 (默认data/figures)')
    parser.add_argument('--figure-format', choices=['png', 'pdf', 'svg', 'jpg'], default='png', help='图片格式 (默认png)')
    parser.add_argument('--figure-dpi', type=int, default=300, help='图片分辨率 (默认300)')
    parser.add_argument('--save-data', action='store_true', help='保存拟合数据到CSV文件')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载光谱数据: {args.file}")
    wavenumber, reflectance = load_spectrum_data(args.file)
    
    # 应用波数范围过滤
    if args.range:
        mask = (wavenumber >= args.range[0]) & (wavenumber <= args.range[1])
        wavenumber = wavenumber[mask]
        reflectance = reflectance[mask]
        print(f"应用波数范围过滤: {args.range[0]}-{args.range[1]} cm⁻¹")
    
    print(f"数据点数量: {len(wavenumber)}")
    print(f"波数范围: {wavenumber.min():.1f} - {wavenumber.max():.1f} cm⁻¹")
    print(f"反射率范围: {reflectance.min():.1f} - {reflectance.max():.1f} %")
    
    # 创建分析器
    analyzer = FullSpectrumAnalyzer(material=args.material)
    
    # 设置小波参数
    if args.wavelet != 'db4':
        analyzer.wavelet_params['wavelet'] = args.wavelet
    
    # 执行拟合
    use_denoise = not args.no_denoise
    result = analyzer.fit_spectrum(
        wavenumber, reflectance, args.angle, 
        method=args.method, use_wavelet_denoise=use_denoise
    )
    
    if not result['success']:
        print("警告: 拟合未收敛，结果可能不可靠")
    
    # Bootstrap不确定性分析
    if args.bootstrap > 0:
        # 使用与拟合相同的数据（原始或去噪后）
        bootstrap_reflectance = reflectance
        if result['denoising']['applied']:
            bootstrap_reflectance = result['denoising']['denoised_reflectance']
            
        uncertainty = analyzer.bootstrap_uncertainty(
            wavenumber, bootstrap_reflectance, args.angle, 
            result['params'], n_bootstrap=args.bootstrap
        )
        
        def safe_format_e(value):
            """将科学记数法中的'e'替换为'×10^'，用于终端输出，避免乱码"""
            return f"{value:.2e}".replace('e', '×10^')

        print("\n=== 参数不确定性 (Bootstrap) ===")
        for name, stats in uncertainty.items():
            if name == 'd':
                print(f"{name}: {stats['mean']:.3f} ± {stats['std']:.3f} μm "
                      f"(95% CI: [{stats['ci95'][0]:.3f}, {stats['ci95'][1]:.3f}])")
            elif 'N_' in name:
                mean_str = safe_format_e(stats['mean'])
                std_str = safe_format_e(stats['std'])
                ci_low_str = safe_format_e(stats['ci95'][0])
                ci_high_str = safe_format_e(stats['ci95'][1])
                print(f"{name}: {mean_str} ± {std_str} cm⁻³ "
                      f"(95% CI: [{ci_low_str}, {ci_high_str}])")
            elif 'gamma_' in name:
                mean_str = safe_format_e(stats['mean'])
                std_str = safe_format_e(stats['std'])
                ci_low_str = safe_format_e(stats['ci95'][0])
                ci_high_str = safe_format_e(stats['ci95'][1])
                print(f"{name}: {mean_str} ± {std_str} s⁻¹ "
                      f"(95% CI: [{ci_low_str}, {ci_high_str}])")
    
    # 可视化
    if args.plot:
        # 应用美学样式
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        else:
            # 使用更现代的样式
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                plt.style.use('default')

        # 主要图：反射率与拟合模型，右侧为残差直方图，底部为残差随波数
        fig = plt.figure(constrained_layout=False, figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        ax_main = fig.add_subplot(gs[0:2, 0:2])
        ax_residual_line = fig.add_subplot(gs[2, 0:2], sharex=ax_main)
        ax_resid_hist = fig.add_subplot(gs[0:2, 2])

        # 定义颜色方案
        colors = {
            'data': '#2E86AB',      # 深蓝色
            'model': '#A23B72',     # 深红紫色
            'residual': '#F18F01',  # 橙色
            'hist': '#C73E1D'       # 深红色
        }

        # 主图：数据与模型
        ax_main.plot(wavenumber, reflectance, color=colors['data'], linewidth=1.5, 
                     alpha=0.8, label='实测数据', zorder=2)
        ax_main.plot(wavenumber, result['model_reflectance'] * 100, color=colors['model'], 
                     linewidth=2, alpha=0.9, label=f'拟合模型', zorder=3)
        
        ax_main.set_ylabel('反射率 (%)', fontproperties=CHINESE_FONT, fontsize=12)
        title = f'全谱拟合结果 - {args.material.upper()} (θ={args.angle}°)'
        if result['denoising']['applied']:
            title += '  [小波去噪]'
        ax_main.set_title(title, fontproperties=CHINESE_FONT, fontsize=14, pad=15)
        ax_main.legend(frameon=True, prop=CHINESE_FONT, loc='best', 
                      framealpha=0.9, fancybox=True)
        ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_main.set_facecolor('#FAFAFA')

        # 底部：残差随波数
        residuals_pct = result['residuals'] * 100
        ax_residual_line.plot(wavenumber, residuals_pct, color=colors['residual'], 
                             linewidth=1.5, alpha=0.8)
        ax_residual_line.axhline(0, color='k', linestyle='--', alpha=0.7, linewidth=1)
        ax_residual_line.set_xlabel('波数 (cm⁻¹)', fontproperties=CHINESE_FONT, fontsize=12)
        ax_residual_line.set_ylabel('残差 (%)', fontproperties=CHINESE_FONT, fontsize=12)
        ax_residual_line.set_title(f'拟合残差 (RMSE={result["rmse"]:.4f})', 
                                  fontproperties=CHINESE_FONT, fontsize=12)
        ax_residual_line.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_residual_line.set_facecolor('#FAFAFA')

        # 右侧：残差直方图与正态拟合
        n, bins, patches = ax_resid_hist.hist(residuals_pct, bins=35, color=colors['hist'], 
                                             alpha=0.7, density=True, edgecolor='white', linewidth=0.5)
        ax_resid_hist.set_title('残差分布', fontproperties=CHINESE_FONT, fontsize=12)
        ax_resid_hist.set_xlabel('残差 (%)', fontproperties=CHINESE_FONT, fontsize=11)
        ax_resid_hist.set_ylabel('概率密度', fontproperties=CHINESE_FONT, fontsize=11)
        ax_resid_hist.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_resid_hist.set_facecolor('#FAFAFA')

        # 添加正态分布拟合曲线
        mu, sigma = np.mean(residuals_pct), np.std(residuals_pct)
        x_norm = np.linspace(residuals_pct.min(), residuals_pct.max(), 100)
        y_norm = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - mu)/sigma)**2)
        ax_resid_hist.plot(x_norm, y_norm, color='red', linewidth=2, linestyle='--', 
                          alpha=0.8, label='正态拟合')
        ax_resid_hist.legend(prop=CHINESE_FONT, frameon=True, framealpha=0.9)

        # 参数文本框已移除以避免遮挡曲线图

        plt.tight_layout()

        # 保存图形到 data/figures，并分别导出单独面板以便检视
        try:
            import os
            out_dir = os.path.join(os.path.dirname(__file__), 'data', 'figures')
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.file))[0]

            out_path = os.path.join(out_dir, f'{base_name}_fit_summary.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"已保存拟合图像: {out_path}")

            # 分别保存各子图，便于单独查看中文/负号问题
            try:
                ax_main.figure = fig
                main_path = os.path.join(out_dir, f'{base_name}_main.png')
                bbox_main = ax_main.get_tightbbox(fig.canvas.get_renderer())
                fig.savefig(main_path, dpi=300, bbox_inches=bbox_main.expanded(1.1, 1.05))
                print(f"已保存主图: {main_path}")
            except Exception:
                # 退回到整图截取
                pass

            try:
                resid_line_path = os.path.join(out_dir, f'{base_name}_residuals_line.png')
                bbox_line = ax_residual_line.get_tightbbox(fig.canvas.get_renderer())
                fig.savefig(resid_line_path, dpi=300, bbox_inches=bbox_line.expanded(1.1, 1.05))
                print(f"已保存残差折线图: {resid_line_path}")
            except Exception:
                pass

            try:
                resid_hist_path = os.path.join(out_dir, f'{base_name}_residuals_hist.png')
                bbox_hist = ax_resid_hist.get_tightbbox(fig.canvas.get_renderer())
                fig.savefig(resid_hist_path, dpi=300, bbox_inches=bbox_hist.expanded(1.1, 1.05))
                print(f"已保存残差直方图: {resid_hist_path}")
            except Exception:
                pass

        except Exception as e:
            warnings.warn(f"保存图像失败: {e}")

        plt.show()


if __name__ == '__main__':
    main()