#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极值点解析迭代优化法

物理原理：
- 基于极值点的干涉条件，但考虑了折射率n是波数ν的函数 n=n(ν)。
- 由于n(ν)依赖于未知的载流子浓度(N)和散射率(γ)，导致厚度(d)与这些参数耦合，无法直接求解。
- 本方法将 d, N, γ 等作为未知参数，通过优化算法寻找最佳参数组合。
- 优化目标：使得在所有实验光谱的极值点上，根据物理模型计算出的干涉级数 m 最接近于整数（或半整数）。

算法核心：
1. 从实验光谱中精确提取极大值和极小值点的位置（波数）。
2. 定义一个目标函数，衡量在给定参数(d, N_epi, γ_epi)下，所有极值点计算出的干涉级数偏离整数/半整数的程度。
3. 使用全局优化算法（差分进化）来最小化该目标函数，从而找到最佳的物理参数。
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import logging
import time
import warnings
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from matplotlib.font_manager import FontProperties

# 小波去噪相关库
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets库未安装，将跳过小波去噪功能")

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
    return None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CHINESE_FONT = find_chinese_font()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('default')

def load_spectrum_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    wavenumber = df.iloc[:, 0].values
    reflectance = df.iloc[:, 1].values
    mask = ~(np.isnan(wavenumber) | np.isnan(reflectance))
    wavenumber = wavenumber[mask]
    reflectance = reflectance[mask]
    sort_idx = np.argsort(wavenumber)
    wavenumber = wavenumber[sort_idx]
    reflectance = reflectance[sort_idx]
    return wavenumber, reflectance

@dataclass
class MaterialParams:
    """材料参数数据类"""
    A: float
    B: float  
    C: float
    m_eff: float
    eps_inf: float
    # Sellmeier 系数列表，每项为 (B_j, C_j) 对，应以波长 (µm) 为单位 (C_j 为 µm^2)
    sellmeier_terms: Optional[List[Tuple[float, float]]] = None

class IterativeExtremumFitter:
    """
    基于极值点迭代优化的拟合器
    """
    C_LIGHT = 2.99792458e10
    E_CHARGE = 1.602176634e-19
    EPS_0 = 8.854187817e-12
    M_ELECTRON = 9.10938356e-31
    
    # 为 sic 提供 topic.md 中的 Sellmeier 系数（lambda 单位 µm）
    MATERIALS = {
        'sic': MaterialParams(
            A=6.7, B=1.5e4, C=200, m_eff=0.5, eps_inf=6.5,
            sellmeier_terms=[
                (0.20075, -12.07224),
                (5.54861, 0.02641),
                (35.65066, 1268.24708),
            ]
        ),
        'si': MaterialParams(A=11.4, B=0.0, C=0, m_eff=0.26, eps_inf=11.7)
    }

    def __init__(self, material='sic'):
        """
        初始化拟合器
        
        Parameters:
        -----------
        material : str
            材料类型, 'sic' 或 'si'
        """
        if material not in self.MATERIALS:
            raise ValueError(f"不支持的材料类型: {material}")
        self.material_params = self.MATERIALS[material]
        self.material = material
        self.wavelet_params = {
            'wavelet': 'db4',
            'mode': 'symmetric',
            'threshold_mode': 'soft'
        }

    def calculate_complex_permittivity(self, wavenumber: np.ndarray, 
                                     carrier_density: float, 
                                     scattering_rate: float) -> np.ndarray:
        """
        计算Drude-Sellmeier复介电函数
        """
        eps_bound = None
        if self.material_params.sellmeier_terms:
            # topic.md 中的 Sellmeier 形式使用波长 (µm)。代码中输入为波数 (cm^-1).
            # 需要将波数转换为波长 (µm): lambda_um = 1e4 / wavenumber (cm^-1) -> µm
            lambda_um = 1e4 / wavenumber
            # Sellmeier: n^2 - 1 = sum( B_j * lambda^2 / (lambda^2 - C_j) )
            # Here sellmeier_terms contains (B_j, C_j) with C_j in µm^2
            s = np.zeros_like(lambda_um, dtype=float)
            for Bj, Cj in self.material_params.sellmeier_terms:
                denom_s = lambda_um**2 - Cj
                denom_s = np.where(np.abs(denom_s) < 1e-12, 1e-12, denom_s)
                s += Bj * (lambda_um**2) / denom_s
            n2 = 1.0 + s
            eps_bound = n2.astype(complex)
        else:
            # 回退到原始简单共振项表示，输入均以波数 (cm^-1) 为单位
            A, B, C = self.material_params.A, self.material_params.B, self.material_params.C
            denom = wavenumber**2 - C**2
            denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
            eps_bound = (A + B / denom).astype(complex)
        
        # 支持 scalar 或 array 的 carrier_density（按元素计算 Drude 项）。
        omega = 2.0 * np.pi * self.C_LIGHT * wavenumber
        # 将载流子浓度规范化为 numpy 数组
        N_arr = np.asarray(carrier_density)
        m_star = self.material_params.m_eff * self.M_ELECTRON

        # 如果 carrier_density 为标量
        if N_arr.ndim == 0:
            if N_arr <= 0:
                return eps_bound
            N_m3 = float(N_arr) * 1e6
            omega_p_squared = (N_m3 * self.E_CHARGE**2) / (self.EPS_0 * m_star)
            denom_drude = omega**2 + 1j * scattering_rate * omega
            epsilon_drude = -omega_p_squared / denom_drude
            return (eps_bound + epsilon_drude).astype(complex)

        # 否则按元素计算（期望 carrier_density 与 wavenumber 形状可广播或相同）
        N_m3 = N_arr * 1e6
        # 尝试广播 N_m3 到与 omega 相同的形状
        try:
            N_m3_b = np.broadcast_to(N_m3, omega.shape)
        except Exception:
            # 若广播失败，尝试展平再匹配长度
            N_m3_b = np.asarray(N_m3).ravel()
            if N_m3_b.size != omega.ravel().size:
                raise ValueError('carrier_density 与 wavenumber 的长度无法匹配进行向量化计算。')
            N_m3_b = N_m3_b.reshape(omega.shape)

        omega_p_squared = (N_m3_b * self.E_CHARGE**2) / (self.EPS_0 * m_star)
        denom_drude = omega**2 + 1j * scattering_rate * omega
        epsilon_drude = -omega_p_squared / denom_drude

        # 对于非正的载流子浓度保持不加 Drude 项
        mask_nonpositive = (np.asarray(carrier_density) <= 0)
        if np.any(mask_nonpositive):
            # 广播 mask 到形状
            try:
                mask_b = np.broadcast_to(mask_nonpositive, omega.shape)
            except Exception:
                mask_b = np.broadcast_to(np.asarray(mask_nonpositive).ravel(), omega.shape)
            epsilon_drude = np.where(mask_b, 0.0 + 0.0j, epsilon_drude)

        return (eps_bound + epsilon_drude).astype(complex)

    def calculate_refractive_index(self, wavenumber: np.ndarray,
                                 carrier_density: float,
                                 scattering_rate: float) -> np.ndarray:
        """
        计算复折射率 ñ = n + ik
        """
        epsilon = self.calculate_complex_permittivity(wavenumber, carrier_density, scattering_rate)
        return np.sqrt(epsilon)

    def wavelet_denoise(self, reflectance: np.ndarray, 
                       wavelet: Optional[str] = None,
                       sigma: Optional[float] = None) -> np.ndarray:
        """
        小波阈值去噪处理
        """
        if not PYWT_AVAILABLE:
            warnings.warn("PyWavelets库不可用，跳过小波去噪")
            return reflectance.copy()
        try:
            wavelet_type = wavelet or self.wavelet_params['wavelet']
            mode = self.wavelet_params['mode']
            threshold_mode = self.wavelet_params['threshold_mode']
            coeffs = pywt.wavedec(reflectance, wavelet_type, mode=mode)
            if sigma is None:
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(reflectance)))
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode=threshold_mode) 
                                for detail in coeffs[1:]]
            denoised = pywt.waverec(coeffs_thresh, wavelet_type, mode=mode)
            if len(denoised) != len(reflectance):
                denoised = denoised[:len(reflectance)]
            return denoised
        except Exception as e:
            warnings.warn(f"小波去噪失败: {e}，返回原始数据")
            return reflectance.copy()

    def find_extrema(self, wavenumber: np.ndarray, reflectance: np.ndarray, 
                     prominence: float = 0.1, distance: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        从光谱数据中寻找极大值和极小值点
        """
        peak_indices, _ = find_peaks(reflectance, prominence=prominence, distance=distance)
        valley_indices, _ = find_peaks(-reflectance, prominence=prominence, distance=distance)
        peaks = wavenumber[peak_indices]
        valleys = wavenumber[valley_indices]
        logger.info("找到 %d 个极大值点, %d 个极小值点。", len(peaks), len(valleys))
        return peaks, valleys

    def calculate_interference_order(self, params: Dict, wavenumber: np.ndarray, 
                                     incident_angle: float) -> np.ndarray:
        """
        计算理论干涉级数 m
        """
        d = params['d']
        n_epi = self.calculate_refractive_index(
            wavenumber, params['N_epi'], params['gamma_epi']
        )
        theta0_rad = np.radians(incident_angle)
        sqrt_term = np.sqrt(
            np.maximum(0, n_epi.real**2 - np.sin(theta0_rad)**2)
        )
        d_cm = d * 1e-4
        m_theory = 2 * d_cm * wavenumber * sqrt_term
        return m_theory

    def objective_function(self, x: np.ndarray, param_names: List[str],
                           peaks: np.ndarray, valleys: np.ndarray, 
                           incident_angle: float) -> float:
        """
        目标函数：衡量干涉级数偏离整数/半整数的程度
        """
        params = dict(zip(param_names, x))
        cost = 0.0
        if len(peaks) > 0:
            m_peaks = self.calculate_interference_order(params, peaks, incident_angle)
            cost += np.sum((m_peaks - np.round(m_peaks))**2)
        if len(valleys) > 0:
            m_valleys = self.calculate_interference_order(params, valleys, incident_angle)
            cost += np.sum((m_valleys - (np.round(m_valleys - 0.5) + 0.5))**2)
        return cost

    def _residuals_vector(self, x: np.ndarray, param_names: List[str],
                          peaks: np.ndarray, valleys: np.ndarray,
                          incident_angle: float) -> np.ndarray:
        """
        构建残差向量：极大值对应 m - round(m)，极小值对应 m - (round(m-0.5)+0.5)
        用于数值雅可比和不确定度估计。
        """
        params = dict(zip(param_names, x))
        res_list = []
        if len(peaks) > 0:
            m_peaks = self.calculate_interference_order(params, peaks, incident_angle)
            res_peaks = m_peaks - np.round(m_peaks)
            res_list.append(res_peaks.astype(float))
        if len(valleys) > 0:
            m_valleys = self.calculate_interference_order(params, valleys, incident_angle)
            res_valleys = m_valleys - (np.round(m_valleys - 0.5) + 0.5)
            res_list.append(res_valleys.astype(float))
        if len(res_list) == 0:
            return np.zeros(0, dtype=float)
        return np.concatenate(res_list)

    def estimate_parameter_uncertainties(self, fitted_params: Dict[str, float],
                                        param_names: List[str],
                                        peaks: np.ndarray, valleys: np.ndarray,
                                        incident_angle: float,
                                        epsilon: float = 1e-6) -> Tuple[Dict[str, float], np.ndarray]:
        # 使用有限差分估计参数不确定度。
        x0 = np.array([fitted_params[name] for name in param_names], dtype=float)
        n_params = len(x0)
        r0 = self._residuals_vector(x0, param_names, peaks, valleys, incident_angle)
        n_resid = len(r0)
        if n_resid == 0 or n_params == 0:
            logger.warning("无法计算不确定度：残差或参数数目为0。")
            return {name: np.nan for name in param_names}, np.full((n_params, n_params), np.nan)

        J = np.zeros((n_resid, n_params), dtype=float)
        for i in range(n_params):
            xi = x0[i]
            delta = epsilon * max(1.0, abs(xi))
            x_pert = x0.copy()
            x_pert[i] = xi + delta
            r_pert = self._residuals_vector(x_pert, param_names, peaks, valleys, incident_angle)
            if len(r_pert) != n_resid:
                delta = epsilon * max(1.0, abs(xi)) * 0.1
                x_pert[i] = xi + delta
                r_pert = self._residuals_vector(x_pert, param_names, peaks, valleys, incident_angle)
                if len(r_pert) != n_resid:
                    raise RuntimeError("数值雅可比计算时残差长度不匹配。")
            J[:, i] = (r_pert - r0) / delta

        dof = max(1, n_resid - n_params)
        rss = float(np.sum(r0**2))
        sigma2 = rss / dof

        JTJ = J.T.dot(J)
        try:
            JTJ_inv = np.linalg.pinv(JTJ)
        except Exception as e:
            logger.warning("计算JTJ逆矩阵失败: %s", e)
            JTJ_inv = np.linalg.pinv(JTJ)

        covariance = JTJ_inv * sigma2
        uncertainties = np.sqrt(np.abs(np.diag(covariance)))
        param_uncertainties = {name: float(uncertainties[i]) for i, name in enumerate(param_names)}
        return param_uncertainties, covariance

    def fit_spectrum(self, wavenumber: np.ndarray, reflectance: np.ndarray,
                     incident_angle: float, initial_guess: Optional[Dict] = None,
                     bounds: Optional[Dict] = None,
                     use_denoise: bool = True) -> Dict:
        """
        执行拟合过程
        """
        reflectance_original = reflectance.copy()
        if use_denoise:
            logger.info("应用小波去噪预处理...")
            reflectance = self.wavelet_denoise(reflectance)

        peaks, valleys = self.find_extrema(wavenumber, reflectance)
        if len(peaks) + len(valleys) < 5:
            warnings.warn("找到的极值点过少，结果可能不可靠。")
            return {'success': False, 'message': '极值点太少'}
            
        if initial_guess is None:
            initial_guess = {
                'd': 10.0,
                'N_epi': 1e16,
                'gamma_epi': 1e13,
            }
        
        if bounds is None:
            bounds = {
                'd': (4.0, 20.0),
                'N_epi': (1e14, 1e18),
                'gamma_epi': (1e11, 1e15),
            }
            
        param_names = list(initial_guess.keys())
        bounds_list = [bounds[name] for name in param_names]
        
        logger.info("=== 极值点迭代优化 (入射角 %s°) ===", incident_angle)
        logger.info("参数边界: %s", bounds)
        
        start_time = time.time()

        de_history = {'x': [], 'cost': [], 'convergence': []}

        def _de_callback(xk, convergence=None):
            try:
                cost_xk = float(self.objective_function(xk, param_names, peaks, valleys, incident_angle))
            except Exception:
                cost_xk = np.nan
            de_history['x'].append(np.array(xk, dtype=float))
            de_history['cost'].append(cost_xk)
            de_history['convergence'].append(float(convergence) if convergence is not None else np.nan)
            return False

        result = differential_evolution(
            self.objective_function,
            bounds=bounds_list,
            args=(param_names, peaks, valleys, incident_angle),
            strategy='best1bin',
            maxiter=500,
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            callback=_de_callback
        )
        
        elapsed_time = time.time() - start_time
        
        fitted_params = dict(zip(param_names, result.x))
        final_cost = result.fun
        
        m_peaks = self.calculate_interference_order(fitted_params, peaks, incident_angle)
        m_valleys = self.calculate_interference_order(fitted_params, valleys, incident_angle)
        
        peak_residuals = m_peaks - np.round(m_peaks)
        valley_residuals = m_valleys - (np.round(m_valleys - 0.5) + 0.5)
        
        rmse_m = np.sqrt(np.mean(np.concatenate([peak_residuals**2, valley_residuals**2])))

        try:
            param_uncertainties, covariance = self.estimate_parameter_uncertainties(
                fitted_params, param_names, peaks, valleys, incident_angle
            )
        except Exception as e:
            logger.warning("参数不确定度估计失败: %s", e)
            param_uncertainties = {name: np.nan for name in param_names}
            covariance = np.full((len(param_names), len(param_names)), np.nan)

        fit_result = {
            'success': result.success,
            'params': fitted_params,
            'de_history': de_history,
            'param_uncertainties': param_uncertainties,
            'covariance': covariance,
            'cost': final_cost,
            'rmse_m': rmse_m,
            'elapsed_time': elapsed_time,
            'extrema': {'peaks': peaks, 'valleys': valleys},
            'm_theory': {'peaks': m_peaks, 'valleys': m_valleys},
            'denoised_reflectance': reflectance if use_denoise else None
        }
        
        logger.info("优化完成 (%.1fs)", elapsed_time)
        logger.info("成功: %s", result.success)
        logger.info("最终Cost: %.4f", final_cost)
        logger.info("干涉级数RMSE: %.4f", rmse_m)
        logger.info("拟合参数:")
        for name, value in fitted_params.items():
            if name == 'd':
                logger.info("  %s: %.3f μm ± %.3f", name, value, fit_result['param_uncertainties'].get(name, np.nan))
            elif 'N_' in name:
                logger.info("  %s: %.2e cm⁻³ ± %.2e", name, value, fit_result['param_uncertainties'].get(name, np.nan))
            elif 'gamma_' in name:
                logger.info("  %s: %.2e s⁻¹ ± %.2e", name, value, fit_result['param_uncertainties'].get(name, np.nan))

        return fit_result

    def plot_fit_result(self, wavenumber: np.ndarray, reflectance_data: np.ndarray,
                        fit_result: Dict, incident_angle: float, save_path: Optional[str] = None,
                        save_dir: Optional[str] = None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        figs = []

        fig = self._plot_spectrum_with_extrema(wavenumber, reflectance_data, fit_result, incident_angle)
        if fig is not None:
            figs.append(fig)
            if save_dir:
                self._save_current_fig(fig, os.path.join(save_dir, 'spectrum_extrema.png'))

        fig = self._plot_interference_orders(fit_result, incident_angle)
        if fig is not None:
            figs.append(fig)
            if save_dir:
                self._save_current_fig(fig, os.path.join(save_dir, 'interference_orders.png'))

        fig = self._plot_refractive_index(wavenumber, fit_result)
        if fig is not None:
            figs.append(fig)
            if save_dir:
                self._save_current_fig(fig, os.path.join(save_dir, 'refractive_index.png'))

        fig = self._plot_residuals_histogram(fit_result)
        if fig is not None:
            figs.append(fig)
            if save_dir:
                self._save_current_fig(fig, os.path.join(save_dir, 'residuals_hist.png'))

        fig = self._plot_residuals_vs_index(fit_result)
        if fig is not None:
            figs.append(fig)
            if save_dir:
                self._save_current_fig(fig, os.path.join(save_dir, 'residuals_vs_index.png'))

        fig = self._plot_optimization_history(fit_result)
        if fig is not None:
            figs.append(fig)
            if save_dir:
                self._save_current_fig(fig, os.path.join(save_dir, 'de_history.png'))

        if save_path:
            self._plot_combined_summary(wavenumber, reflectance_data, fit_result, incident_angle, save_path)

        try:
            if len(figs) > 0:
                plt.show()
        finally:
            for f in figs:
                try:
                    plt.close(f)
                except Exception:
                    pass

    def _save_current_fig(self, path: str):
        try:
            fig = plt.gcf()
            fig.savefig(path, dpi=plt.rcParams.get('savefig.dpi', 300), bbox_inches=plt.rcParams.get('savefig.bbox', 'tight'))
            logger.info('Saved figure to %s', path)
        except Exception as e:
            logger.warning('Failed to save figure %s: %s', path, e)
        finally:
            plt.close('all')

    def _save_current_fig(self, fig, path: str):
        try:
            fig.savefig(path, dpi=plt.rcParams.get('savefig.dpi', 300), bbox_inches=plt.rcParams.get('savefig.bbox', 'tight'))
            logger.info('Saved figure to %s', path)
        except Exception as e:
            logger.warning('Failed to save figure %s: %s', path, e)
        finally:
            try:
                plt.close(fig)
            except Exception:
                plt.close('all')

    def _apply_plot_style(self):
        """
        应用统一的绘图样式：配色、线宽、网格和中文字体等。
        在所有绘图函数开头调用以保证风格一致。
        """
        try:
            # 尝试设置全局字体为中文字体
            if CHINESE_FONT is not None:
                try:
                    fname = CHINESE_FONT.get_name()
                    if fname:
                        plt.rcParams['font.family'] = fname
                except Exception:
                    # FontProperties 可能由文件路径构造，跳过全局设置
                    pass

            # 基础样式
            plt.rcParams['figure.figsize'] = (12, 6)
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['xtick.labelsize'] = 11
            plt.rcParams['ytick.labelsize'] = 11
            plt.rcParams['legend.fontsize'] = 11
            plt.rcParams['lines.linewidth'] = 1.8
            plt.rcParams['lines.markersize'] = 6
            plt.rcParams['grid.linewidth'] = 0.8
            plt.rcParams['grid.alpha'] = 0.28
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['savefig.bbox'] = 'tight'
            # 颜色调色板
            self._palette = [
                '#2E86AB',  # 深蓝
                '#F6A01A',  # 明橙
                '#3CA55C',  # 绿色
                '#D7263D',  # 红
                '#6A4C93',  # 紫
                '#8C6239',  # 棕
                '#E76F8A',  # 粉
                '#6C757D',  # 中灰
                '#9AAE00',  # 黄绿
                '#17BECF',  # 青
            ]
            try:
                plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self._palette)
            except Exception:
                pass
        except Exception:
            pass

    def _plot_spectrum_with_extrema(self, wavenumber: np.ndarray, reflectance_data: np.ndarray,
                                    fit_result: Dict, incident_angle: float):
        self._apply_plot_style()
        fig, ax = plt.subplots(1, 1)
        palette = getattr(self, '_palette', None) or plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)
        color_raw = palette[0] if palette else '#1f77b4'
        color_denoised = palette[2] if palette and len(palette) > 2 else '#2ca02c'
        ax.plot(wavenumber, reflectance_data, color=color_raw, alpha=0.6, label='原始光谱')
        if fit_result.get('denoised_reflectance') is not None:
            ax.plot(wavenumber, fit_result['denoised_reflectance'], color=color_denoised, alpha=0.9, label='去噪后光谱')
        peaks = fit_result['extrema']['peaks']
        valleys = fit_result['extrema']['valleys']
        reflect_interp = np.interp(np.concatenate([peaks, valleys]), wavenumber, reflectance_data)
        marker_peak_color = palette[3] if palette and len(palette) > 3 else '#d62728'
        marker_valley_color = palette[7] if palette and len(palette) > 7 else '#7f7f7f'
        ax.scatter(peaks, reflect_interp[:len(peaks)], marker='o', color=marker_peak_color, edgecolor='white', label='极大值点', zorder=3)
        ax.scatter(valleys, reflect_interp[len(peaks):], marker='v', color=marker_valley_color, edgecolor='white', label='极小值点', zorder=3)
        ax.set_ylabel('反射率 (%)', fontproperties=CHINESE_FONT)
        ax.set_xlabel('波数 (cm$^{-1}$)', fontproperties=CHINESE_FONT)
        ax.set_title(f'光谱与极值点 - {self.material.upper()} (入射角 {incident_angle}°)', fontproperties=CHINESE_FONT)
        leg = ax.legend(prop=CHINESE_FONT, frameon=True, loc='upper right')
        leg.get_frame().set_alpha(0.95)
        ax.grid(True, linestyle='--', linewidth=0.6)
        plt.tight_layout()
        return fig

    def _plot_interference_orders(self, fit_result: Dict, incident_angle: float):
        peaks = fit_result['extrema']['peaks']
        valleys = fit_result['extrema']['valleys']
        m_peaks_theory = fit_result['m_theory']['peaks']
        m_valleys_theory = fit_result['m_theory']['valleys']
        m_peaks_int = np.round(m_peaks_theory)
        m_valleys_int = np.round(m_valleys_theory - 0.5) + 0.5
        self._apply_plot_style()
        fig, ax = plt.subplots(1, 1)
        palette = getattr(self, '_palette', None)
        p_peak = palette[3] if palette else '#d62728'
        p_valley = palette[7] if palette else '#7f7f7f'
        ax.scatter(peaks, m_peaks_theory, marker='o', color=p_peak, label='理论干涉级数（极大值）')
        ax.plot(peaks, m_peaks_int, linestyle='--', color=p_peak, alpha=0.8, label='最近整数')
        ax.scatter(valleys, m_valleys_theory, marker='v', color=p_valley, label='理论干涉级数（极小值）')
        ax.plot(valleys, m_valleys_int, linestyle=':', color=p_valley, alpha=0.8, label='最近半整数')
        ax.set_xlabel('波数 (cm$^{-1}$)', fontproperties=CHINESE_FONT)
        ax.set_ylabel('干涉级数 m', fontproperties=CHINESE_FONT)
        ax.set_title(f'干涉级数拟合结果（RMSE_m = {fit_result["rmse_m"]:.4f}）', fontproperties=CHINESE_FONT)
        leg = ax.legend(prop=CHINESE_FONT, loc='best', frameon=True)
        leg.get_frame().set_alpha(0.95)
        ax.grid(True, linestyle='--', linewidth=0.6)
        plt.tight_layout()
        return fig

    def _plot_refractive_index(self, wavenumber: np.ndarray, fit_result: Dict):
        params = fit_result['params']
        n_complex = self.calculate_refractive_index(wavenumber, params['N_epi'], params['gamma_epi'])
        n_real = n_complex.real
        n_imag = n_complex.imag
        self._apply_plot_style()
        fig, ax = plt.subplots(1, 1)
        palette = getattr(self, '_palette', None)
        color_n = palette[0] if palette else '#1f77b4'
        color_k = palette[3] if palette else '#d62728'
        ax.plot(wavenumber, n_real, color=color_n, label='n（实部）')
        ax.plot(wavenumber, n_imag, color=color_k, label='k（虚部）')
        ax.set_xlabel('波数 (cm$^{-1}$)', fontproperties=CHINESE_FONT)
        ax.set_ylabel('折射率', fontproperties=CHINESE_FONT)
        ax.set_title('拟合折射率 n 与 k', fontproperties=CHINESE_FONT)
        leg = ax.legend(prop=CHINESE_FONT, loc='best', frameon=True)
        leg.get_frame().set_alpha(0.95)
        ax.grid(True, linestyle='--', linewidth=0.6)
        plt.tight_layout()
        return fig

    def _plot_residuals_histogram(self, fit_result: Dict):
        peaks = fit_result['extrema']['peaks']
        valleys = fit_result['extrema']['valleys']
        m_peaks = fit_result['m_theory']['peaks']
        m_valleys = fit_result['m_theory']['valleys']
        peak_residuals = m_peaks - np.round(m_peaks)
        valley_residuals = m_valleys - (np.round(m_valleys - 0.5) + 0.5)
        residuals = np.concatenate([peak_residuals, valley_residuals])
        self._apply_plot_style()
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(residuals, bins=20, color='#7f7f7f', edgecolor='white')
        ax.set_xlabel('干涉级数残差', fontproperties=CHINESE_FONT)
        ax.set_ylabel('计数', fontproperties=CHINESE_FONT)
        ax.set_title('干涉级数残差直方图', fontproperties=CHINESE_FONT)
        ax.grid(True, linestyle='--', linewidth=0.6)
        plt.tight_layout()
        return fig

    def _plot_residuals_vs_index(self, fit_result: Dict):
        """
        绘制残差随样本索引的分布（便于检查系统性趋势）
        """
        peaks = fit_result['extrema']['peaks']
        valleys = fit_result['extrema']['valleys']
        m_peaks = fit_result['m_theory']['peaks']
        m_valleys = fit_result['m_theory']['valleys']
        res_peaks = m_peaks - np.round(m_peaks)
        res_valleys = m_valleys - (np.round(m_valleys - 0.5) + 0.5)
        self._apply_plot_style()
        fig, ax = plt.subplots(1, 1)
        palette = getattr(self, '_palette', None)
        p_peak = palette[3] if palette else '#d62728'
        p_valley = palette[7] if palette else '#7f7f7f'
        if len(res_peaks) > 0:
            ax.scatter(np.arange(len(res_peaks)), res_peaks, marker='o', color=p_peak, label='极大值残差')
        if len(res_valleys) > 0:
            ax.scatter(np.arange(len(res_peaks), len(res_peaks) + len(res_valleys)), res_valleys, marker='v', color=p_valley, label='极小值残差')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('样本索引', fontproperties=CHINESE_FONT)
        ax.set_ylabel('干涉级数残差', fontproperties=CHINESE_FONT)
        ax.set_title('残差随样本索引分布', fontproperties=CHINESE_FONT)
        leg = ax.legend(prop=CHINESE_FONT, frameon=True)
        leg.get_frame().set_alpha(0.95)
        ax.grid(True, linestyle='--', linewidth=0.6)
        plt.tight_layout()
        return fig

    def _plot_optimization_history(self, fit_result: Dict):
        """
        可视化差分进化的优化历史（cost随迭代变化，以及参数轨迹的投影）。
        要求 fit_result 包含 'de_history'。
        """
        de_hist = fit_result.get('de_history')
        if not de_hist or len(de_hist.get('cost', [])) == 0:
            logger.info('没有差分进化历史可用以绘图。')
            return

        costs = np.array(de_hist['cost'], dtype=float)
        xs = np.array(de_hist.get('x', []))

        if xs.size != 0 and xs.ndim == 2 and xs.shape[1] >= 2:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
            ax0.plot(np.arange(len(costs)), costs, '-o', markersize=4)
            ax0.set_yscale('log')
            ax0.set_xlabel('回调次数', fontproperties=CHINESE_FONT)
            ax0.set_ylabel('目标函数 cost (log scale)', fontproperties=CHINESE_FONT)
            ax0.set_title('差分进化优化历史', fontproperties=CHINESE_FONT)
            ax0.grid(True, alpha=0.2)

            ax1.plot(xs[:, 0], xs[:, 1], '-o', markersize=3)
            ax1.set_xlabel('param 0', fontproperties=CHINESE_FONT)
            ax1.set_ylabel('param 1', fontproperties=CHINESE_FONT)
            ax1.set_title('参数轨迹（前两参数）', fontproperties=CHINESE_FONT)
            ax1.grid(True, alpha=0.2)
            plt.tight_layout()
            return fig
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(np.arange(len(costs)), costs, '-o', markersize=4)
            ax.set_yscale('log')
            ax.set_xlabel('回调次数', fontproperties=CHINESE_FONT)
            ax.set_ylabel('目标函数 cost (log scale)', fontproperties=CHINESE_FONT)
            ax.set_title('差分进化优化历史', fontproperties=CHINESE_FONT)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            return fig

    def _plot_combined_summary(self, wavenumber: np.ndarray, reflectance_data: np.ndarray,
                               fit_result: Dict, incident_angle: float, save_path: str):
        """Create a combined summary figure with small panels and save to path."""
        params = fit_result['params']
        peaks = fit_result['extrema']['peaks']
        valleys = fit_result['extrema']['valleys']
        m_peaks_theory = fit_result['m_theory']['peaks']
        m_valleys_theory = fit_result['m_theory']['valleys']
        n_complex = self.calculate_refractive_index(wavenumber, params['N_epi'], params['gamma_epi'])
        n_real = n_complex.real
        n_imag = n_complex.imag

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax0 = axes[0, 0]
        ax0.plot(wavenumber, reflectance_data, 'b-', alpha=0.6)
        if fit_result.get('denoised_reflectance') is not None:
            ax0.plot(wavenumber, fit_result['denoised_reflectance'], 'g-', alpha=0.8)
        ax0.plot(peaks, np.interp(peaks, wavenumber, reflectance_data), 'ro')
        ax0.plot(valleys, np.interp(valleys, wavenumber, reflectance_data), 'kv')
        ax0.set_title('光谱与极值点', fontproperties=CHINESE_FONT)

        ax1 = axes[0, 1]
        ax1.plot(peaks, m_peaks_theory, 'ro')
        ax1.plot(peaks, np.round(m_peaks_theory), 'r--')
        ax1.plot(valleys, m_valleys_theory, 'kv')
        ax1.plot(valleys, np.round(m_valleys_theory - 0.5) + 0.5, 'k:')
        ax1.set_title('干涉级数', fontproperties=CHINESE_FONT)

        ax2 = axes[1, 0]
        ax2.plot(wavenumber, n_real, label='n')
        ax2.plot(wavenumber, n_imag, label='k')
        ax2.set_title('折射率 n,k', fontproperties=CHINESE_FONT)
        ax2.legend(prop=CHINESE_FONT)

        ax3 = axes[1, 1]
        peak_residuals = m_peaks_theory - np.round(m_peaks_theory)
        valley_residuals = m_valleys_theory - (np.round(m_valleys_theory - 0.5) + 0.5)
        residuals = np.concatenate([peak_residuals, valley_residuals])
        ax3.hist(residuals, bins=20, color='gray', edgecolor='black')
        ax3.set_title('残差直方图', fontproperties=CHINESE_FONT)

        plt.suptitle(f'拟合摘要 - {self.material.upper()} (θ={incident_angle}°)  RMSE_m={fit_result["rmse_m"]:.4f}', fontproperties=CHINESE_FONT)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            fig.savefig(save_path, dpi=200)
            logger.info('已保存摘要图到 %s', save_path)
        except Exception as e:
            logger.warning('保存摘要图失败: %s', e)
        plt.show()

    def _plot_refractive_index_surface(self, wavenumber: np.ndarray, fit_result: Dict,
                                       wavenumber_grid_points: int = 80, carrier_grid_points: int = 80,
                                       save_path: Optional[str] = None):
        """
        绘制三维折射率（实部/虚部）随波数与载流子浓度变化的表面图。
        提供可视化色散和吸收对薄膜光学性质的影响。
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception:
            logger.warning('mpl_toolkits.mplot3d 不可用，跳过三维可视化。')
            return

        self._apply_plot_style()
        wn_min, wn_max = float(np.min(wavenumber)), float(np.max(wavenumber))
        wn_grid = np.linspace(wn_min, wn_max, wavenumber_grid_points)
        N_fit = fit_result['params'].get('N_epi', 1e16)
        N_min, N_max = max(1e14, N_fit / 100.0), min(1e18, N_fit * 100.0)
        N_grid = np.logspace(np.log10(N_min), np.log10(N_max), carrier_grid_points)
        WN, NGRID = np.meshgrid(wn_grid, N_grid)
        gamma_fit = fit_result['params'].get('gamma_epi', 1e13)
        rn_flat = self.calculate_refractive_index(WN.ravel(), NGRID.ravel(), gamma_fit).real
        rn = rn_flat.reshape(NGRID.shape)

        # plot surface (实部)
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(WN, np.log10(NGRID), rn, cmap='viridis', edgecolor='none', alpha=0.9)
        
        # 为3D图设置适中的中文字体
        chinese_font_small = None
        if CHINESE_FONT is not None:
            try:
                chinese_font_small = FontProperties(fname=CHINESE_FONT.get_name() if hasattr(CHINESE_FONT, 'get_name') else None, size=12)
                if chinese_font_small.get_name() is None and hasattr(CHINESE_FONT, '_fname'):
                    chinese_font_small = FontProperties(fname=CHINESE_FONT._fname, size=12)
            except Exception:
                chinese_font_small = FontProperties(size=12)
        else:
            chinese_font_small = FontProperties(size=12)
            
        ax.set_xlabel('波数 (cm$^{-1}$)', fontproperties=chinese_font_small)
        ax.set_ylabel('载流子浓度 N (cm$^{-3}$) — 以 $\log_{10}$ 显示', fontproperties=chinese_font_small)
        ax.set_zlabel('n (实部)', fontproperties=chinese_font_small)
        ax.set_title('折射率实部随波数与载流子浓度的表面', fontproperties=chinese_font_small, pad=20)
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=30)

        try:
            common_powers = np.array([14, 15, 16, 17, 18])
            common_vals = 10.0 ** common_powers
            within = (common_vals >= N_min) & (common_vals <= N_max)
            if np.any(within):
                ytick_vals = common_vals[within]
                ytick_positions = np.log10(ytick_vals)
                ax.set_yticks(ytick_positions)
                ytick_labels = [f'$10^{{{int(p)}}}$' for p in common_powers[within]]
                ax.set_yticklabels(ytick_labels, fontproperties=chinese_font_small)
        except Exception:
            pass
        plt.tight_layout()
        if save_path:
            try:
                fig.savefig(save_path, dpi=200)
                logger.info('已保存三维表面图到 %s', save_path)
            except Exception as e:
                logger.warning('保存三维表面图失败: %s', e)
        plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='极值点迭代优化拟合器')
    parser.add_argument('file', help='CSV文件路径')
    parser.add_argument('--angle', type=float, default=10.0, help='入射角度 (默认10°)')
    parser.add_argument('--material', choices=['sic', 'si'], default='sic', help='材料类型')
    parser.add_argument('--plot', action='store_true', help='显示拟合结果图（包含高级图）')
    parser.add_argument('--out-dir', type=str, default=None, help='保存图像的目录，默认为输入文件同目录下的 figures 文件夹')
    parser.add_argument('--no-advanced', action='store_true', help='在 --plot 时不显示高级图（示意图、3D）')
    parser.add_argument('--range', nargs=2, type=float, help='波数范围 (最小值 最大值)')
    parser.add_argument('--no-denoise', action='store_true', help='禁用小波去噪预处理')
    
    args = parser.parse_args()
    
    print(f"加载光谱数据: {args.file}")
    wavenumber, reflectance = load_spectrum_data(args.file)
    
    if args.range:
        mask = (wavenumber >= args.range[0]) & (wavenumber <= args.range[1])
        wavenumber = wavenumber[mask]
        reflectance = reflectance[mask]
        print(f"应用波数范围过滤: {args.range[0]}-{args.range[1]} cm$^{-1}$")
        
    fitter = IterativeExtremumFitter(material=args.material)
    
    use_denoise = not args.no_denoise
    result = fitter.fit_spectrum(
        wavenumber, reflectance, args.angle, 
        use_denoise=use_denoise
    )
    
    if not result['success']:
        print(f"警告: 拟合失败或未收敛 ({result.get('message', '')})")
    
    if args.plot and result['success']:
        out_dir = args.out_dir if args.out_dir else os.path.join(os.path.dirname(args.file) or '.', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        combined_path = os.path.join(out_dir, 'combined_summary.png')
        fitter.plot_fit_result(wavenumber, reflectance, result, args.angle, save_path=combined_path, save_dir=out_dir)

        if not args.no_advanced:
            try:
                fitter._plot_refractive_index_surface(wavenumber, result, save_path=os.path.join(out_dir, 'n_surface.png'))
            except Exception as e:
                logger.warning('绘制/保存三维折射率表面失败: %s', e)


if __name__ == '__main__':
    main()