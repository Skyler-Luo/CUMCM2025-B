"""
碳化硅外延层厚度测量数据可视化分析
红外干涉法光谱数据分析

数据说明：
- 附件1,2：碳化硅晶圆片，入射角10°和15°
- 附件3,4：硅晶圆片，入射角10°和15°
- 数据格式：波数(cm^-1), 反射率(%)
"""

from cycler import cycler
from scipy import signal  # 信号处理库
from scipy.ndimage import uniform_filter1d
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pywt  # 小波变换库
PYWT_AVAILABLE = True

class SpectralAnalyzer:
    """
    光谱数据分析器，用于处理和可视化碳化硅外延层厚度测量的红外干涉光谱数据。
    
    主要功能包括：
    - 数据加载与预处理
    - 数据质量评估
    - 原始光谱可视化
    - 基于小波变换的信号去噪
    - 去噪效果的定量与定性分析
    - 分析结果的图形化展示
    """
    
    def __init__(self):
        """初始化分析器，设置数据和绘图样式"""
        self.data = {}
        self.materials = {
            '附件1': ('碳化硅', '10°'),
            '附件2': ('碳化硅', '15°'),
            '附件3': ('硅', '10°'),
            '附件4': ('硅', '15°')
        }
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        """设置全局绘图样式和中文字体"""
        import matplotlib
        
        sns.set_style('whitegrid')
        sns.set_context('talk', font_scale=0.9)
        matplotlib.rcParams['axes.prop_cycle'] = cycler('color', sns.color_palette('deep', 8))
        matplotlib.rcParams['figure.figsize'] = (12, 8)
        matplotlib.rcParams['axes.titleweight'] = 'bold'
        matplotlib.rcParams['axes.titlesize'] = 13
        matplotlib.rcParams['axes.labelsize'] = 11
        matplotlib.rcParams['legend.frameon'] = False
        matplotlib.rcParams['savefig.dpi'] = 300
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
        matplotlib.rcParams['axes.unicode_minus'] = True

        # 设置中文字体
        self.setup_chinese_fonts()
    
    def setup_chinese_fonts(self):
        """设置中文字体"""
        import platform
        import os
        from matplotlib.font_manager import FontProperties, fontManager
        system = platform.system()
        
        try:
            import matplotlib
            matplotlib.font_manager._rebuild()
        except:
            pass
        
        # 根据系统选择字体
        if system == "Windows":
            font_candidates = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'Microsoft JhengHei']
        elif system == "Darwin":  # macOS
            font_candidates = ['PingFang SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
        else:  # Linux
            font_candidates = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        
        font_set = False
        for font in font_candidates:
            try:
                # 设置全局字体
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                # 确保负号能正常显示
                plt.rcParams['axes.unicode_minus'] = True
                
                # 简单测试
                test_fig = plt.figure(figsize=(1, 1))
                test_ax = test_fig.add_subplot(111)
                test_ax.text(0.5, 0.5, '中文测试', fontsize=10)
                plt.close(test_fig)
                
                print(f"成功设置中文字体: {font}")
                font_set = True
                break
            except Exception as e:
                continue
        
        if not font_set:
            print("无法设置中文字体，将使用备用方案")
            # 直接设置字体参数
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
            plt.rcParams['axes.unicode_minus'] = False

    def save_figure(self, fig, filename):
        """统一保存图片到 data/figures 目录并关闭图像以释放内存"""
        out_dir = 'data/figures'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        path = os.path.join(out_dir, filename)
        try:
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"已保存图片: {path}")
        except Exception as e:
            print(f"保存图片失败: {e}")
        plt.close(fig)
    
    def load_data(self):
        """加载所有数据文件"""
        print("正在加载数据文件...")
        
        for filename in ['附件1.csv', '附件2.csv', '附件3.csv', '附件4.csv']:
            try:
                filepath = f'data/{filename}'
                df = pd.read_csv(filepath)
                
                # 标准化列名
                df.columns = ['wavenumber', 'reflectance']
                
                # 数据清洗：移除无效值
                df = df[df['reflectance'] > 0]  # 移除反射率为0的点
                df = df.dropna()  # 移除缺失值
                
                key = filename.replace('.csv', '')
                self.data[key] = df
                
                print(f"✓ {filename}: {len(df)} 个数据点")
                
            except Exception as e:
                print(f"✗ 加载 {filename} 失败: {e}")
    
    def basic_statistics(self):
        """基本统计信息"""
        print("数据基本统计信息")
        
        for key, df in self.data.items():
            material, angle = self.materials[key]
            print(f"\n{key} ({material}, 入射角{angle}):")
            print(f"  波数范围: {df['wavenumber'].min():.1f} - {df['wavenumber'].max():.1f} cm^-1")
            print(f"  反射率范围: {df['reflectance'].min():.2f} - {df['reflectance'].max():.2f} %")
            print(f"  平均反射率: {df['reflectance'].mean():.2f} %")
            print(f"  反射率标准差: {df['reflectance'].std():.2f} %")
    
    def plot_raw_spectra(self):
        """绘制原始光谱数据"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
        axes = [ax1, ax2, ax3, ax4]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (key, df) in enumerate(self.data.items()):
            material, angle = self.materials[key]
            ax = axes[i]
            
            ax.plot(df['wavenumber'], df['reflectance'], 
                   color=colors[i], linewidth=1, alpha=0.8)
            ax.set_title(f'{key}: {material} (入射角{angle})', fontsize=12, fontweight='bold')
            ax.set_xlabel('波数 (cm$^{-1}$)')
            ax.set_ylabel('反射率 (%)')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_ref = df['reflectance'].mean()
            ax.axhline(y=mean_ref, color='red', linestyle='--', alpha=0.7, 
                      label=f'平均值: {mean_ref:.2f}%')
            ax.legend()
        
        # 保存并展示
        self.save_figure(fig, 'raw_spectra_analysis.png')
        try:
            fig.show()
        except:
            pass
    
    
    def wavelet_threshold_filter(self, y, wavelet='db4', sigma=None):
        """
        小波阈值滤波去噪
        
        参数:
        y: 输入信号
        wavelet: 小波类型
        sigma: 噪声标准差 (如果为None则自动估计)
        
        返回:
        filtered_y: 滤波后的信号
        """
        if not PYWT_AVAILABLE:
            print("PyWavelets库不可用，返回原始信号")
            return y.copy()
        
        try:
            # 小波分解
            coeffs = pywt.wavedec(y, wavelet, mode='symmetric')
            
            # 估计噪声标准差
            if sigma is None:
                # 使用最高频细节系数估计噪声
                # 0.6745是高斯分布中位数绝对偏差(MAD)与标准差的关系常数 (sigma = MAD / 0.6745)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # 计算阈值 (软阈值)
            threshold = sigma * np.sqrt(2 * np.log(len(y)))
            
            # 对细节系数进行阈值处理
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode='soft') 
                                for detail in coeffs[1:]]
            
            # 小波重构
            filtered_y = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')
            
            # 确保输出长度与输入相同
            if len(filtered_y) != len(y):
                filtered_y = filtered_y[:len(y)]
            
            return filtered_y
            
        except Exception as e:
            print(f"小波滤波失败: {e}")
            return y.copy()
    
    def estimate_noise_level(self, y, wavelet='db4'):
        """
        使用小波系数的绝对中位差(MAD)稳健地估计噪声水平
        
        参数:
        y: 输入信号
        wavelet: 小波类型
        
        返回:
        sigma: 估计的噪声标准差
        """
        if not PYWT_AVAILABLE:
            return np.std(np.diff(y))
        
        try:
            coeffs = pywt.wavedec(y, wavelet, mode='symmetric')
            # 根据高斯噪声的MAD公式估计sigma
            # 0.6745是高斯分布中位数绝对偏差(MAD)与标准差的关系常数
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            return sigma
        except Exception as e:
            print(f"噪声估计失败: {e}")
            return np.std(np.diff(y))
    
    def calculate_snr(self, original, filtered):
        """计算信噪比改善"""
        try:
            # 使用稳健的小波方法估计噪声
            noise_original = self.estimate_noise_level(original)
            noise_filtered = self.estimate_noise_level(filtered)
            
            # 信号功率
            signal_power = np.var(filtered)
            
            # SNR计算 (添加epsilon防止除以零)
            snr_original = 10 * np.log10(signal_power / (noise_original**2 + np.finfo(float).eps))
            snr_filtered = 10 * np.log10(signal_power / (noise_filtered**2 + np.finfo(float).eps))
            
            improvement = snr_filtered - snr_original
            
            return snr_original, snr_filtered, improvement
            
        except Exception as e:
            print(f"SNR计算失败: {e}")
            return 0, 0, 0
    
    def detailed_denoising_analysis(self):
        """详细的小波滤波性能分析, 并保存去噪后的数据"""
        print("\n正在进行详细小波滤波性能分析...")
        
        # 创建用于保存去噪数据的目录
        output_dir = 'data/denoised'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")

        # 分析所有数据文件
        results = {}
        
        for key, df in self.data.items():
            material, angle = self.materials[key]
            x = df['wavenumber'].values
            y = df['reflectance'].values
            
            # 应用小波滤波
            y_wavelet = self.wavelet_threshold_filter(y)
            
            # 保存去噪后的数据
            denoised_df = pd.DataFrame({'wavenumber': x, 'reflectance': y_wavelet})
            output_path = os.path.join(output_dir, f'denoised_{key}.csv')
            denoised_df.to_csv(output_path, index=False)
            print(f"  ✓ {key}: 去噪数据已保存到 {output_path}")

            # 计算性能指标
            snr_orig, snr_wt, snr_improve_wt = self.calculate_snr(y, y_wavelet)
            
            # 计算均方根误差 (相对于高度平滑版本)
            # 使用移动平均作为参考基准
            window_size = max(51, len(y)//20)
            y_smooth = uniform_filter1d(y, size=window_size)
            rmse_orig = np.sqrt(np.mean((y - y_smooth)**2))
            rmse_wt = np.sqrt(np.mean((y_wavelet - y_smooth)**2))
            
            # 计算保边缘能力 (梯度保持)
            grad_orig = np.gradient(y)
            grad_wt = np.gradient(y_wavelet)
            
            edge_preserve_wt = np.corrcoef(grad_orig, grad_wt)[0,1]
            
            # 计算噪声减少 (添加epsilon防止除以零)
            noise_orig = self.estimate_noise_level(y)
            noise_wt = self.estimate_noise_level(y_wavelet)
            noise_reduction = (1 - noise_wt / (noise_orig + np.finfo(float).eps)) * 100
            
            results[key] = {
                'material': material,
                'angle': angle,
                'snr_improve_wt': snr_improve_wt,
                'rmse_reduction_wt': (rmse_orig - rmse_wt) / rmse_orig * 100,
                'edge_preserve_wt': edge_preserve_wt,
                'noise_reduction': noise_reduction
            }
        
        # 打印分析结果
        print("\n" + "="*80)
        print("小波滤波详细性能分析结果")
        print("="*80)
        
        for key, result in results.items():
            print(f"\n{key} ({result['material']}, 入射角{result['angle']}):")
            print(f"  小波阈值滤波性能:")
            print(f"    SNR改善: {result['snr_improve_wt']:.2f} dB")
            print(f"    RMSE减少: {result['rmse_reduction_wt']:.2f}%")
            print(f"    边缘保持: {result['edge_preserve_wt']:.4f}")
            print(f"    噪声减少: {result['noise_reduction']:.1f}%")
            
            # 性能评估
            overall_score = (result['snr_improve_wt'] + result['rmse_reduction_wt']/10 + 
                           result['edge_preserve_wt']*10 + result['noise_reduction']/10)
            
            if overall_score > 15:
                quality = "优秀"
            elif overall_score > 10:
                quality = "良好"
            elif overall_score > 5:
                quality = "一般"
            else:
                quality = "较差"
            
            print(f"    综合评价: {quality} (得分: {overall_score:.2f})")
        
        # 总体统计
        print(f"\n总体统计 (平均值):")
        avg_snr_wt = np.mean([r['snr_improve_wt'] for r in results.values()])
        avg_edge_wt = np.mean([r['edge_preserve_wt'] for r in results.values()])
        avg_noise_reduction = np.mean([r['noise_reduction'] for r in results.values()])
        
        print(f"  小波阈值滤波:")
        print(f"    平均SNR改善: {avg_snr_wt:.2f} dB")
        print(f"    平均边缘保持: {avg_edge_wt:.4f}")
        print(f"    平均噪声减少: {avg_noise_reduction:.1f}%")
        
        return results

    def plot_zoomed_denoising_comparison(self):
        """
        绘制合并后的去噪效果分析图。
        包含三个局部放大区域和一个带SNR改善指标的全光谱对比图。
        """
        print("\n正在生成合并去噪分析图...")
        
        # 选择附件1进行分析
        sample_key = '附件1'
        df = self.data[sample_key]
        x = df['wavenumber'].values
        y = df['reflectance'].values
        
        # 应用小波滤波
        y_wavelet = self.wavelet_threshold_filter(y)
        
        # 选择三个有代表性的区域进行放大
        # 区域1: 低频区域 (800-1200 cm^-1)
        # 区域2: 中频区域 (1500-2000 cm^-1)
        # 区域3: 高频区域 (2500-3000 cm^-1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        fig.suptitle('小波滤波去噪效果综合分析', fontsize=16, fontweight='bold')
        
        # 区域1: 低频区域
        mask1 = (x >= 800) & (x <= 1200)
        x1 = x[mask1]
        y1 = y[mask1]
        y1_wt = y_wavelet[mask1]
        
        ax = axes[0, 0]
        ax.plot(x1, y1, 'b-', linewidth=1, alpha=0.6, label='原始数据')
        ax.plot(x1, y1_wt, 'g-', linewidth=2, label='小波滤波')
        ax.set_title('局部放大 (低频区域: 800-1200 cm$^{-1}$)', fontweight='bold')
        ax.set_xlabel('波数 (cm$^{-1}$)')
        ax.set_ylabel('反射率 (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 区域2: 中频区域
        mask2 = (x >= 1500) & (x <= 2000)
        x2 = x[mask2]
        y2 = y[mask2]
        y2_wt = y_wavelet[mask2]
        
        ax = axes[0, 1]
        ax.plot(x2, y2, 'b-', linewidth=1, alpha=0.6, label='原始数据')
        ax.plot(x2, y2_wt, 'g-', linewidth=2, label='小波滤波')
        ax.set_title('局部放大 (中频区域: 1500-2000 cm$^{-1}$)', fontweight='bold')
        ax.set_xlabel('波数 (cm$^{-1}$)')
        ax.set_ylabel('反射率 (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 区域3: 高频区域
        mask3 = (x >= 2500) & (x <= 3000)
        x3 = x[mask3]
        y3 = y[mask3]
        y3_wt = y_wavelet[mask3]
        
        ax = axes[1, 0]
        ax.plot(x3, y3, 'b-', linewidth=1, alpha=0.6, label='原始数据')
        ax.plot(x3, y3_wt, 'g-', linewidth=2, label='小波滤波')
        ax.set_title('局部放大 (高频区域: 2500-3000 cm$^{-1}$)', fontweight='bold')
        ax.set_xlabel('波数 (cm$^{-1}$)')
        ax.set_ylabel('反射率 (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 第四个子图：替换为带SNR和噪声减少指标的全光谱图
        ax = axes[1, 1]
        ax.plot(x, y, 'b-', linewidth=1, alpha=0.4, label='原始数据')
        ax.plot(x, y_wavelet, 'g-', linewidth=1.5, label='小波滤波')
        ax.set_title('全光谱去噪效果与性能指标', fontweight='bold', fontsize=12)
        ax.set_xlabel('波数 (cm$^{-1}$)')
        ax.set_ylabel('反射率 (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 计算SNR改善
        snr_orig, snr_wt, snr_improve_wt = self.calculate_snr(y, y_wavelet)
        ax.text(0.02, 0.98, f'SNR改善: {snr_improve_wt:.2f} dB', 
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 计算噪声减少百分比
        noise_orig = self.estimate_noise_level(y)
        noise_wt = self.estimate_noise_level(y_wavelet)
        noise_reduction = (1 - noise_wt / (noise_orig + np.finfo(float).eps)) * 100
        ax.text(0.02, 0.85, f'噪声减少: {noise_reduction:.1f}%', 
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 保存合并后的图
        self.save_figure(fig, 'wavelet_denoising_summary.png')
        try:
            fig.show()
        except:
            pass
    
    def plot_denoising_residual(self, sample_key='附件1'):
        """
        对去噪过程的残差进行时域和频域分析。
        一个好的去噪算法，其残差应表现为白噪声特性。
        """
        print(f"\n正在对 {sample_key} 进行去噪残差分析...")
        
        df = self.data[sample_key]
        x = df['wavenumber'].values
        y_orig = df['reflectance'].values
        
        # 获取去噪后的数据
        y_denoised = self.wavelet_threshold_filter(y_orig)
        
        # 计算残差
        residual = y_orig - y_denoised
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        fig.suptitle('去噪残差分析', fontsize=16, fontweight='bold')
        
        # 时域分析
        ax1 = axes[0]
        ax1.plot(x, residual, color='gray', linewidth=0.8, alpha=0.9)
        ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
        ax1.set_title('时域残差 (原始信号 - 去噪信号)', fontweight='bold')
        ax1.set_xlabel('波数 (cm$^{-1}$)')
        ax1.set_ylabel('残差幅值')
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_res = np.mean(residual)
        # 避免因浮点数精度问题显示 "-0.0000"
        if abs(mean_res) < 5e-5:  # 阈值小于格式化显示的最小精度
            mean_res = 0.0
        std_res = np.std(residual)
        ax1.text(0.02, 0.98, f'均值: {mean_res:.4f}\n标准差: {std_res:.4f}',
                 transform=ax1.transAxes, va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # 频域分析 (功率谱密度)
        ax2 = axes[1]
        # 使用 signal.welch 来分析残差的频率成分
        fs = 1 / np.mean(np.diff(x)) # 采样频率的倒数是波数间隔
        freqs, psd = signal.welch(residual, fs=fs, nperseg=1024)
        
        ax2.semilogy(freqs, psd, color='darkblue')
        ax2.set_title('残差功率谱密度 (PSD)', fontweight='bold')
        ax2.set_xlabel('频率 (cycles / cm$^{-1}$)')
        ax2.set_ylabel('功率 / 频率 (dB/Hz)')
        ax2.grid(True, which='both', linestyle='--', alpha=0.3)
        ax2.text(0.98, 0.98, '注: 理想的去噪残差应接近白噪声,其能量在频域中分布平坦, 无明显峰值。',
                 transform=ax2.transAxes, va='top', ha='right', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        self.save_figure(fig, 'denoising_residual_analysis.png')
        try:
            fig.show()
        except:
            pass
    
    def data_quality_assessment(self):
        """数据质量评估"""
        print("\n" + "="*60)
        print("数据质量评估")
        print("="*60)
        
        for key, df in self.data.items():
            material, angle = self.materials[key]
            print(f"\n{key} ({material}, 入射角{angle}):")
            
            # 数据完整性
            total_points = len(df)
            valid_points = df['reflectance'].notna().sum()
            print(f"  数据完整性: {valid_points}/{total_points} ({100*valid_points/total_points:.1f}%)")
            
            # 数据范围合理性
            ref_range = df['reflectance'].max() - df['reflectance'].min()
            print(f"  反射率动态范围: {ref_range:.2f}%")
            
            # 数据平滑性 (基于更稳健的噪声估计)
            noise_level = self.estimate_noise_level(df['reflectance'].values)
            smoothness = 1 / (1 + noise_level)
            print(f"  数据平滑性指标: {smoothness:.4f} (基于小波噪声估计)")
            
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始碳化硅外延层厚度测量数据可视化分析")
        print("="*60)
        
        self.load_data()  # 加载数据
        
        self.basic_statistics()  # 基本统计
        
        print("\n正在生成原始光谱图...")
        self.plot_raw_spectra()  # 绘制原始光谱
        
        print("\n" + "="*60)
        print("开始去噪分析")
        self.detailed_denoising_analysis()  # 详细去噪分析与数据保存
        
        print("\n正在生成合并去噪分析图...")
        self.plot_zoomed_denoising_comparison()  # 合并去噪效果综合分析图
        
        self.plot_denoising_residual()  # 残差分析
        
        self.data_quality_assessment()  # 数据质量评估
        
        print("\n" + "="*80)
        print("分析完成！")


if __name__ == "__main__":
    analyzer = SpectralAnalyzer()
    analyzer.run_complete_analysis()