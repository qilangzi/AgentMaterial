import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from tmm import coh_tmm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import differential_evolution
from .Material import Material
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
class MaterialTool:

    def __init__(self):
        pass
    @staticmethod
    def create_method_mapping():
        """
        创建一个公共的参数映射方法
        """
        return {
            "interpolite_composites": MaterialTool.interpolite_composites,
            "fit_composites": MaterialTool.fit_composites,
        }
    @staticmethod
    def composites_calculate_rt_tmm(composites: dict, wl, n_air: int = 1, plot1: bool = False):
        '''
        计算材料反射率、透射率
        使用转移矩阵的方法
        :param composites: 待计算材料集
        :param wl: 波长（nm）
        :param n_air: 空气折射率
        :param plot1: 是否绘制反射率、透射率曲线
        :return:反射率、透射率
        '''
        R, T = [], []
        Ab = []
        img_url=''
        all_composites = list(composites.values())
        dw = [j.get_thickness() for j in all_composites]
        print(dw)
        for i in range(len(wl)):
            nw = [j.get_fitted_data()["n_k"][i] for j in all_composites]
            n_list = [n_air] + nw + [n_air]  # 层结构折射率列表
            d_list = [np.inf] + dw + [np.inf]  # 各层厚度（空气层设为无穷大）
            result = coh_tmm('s', n_list, d_list, 0, wl[i])
            R.append(result['R'])
            T.append(result['T'])
            ab = 1 - result['R'] - result['T']
            Ab.append(ab)
        if plot1:
            plt.figure(figsize=(8, 5))
            plt.plot(wl, R, 'g-', label='Reflectance (R)')
            plt.plot(wl, T, 'b-', label='Transmittance (T)')
            plt.plot(wl, Ab, 'r-', label='Absorption')
            plt.grid()
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance / Transmittance")
            plt.legend()
            img_url = fr"content\image_temp\DE_image\{list(composites.keys())}.png"
            # 保存图片
            plt.savefig(img_url)
            # plt.show()

        return np.array(R), np.array(T), np.array(Ab),img_url

    @staticmethod
    def build_composites_set(old_composites, build_composites: list):
        '''
        构建预设材料模型
        根据材料构建方式build_composites,调整材料数据集old_composites的排列方式和数目
        :param old_composites: 材料集
        :param build_composites: 待构建材料集
        '''
        composites = {}
        jsq = {i: 1 for i in old_composites.keys()}
        for i in build_composites:
            if i in composites.keys():
                new_i = i + f'_{jsq[i]}'
                composites[new_i] = copy.copy(old_composites[i])
                composites[new_i].set_name(new_i)
                jsq[i] += 1
            else:
                composites[i] = old_composites[i]
        return composites

    @staticmethod
    def set_thickness_method(composites, set_thickness_value: list):
        '''
        设置材料厚度
        :param composites: 待设置厚度材料集
        :param set_thickness_value: 设置厚度值（nm）
        '''
        for i in range(len(set_thickness_value)):
            list(composites.values())[i].set_thickness(set_thickness_value[i])

    @staticmethod
    def load_Material(dir_path: str, build_composites: list):
        '''
        读取需要使用的材料数据，并返回材料集（包含材料对象）
        :param dir_path: 材料数据文件夹路径
        :param build_composites: 待构建材料名称
        '''
        build_Material = set(build_composites)
        composites = {}
        for i in os.listdir(dir_path):
            if i.endswith(".csv") or i.endswith(".txt"):
                name = os.path.basename(i).split(".")[0]
                if name in build_Material:
                    composites[name] = Material(name, os.path.join(dir_path, i), 100)
        return composites

    @staticmethod
    def calculate_fitting_errors(y_true, y_pred):
        """
        此函数用于计算多种拟合误差指标。

        参数:
        y_true (array-like): 真实值数组。
        y_pred (array-like): 预测值数组。

        返回:
        dict: 包含均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）和决定系数（R^2）的字典。
        """
        # 计算均方误差（MSE）
        mse = mean_squared_error(y_true, y_pred)
        # 计算均方根误差（RMSE）
        rmse = np.sqrt(mse)
        # 计算平均绝对误差（MAE）
        mae = mean_absolute_error(y_true, y_pred)
        # 计算决定系数（R^2）
        r2 = r2_score(y_true, y_pred)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R^2': r2
        }

    @staticmethod
    def fit_composites(
            method:str,
            composites: dict,
            common_wl,
            number_polyfit: list,
            all_plot_data: bool = False,
            *args,
            **kwargs
    ):
        '''
        初步拟合所有材料
        :param mathod: 拟合方法
        :param composites: 待拟合材料集
        :param common_wl: 待拟合数据波长（nm）
        :param number_polyfit: 拟合参数
        :param all_plot_data: 是否绘制所有拟合数据
        :return:
        '''
        sm = len(composites.keys())
        # print(f"共{sm}种材料")
        nk = 1
        fit_error = []
        mathod=MaterialTool.create_method_mapping()[method]
        if all_plot_data:
            plt.figure(figsize=(12, 3 * 6))
        for i in composites.values():
            data_error={}
            data = i.get_data()
            data.sort_values(by="wl", ascending=True, inplace=True)
            wl = data["wl"].values * 1e3
            n = data["n"].values
            k = data["k"].values
            n_1, k_1,n_error,k_error = mathod(wl, common_wl, n, k, number_polyfit, *args, **kwargs)
            data_error["n_error"] = n_error
            data_error["k_error"] = k_error
            fit_error.append(data_error)
            n_k = n_1 + 1j * k_1
            i.set_fitted_data(pd.DataFrame({"wl": common_wl, "n_k": n_k}))
            maxindex = np.where(wl < max(common_wl))[0][-1]
            minindex = np.where(wl > min(common_wl))[0][0]
            if all_plot_data:
                plt.subplot(sm, 2, nk)
                plt.plot(wl[minindex:maxindex + 1], n[minindex:maxindex + 1], 'b.', label='n')
                plt.plot(common_wl, n_1, "r-", label='polyfit')
                plt.title(i.get_name())
                plt.legend()
                plt.subplot(sm, 2, nk + 1)
                nk += 2
                plt.plot(wl[minindex:maxindex + 1], k[minindex:maxindex + 1], 'b.', label='k')
                plt.plot(common_wl, k_1, "r-", label='polyfit')
                plt.title(i.get_name())
                plt.legend()
        zipped=dict(zip(list(composites.keys()),fit_error))
        if all_plot_data:
            img_url = fr"content\image_temp\fit_image\{list(composites.keys())}.png"
            plt.tight_layout()
            #保存图片
            plt.savefig(img_url)
            #缓存图片url地址给用户
            return f'{list(composites.keys())}.png',zipped
        else:
            return zipped

    @staticmethod
    def plot_composites(wl, common_wl, n, k, number_polyfit: list):
        '''
        拟合方法：多项式拟合
        :param wl: 原始数据波长（nm）
        :param common_wl: 待拟合数据波长（nm）
        :param n: 原始数据n
        :param k: 原始数据k
        :param number_polyfit: 拟合参数，此处仅一个n阶数
        :return:
        '''
        eq1 = np.polyfit(wl, n, number_polyfit[0])
        n_1 = np.polyval(eq1, common_wl)
        n_fit=np.polyval(eq1, wl)
        n_error=MaterialTool.calculate_fitting_errors(n, n_fit)
        eq2 = np.polyfit(wl, k, number_polyfit[0])
        k_1 = np.polyval(eq2, common_wl)
        k_fit=np.polyval(eq2, wl)
        k_error=MaterialTool.calculate_fitting_errors(k, k_fit)
        return n_1, k_1, n_error, k_error

    @staticmethod
    def interpolite_composites(wl, common_wl, n, k1, number_polyfit: list):
        '''
        拟合方法：样条插值拟合
        :param wl: 原始数据波长（nm）
        :param common_wl: 待拟合数据波长（nm）
        :param n: 原始数据n
        :param k1: 原始数据k
        :param number_polyfit: 拟合参数，此处仅1个参述（阶数）
        :return:
        '''
        try:
            eq1 = UnivariateSpline(wl, n, k=number_polyfit[0], s=0.5)
            n_1 = eq1(common_wl)
            n_fit=eq1(wl)
            n_error=MaterialTool.calculate_fitting_errors(n, n_fit)
            eq2 = UnivariateSpline(wl, k1, k=number_polyfit[0], s=0.5)
            k_1 = eq2(common_wl)
            k_fit=eq2(wl)
            k_error=MaterialTool.calculate_fitting_errors(k1, k_fit)
            return n_1, k_1, n_error, k_error
        except Exception as e:
            print(e)

    @staticmethod
    def plot_name(name, composites: dict):
        '''
        绘制单个材料,用于查看拟合效果
        :param name: 待绘制材料名称
        :param composites: 待绘制材料集
        :return:
        '''
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        data = composites[name].get_data()
        fit_data = composites[name].get_fitted_data()
        data.sort_values(by="wl", ascending=True, inplace=True)
        wl = data["wl"].values * 1e3
        n = data["n"].values
        k = data["k"].values
        plt.plot(wl, n, 'b.', label='n')
        wl_fit = fit_data["wl"]
        n_fit = fit_data["n_k"].apply(lambda x: x.real)
        plt.plot(wl_fit, n_fit, "r-", label='polyfit')
        plt.legend()
        plt.title(name)
        plt.subplot(1, 2, 2)
        plt.plot(wl, k, 'b.', label='k')
        wl_fit = fit_data["wl"]
        k_fit = fit_data["n_k"].apply(lambda x: x.imag)
        plt.plot(wl_fit, k_fit, "r-", label='polyfit')
        plt.legend()
        plt.title(name)
        plt.show()

    @staticmethod
    def fit_single_composites(
            mathod,
            fit_name: str,
            composites: dict,
            common_wl,
            number_polyfit: list,
            *args,
            **kwargs
    ):
        '''
        拟合单个材料
        :param mathod: 拟合方法
        :param fit_name: 待拟合材料名称
        :param composites: 待拟合材料集
        :param common_wl: 波段
        :param number_polyfit: 拟合输入参数
        :param args: 拟合参数
        :param kwargs: 拟合参数
        :return: None
        '''
        data = composites[fit_name].get_data()
        data.sort_values(by="wl", ascending=True, inplace=True)
        wl = data["wl"].values * 1e3
        n = data["n"].values
        k = data["k"].values
        n_1, k_1 = mathod(wl, common_wl, n, k, number_polyfit, *args, **kwargs)
        composites[fit_name].set_fitted_data(pd.DataFrame({"wl": common_wl, "n_k": n_1 + 1j * k_1}))

    @staticmethod
    def object_func_A(optimal_thickness: list, composites: dict, wl, target):
        '''
        进化差分算法的目标函数
        :param optimal_thickness: 待优化的厚度
        :param composites: 待优化的材料集
        :param wl: 波段
        :param target: 目标反射率
        :return: 目标函数值
        '''
        n = 0
        for i in composites.values():
            i.set_thickness(optimal_thickness[n])
            n += 1
        _, _, A = MaterialTool.composites_calculate_rt_tmm(
            composites,
            wl,
        )
        target_function = np.sum((A - target) ** 2)
        return target_function

    @staticmethod
    def object_func_T(optimal_thickness: list, composites: dict, wl, target):
        '''
        进化差分算法的目标函数
        :param optimal_thickness: 待优化的厚度
        :param composites: 待优化的材料集
        :param wl: 波段
        :param target: 目标反射率
        :return: 目标函数值
        '''
        n = 0
        for i in composites.values():
            i.set_thickness(optimal_thickness[n])
            n += 1
        _, T, _ = MaterialTool.composites_calculate_rt_tmm(
            composites,
            wl,
        )
        target_function = np.sum((T - target) ** 2)
        return target_function

    @staticmethod
    def optimal_defferential_evolution(mathod, composites: dict, bounds, wl, target):
        """
        差分进化算法优化厚度
        :param composites: 待优化的材料集
        :param bounds: 厚度范围0.
        :param wl: 波段
        :param target: 预期值
        :return: 最佳厚度
        """
        optimal_thickness = differential_evolution(
            mathod,
            bounds,
            args=(composites, wl, target),
            strategy='best1bin',
            tol=1e-6
        ).x
        return optimal_thickness

    @staticmethod
    def filter_data(wl):
        """
        预期模型：可见光高透过全为 1，其他波段全为 0
        :param wl:目标波段
        """
        filtered_data = []
        for i in range(len(wl)):
            if wl[i] < 400 or wl[i] > 700:
                filtered_data.append(0)
            else:
                filtered_data.append(1)
        return filtered_data

    @staticmethod
    def all_absorb(wl):
        """
        预期模型：全波段，预期吸收率全为abs = 1
        :param wl:目标波段
        """
        filtered_data = []
        for i in range(len(wl)):
            filtered_data.append(1)
        return filtered_data


