from MaterialTool import MaterialTool as MT
import numpy as np
class CalculatedMaterials:
    def __init__(self, number_polyfit:list,
                 build_composites:list, set_thickness:list, wl:list,
                 ):
        self.number_polyfit=number_polyfit
        self.build_composites=build_composites
        self.set_thickness:list=set_thickness
        self.wl:list=wl
        self._dir_path = rf"E:\BaiduSyncdisk\1-毕业论文\仿真\材料数据库\新建文件夹"
        pass

    def calculate_fit_data(self, method=MT.interpolite_composites):
        common_wl =np.linspace(self.wl[0],self.wl[1] ,self.wl[2])
        composites = MT.load_Material(self._dir_path, self.build_composites)
        solution_fit=MT.fit_composites(method, composites, common_wl, self.number_polyfit, True)

        return solution_fit



