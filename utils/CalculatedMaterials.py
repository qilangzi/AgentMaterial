from .MaterialTool import MaterialTool as MT
import numpy as np
class CalculatedMaterials:
    def __init__(self,
                 build_composites:list, set_thickness:list, wl:list
                 ):
        self.build_composites:list=build_composites
        self.set_thickness:list=set_thickness
        self.wl:list=wl
        self._dir_path = rf"content/materialData"
        self.__composites = MT.load_Material(self._dir_path, self.build_composites)
        self.__common_wl = np.linspace(self.wl[0], self.wl[1], self.wl[2])
        pass

    def calculate_fit_data(self, number_polyfit:list[int],method:str,sub_composites:list):
        if sub_composites:
            composites = {i: self.__composites[i] for i in sub_composites}
            solution_fit,zipped= MT.fit_composites(method, composites, self.__common_wl, number_polyfit, True)
        else:
            solution_fit,zipped=MT.fit_composites(method, self.__composites, self.__common_wl, number_polyfit, True)
        return solution_fit,zipped

    def calculate_tmm_DE_T(self, bounds:list):
        composites = MT.build_composites_set(self.__composites, self.build_composites)
        MT.set_thickness_method(composites, self.set_thickness)
        # R,T,A=MT.composites_calculate_rt_tmm(composites, self.__common_wl)
        target_T = MT.filter_data(self.__common_wl)
        optimal_thickness = MT.optimal_defferential_evolution(MT.object_func_T, composites, bounds, self.__common_wl,
                                                              target_T)
        R, T, A,img_url = MT.composites_calculate_rt_tmm(composites,self.__common_wl, plot1=True)
        return optimal_thickness,R, T, A, img_url

    def calculate_tmm_DE_A(self, bounds:list):
        composites = MT.build_composites_set(self.__composites, self.set_thickness)
        MT.set_thickness_method(composites, self.set_thickness)
        pass
    def calculate_methods(self, method:str,bounds:list):
        methods={
            'calculate_tmm_DE_T':self.calculate_tmm_DE_T,
            'calculate_tmm_DE_A':self.calculate_tmm_DE_A,
        }
        return methods[method](bounds)


