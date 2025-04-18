import pandas as pd
from pandas import DataFrame
class Material:
    #设置私有成员thickness

    def __init__(self, name, path, thickness):
        self.fitted_date = None
        self.name: str = name
        self.path: str = path
        self.thickness: int = thickness
        self.date: DataFrame = pd.read_csv(self.path, sep='\s+')
    #定义get和set方法
    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        pass
    def get_path(self):
        return self.path

    def set_path(self, path):
        self.path = path
        pass

    def get_thickness(self):
        return self.thickness

    def set_thickness(self, thickness):
        self.thickness = thickness
        pass

    def get_data(self):
        return self.date

    def set_data(self, data: DataFrame):
        self.date = data
        pass
    def set_fitted_data(self, data: DataFrame):
        self.fitted_date = data
        pass

    def get_fitted_data(self):
        return self.fitted_date
    def __str__(self):
        return "name:" + self.name + " path:" + self.path + " thickness:" + str(self.thickness)
