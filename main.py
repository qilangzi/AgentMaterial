from fastapi import FastAPI
from utils.lifespan import lifespan

from routers import images,param


app = FastAPI(lifespan=lifespan)
app.include_router(param.router)
app.include_router(images.router)

if __name__=='__main__':
    # from utils.CalculatedMaterials import CalculatedMaterials
    # from utils.MaterialTool import MaterialTool
    # data={
    #     "text": "nihao",
    #     "file_from": "E://ow/nam/02.png",
    #     "file_DIO": "10.1016/j.foodres.2017.07.078",
    #     "status": 0
    # }


    # CalculatedMaterials=CalculatedMaterials(
    #     number_polyfit=[3],
    #     build_composites=['TiO2','Ag','TiO2'],
    #     set_thickness=[30, 18, 35],
    #     wl=[200,1200,1000]
    # )
    # MY_fit=CalculatedMaterials.calculate_fit_data()
    # print(MY_fit)







