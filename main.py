import asyncio
import os.path

from fastapi import FastAPI
from utils.lifespan import lifespan
from routers import images,param
from utils.readDocument import pdf_to_database
from utils.readDocument import communication_model


app = FastAPI(lifespan=lifespan)
app.include_router(param.router)
app.include_router(images.router)




if __name__=='__main__':
    from database.connection import Base, engine, get_db
    from database.modles import *
    from model.qwenModel import QwenModel
    from utils.CalculatedMaterials import CalculatedMaterials

    Base.metadata.create_all(bind=engine)
    async def main():
        # await pdf_to_database(["qwen-long", "qwen-plus","text-embedding-v3"])
        # await communication_model("qwq-plus",  deepmind=True, stream=False)
        # await communication_format("qwen-plus",  deepmind=True, stream=False)
        qw=QwenModel(
            api_key="sk-568bd13551dd42ae9c623bd04504ba02",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        CM=CalculatedMaterials(
            build_composites=['TiO2', 'MoS2', 'SiO2'],
            set_thickness=[50,20,50],
            wl=[200, 1200, 1000]
        )
        # message=await qw.fit_format(qw_model_name="qwen-plus",CM=CM)
        await qw.fit_Agent(qw_model_name="qwen-plus", qw_model_vl_name="qwen-vl-max-latest", CM=CM)
        await qw.client.close()
    asyncio.run(main())

    # from utils.CalculatedMaterials import CalculatedMaterials
    # import os
    #
    # # 设置默认路径，为避免每次都要输入路径
    # # os.chdir(r'E:\PycharmProjects\AgentMaterial')
    # #
    # CM = CalculatedMaterials(build_composites=['SiO2', 'Ag', 'TiO2'],
    #                          set_thickness=[50, 20, 50],
    #                          wl=[200, 1200, 1000])
    # img_path, zipped = CM.calculate_fit_data(number_polyfit=[3], method='interpolite_composites')
    # print(img_path, zipped)













