# data =
# data_embedding =
# path = r'AM 铁磁石墨烯石英织物超宽带强电磁干扰屏蔽.pdf'
# import json
#
# print(type(data_embedding))
# jsonData = json.loads(data)
# jsonData['pathName'] = path
# jsonData.update(data_embedding)
#
# Base.metadata.create_all(bind=engine)
# db = next(get_db())
# db.text = Electromagentic(
#     pathname=jsonData['pathName'],
#     dio=jsonData['论文DIO'],
#     keywords=jsonData['关键字'],
#     materials=jsonData['实验材料'],
#     principle=jsonData['实验原理'],
#     methods=jsonData['实验方法'],
#     measure=jsonData['实验步骤'],
#     conclusion=jsonData['实验结论'],
#     materials_embedding=jsonData['实验材料_embedding'],
#     principle_embedding=jsonData['实验原理_embedding'],
#     methods_embedding=jsonData['实验方法_embedding'],
#     measure_embedding=jsonData['实验步骤_embedding'],
#     conclusion_embedding=jsonData['实验结论_embedding'],
#     text=jsonData['text'])
#
# db.add(db.text)
# db.commit()
# db.refresh(db.text)
#
# text=''
#     Base.metadata.create_all(bind=engine)
#     db = next(get_db())
#     record=db.query(Electromagentic).filter_by(id=2).first()
#     record.text=text
#     db.commit()

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


# async def main():
#     qw = QwenModel(api_key="sk-568bd13551dd42ae9c623bd04504ba02",
#                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
#     # xj=await qw.text_to_database(qw_model_name="qwen-long", ipath=r"content/document/input")
#     # print("main函数执行完毕")
#     # print(f"消耗的token为:{xj}")
#
#     # dj=await qw.text_to_database_by_fileid(qw_model_name="qwen-plus", fileid=[14,14])
#     # print(f"消耗的token为:{dj}")
#
#     ej = await qw.text_to_database_embedding(embedding_name="text-embedding-v3", fileid=[5, 15])
#
#     await qw.client.close()
# asyncio.run(main())