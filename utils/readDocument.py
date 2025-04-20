from model.qwenModel import QwenModel
from model.documentProcessing import documentProcessing

def readDocument(path:str,embedding_name:str,qw_model_name:str):
    qwenModel1 =QwenModel(
        api_key="sk-568bd13551dd42ae9c623bd04504ba02",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwenModel1.material_readpaper(path,qw_model_name)


