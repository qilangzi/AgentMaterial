import os.path
from model.qwenModel import QwenModel
from model.documentProcessing import documentProcessing
import logging



def readDocument(path: str, embedding_name: str, qw_model_name: str) :
    """
    保留目前不用
    :param path:
    :param embedding_name:
    :param qw_model_name:
    :return:
    """
    qwenModel1 = QwenModel(
        api_key="sk-568bd13551dd42ae9c623bd04504ba02",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    #读取pdf,返回str
    dcpdf = documentProcessing()
    paper = dcpdf.ext_from_pdf(path)

    #ai进行总结，返回json
    ai_answer = qwenModel1.material_readpaper(paper, qw_model_name)
    print(ai_answer)

    #得到embedding的数据，json
    embedding_json = {}
    for i in ai_answer.keys()[2:]:
        data_i = ai_answer[i]
        ai_answere = qwenModel1.text_embeding(embedding_name, data_i)
        embedding_json[f'{data_i}-embed'] = ai_answere

    ai_answer['path'] = os.path.basename(path)
    return ai_answer, embedding_json


def batches_updatatodatabase(qw_model_name:str, embedding_name:str):
    #写下注释
    """
    保留目前不用
    :param qw_model_name:
    :param embedding_name:
    :return:
    """
    dcpdf = documentProcessing()
    datas = dcpdf.batches_text_pdf()
    all_datas = []
    all_embedding = []
    for k in datas:
        paper = datas[k]
        qwenModel1 = QwenModel(
            api_key="sk-568bd13551dd42ae9c623bd04504ba02",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        ai_answer = qwenModel1.material_readpaper(paper, qw_model_name)
        embedding_json = {}
        for i in ai_answer.keys()[2:]:
            data_i = ai_answer[i]
            ai_answere = qwenModel1.text_embeding(embedding_name, data_i)
            embedding_json[f'{data_i}-embed'] = ai_answere
        ai_answer['path'] = os.path.basename(k)
        all_datas.append(ai_answer)
        all_embedding.append(embedding_json)
    return all_datas, all_embedding

async def pdf_to_database(qw_model_name: list):
    """
    主要的执行函数：输入列表3个值，1为读取pdf的模型，2为参数结构化的模型，3为文本转向量的模型，
    提前数据，并存入数据库
    :param qw_model_name: 输入列表3个值，1为读取pdf的模型，2为参数结构化的模型，3为文本转向量的模型
    """
    qw = QwenModel(
        api_key="sk-568bd13551dd42ae9c623bd04504ba02",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        logger=None
    )
    xj,maxId = await qw.text_to_database(qw_model_name=qw_model_name[0], ipath=r"content/document/input")
    print(f'text 存储完毕,消耗的token为:{xj}\n\n')
    minvalue=min(maxId)
    maxvalue=max(maxId)
    dj = await qw.text_to_database_by_fileid(qw_model_name=qw_model_name[1], fileid=[minvalue, maxvalue])
    print(f'json 存储完毕,消耗的token为:{dj}\n\n')

    ej = await qw.text_to_database_embedding(embedding_name=qw_model_name[2], fileid=[minvalue, maxvalue])
    print('embedding 存储完毕')
    await qw.client.close()

async def communication_model(qw:QwenModel,qw_model_name: str, deepmind: bool = False, stream: bool = False):
    logger = logging.getLogger('qwenModel')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('model/qwenModel.log', encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    qw.logger=logger
    await qw.communication_model(qw_model_name=qw_model_name, deepmind=deepmind, stream=stream)
    # try:
    #     qw.logger=logger
    #     await qw.communication_model(qw_model_name=qw_model_name, deepmind=deepmind, stream=stream)
    # except Exception as e:
    #     logger.info(e)
    await qw.client.close()

