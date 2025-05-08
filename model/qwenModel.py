import base64
import os.path
from sqlalchemy import func
from openai import OpenAI
import json
from openai import AsyncOpenAI
from pathlib import Path
from database.connection import get_db
from database.modles import *
from sqlalchemy import and_
from utils.CalculatedMaterials import CalculatedMaterials


class QwenModel:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.__messages = []
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        pass

    async def material_model_prediction(self, qw_model_name: str, system: str, prompt: str):
        """
        待使用
        最简单的聊天聊天程序
        :param qw_model_name:模型名字
        :param system: 高级命令
        :param prompt: 聊天提示词
        :return: 聊天回复
        """
        completion = await self.client.chat.completions.create(
            model=qw_model_name,  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt}
            ]
        )
        return completion.choices[0].message.content

    # 阅读论文的模型
    async def material_readpaper(self, paper: str, qw_model_name: str):
        """
        待使用
        格式化输出json的Agent
        :param paper: 经过pdf阅读Agent总结之后的数据
        :param qw_model_name: 模型名称
        :return: json格式数据
        """
        system = """
        你是一个论文实验总结助手,是一个严瑾是科学助手，总结该论文中的[DOI,关键字,实验材料,实验原理，实验方法，实验步骤，实验结论]。
                要求：
                1.	使用json格式返回数据。[]中的7个部分作为键，返回内容为值
                2.	严格遵守用户要求，不要返回json以外的任何内容，便于用户后期调用
        """
        prompt = f"""
                # 原文：{paper}
                """
        try:
            odj = await self.material_model_prediction(qw_model_name, system, prompt)
            positions1 = odj.find('{')
            positions2 = odj.rfind('}')
            od = odj[positions1:positions2 + 1]
            xd = json.loads(od)
            return xd
        except Exception as e:
            print(e)

    async def pdf_reader(self, path: str, qw_model_name: str, file_delete: bool = True):
        """
            阅读pdf的Agent,返回论文总结的文本数据
            :param path: 论文路径
            :param qw_model_name:模型名字
            :param file_delete: 是否删除上传的任务id
            :return: 总结后的文本数据
            """
        system = """
                      你是一个论文实验总结助手,是一个严瑾是科学助手，总结该论文中的[论文的DIO,关键字,实验材料,实验原理，实验方法，实验步骤(列表类型)，实验结论]。
                                        要求：
                                        1.	[]中的6个部分为标题，返回内容。
                                        2.	如果你找不到相关标题的内容请在该标题下写None。
                                        3.  回复请使用中文
                                        4.  实验步骤的内容是列表
                                        5.  尽量详细一点
             """
        # 提交任务
        try:
            file_object = await self.client.files.create(file=Path(path), purpose="file-extract")
            print(file_object.id)
        except Exception as e:
            return e
        completion = await self.client.chat.completions.create(
            model=qw_model_name,
            messages=[
                {'role': 'system', 'content': f'{system}'},
                # 请将 'file-fe-xxx'替换为您实际对话场景所使用的 file-id。
                {'role': 'system', 'content': f"fileid://{file_object.id}"},
                {'role': 'user', 'content': '这篇文章讲了什么？'}
            ],
            stream=True,
            stream_options={"include_usage": True}
        )
        full_content = ""
        total_tokens1 = 0
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                # 拼接输出内容
                full_content += chunk.choices[0].delta.content
                # print(chunk.model_dump())
            if chunk.usage:
                # 累加 token 数量
                total_tokens1 += chunk.usage.total_tokens
        if file_delete:
            await self.client.files.delete(file_object.id)

        return full_content, total_tokens1

    async def formatContent(self, full_content: str, qw_model_name: str):
        """
        格式化成json输出的Agent
        :param full_content:阅读Agent总结的数据
        :param qw_model_name:模型名字
        :return:json数据和tokens使用数目
        """
        system = """
                           总结该论文中的[论文DIO,关键字,实验材料,实验原理，实验方法，实验步骤，实验结论]。
                                   要求：
                                   1.  使用json格式返回数据。列表中的6个部分作为键，返回内容为值
                                   2.  严格遵守用户要求，不要返回json以外的任何内容，便于用户后期调用
                                   3.  回复请使用中文
                                   4.  实验步骤的值是列表
                                   5.  如果文章中没有该模块的内容，请返回None。
                                   6.  尽量不要删减文中的内容
                                   7,  不要出现混乱的符号r'">'
                                   8,  不要出现大量重复的字符
                           """
        completion = await self.client.chat.completions.create(
            model=qw_model_name,
            messages=[
                {'role': 'system', 'content': f'{system}'},
                {'role': 'user', 'content': f'{full_content}'},
            ],
            response_format={"type": "json_object"}
        )
        # 获取生成的 JSON 内容
        json_content = completion.choices[0].message.content

        # 获取 token 使用情况
        total_tokens = completion.usage.total_tokens if hasattr(completion, 'usage') else None

        return json_content, total_tokens

    def content_to_json(self, full_content: str):
        """
        将文本str转json
        :param full_content:文本
        :return: json
        """
        positions1 = full_content.find('{')
        positions2 = full_content.rfind('}')
        od = full_content[positions1:positions2 + 1]
        xd = json.loads(od)
        return xd

    async def text_embeding(self, embedding_name: str, text: str):
        """
        是数据向量化
        :param embedding_name:模型名字
        :param text: 数据
        :return: 向量化浮点数；list
        """
        embedding = await self.client.embeddings.create(
            model=embedding_name,
            input=text
        )
        return embedding.data[0].embedding

    async def textList_embeding(self, embedding_name: str, jsonlist: dict):
        """
        使数据向量化
        :param embedding_name:模型名字
        :param jsonlist: 需要向量化的元素的key放在list里
        :return: jsonlist的key_embedding:向量化浮点数；list
        """
        namelist = list(jsonlist.keys())
        textList = [f'{i}' for i in list(jsonlist.values())]
        # print(textList)
        completion = await self.client.embeddings.create(
            model=embedding_name,
            input=textList,
            # dimensions=1024,
            encoding_format='float',
        )
        data1 = json.loads(completion.model_dump_json())
        allData = {}
        for i in data1['data']:
            allData[f"{namelist[i['index']]}_embedding"] = i['embedding']
        return allData

    async def text_to_database(self, qw_model_name: str, ipath: str):
        """
        将pdf阅读Agent数据写入数据库
        :param qw_model_name:模型名字
        :param ipath:文件存放地址
        :return:taken使用情况
        """
        maxId = []
        total_tokens = 0
        for i in os.listdir(ipath):
            path = os.path.join(ipath, i)
            full_content, total_tokens1 = await self.pdf_reader(qw_model_name=qw_model_name, path=path)

            pathname = os.path.basename(path)
            db = next(get_db())
            max_id = db.query(func.max(Electromagentic.id)).scalar()
            maxId.append(max_id + 1)
            electroMagentic = Electromagentic(
                id=max_id + 1,
                text=full_content,
                pathname=pathname
            )
            db.add(electroMagentic)
            db.commit()
            print(f'{i}存储成功')
            db.refresh(electroMagentic)
            total_tokens += total_tokens1
        return total_tokens, maxId

    async def text_to_database_by_fileid(self, qw_model_name: str, fileid: list):
        """
        使用在数据库已经有text的情况，使用text数据产生格式化的json数据并填入数据库
        :param qw_model_name: 模型名字
        :param fileid: 数据库的id，2个元素列表，0开始id,1介绍id
        :return: takens的使用情况
        """
        db = next(get_db())
        full_content_list = db.query(Electromagentic.text).filter(and_(Electromagentic.id >= fileid[0],
                                                                       Electromagentic.id <= fileid[1])).all()
        text_list = [full_content[0] for full_content in full_content_list]
        taken = 0
        for i in range(fileid[0], fileid[1] + 1):
            xj, takens = await self.formatContent(full_content=text_list[i - fileid[0]], qw_model_name=qw_model_name)
            taken += takens
            try:
                jsonList = self.content_to_json(full_content=xj)
                record = db.query(Electromagentic).filter(Electromagentic.id == i).first()
                record.dio = jsonList['论文DIO']
                record.keywords = jsonList['关键字']
                record.measure = jsonList['实验步骤']
                record.methods = jsonList['实验方法']
                record.principle = jsonList['实验原理']
                record.materials = jsonList['实验材料']
                record.conclusion = jsonList['实验结论']
                db.commit()
                print(f'{i} 转换成功')
            except Exception as e:
                print(e)
                print(xj)
                continue
        return taken

    async def text_to_database_embedding(self, embedding_name: str, fileid: list):
        """
        使用在数据库有json数据填充后的情况，将几个部分的数据格向量化，并填入数据库
        :param embedding_name:模型名字
        :param fileid: 数据库id范围，2个元素的列表，0：id开始，1：id计算
        """
        db = next(get_db())
        full_content_list = db.query(Electromagentic).filter(and_(Electromagentic.id >= fileid[0],
                                                                  Electromagentic.id <= fileid[1])).all()
        materials_list = [full_content.materials for full_content in full_content_list]
        methods_list = [full_content.methods for full_content in full_content_list]
        principle_list = [full_content.principle for full_content in full_content_list]
        measure_list = [full_content.measure for full_content in full_content_list]
        conclusion_list = [full_content.conclusion for full_content in full_content_list]
        for i in range(fileid[0], fileid[1] + 1):
            jsonList = {'实验材料': materials_list[i - fileid[0]],
                        '实验方法': methods_list[i - fileid[0]],
                        '实验原理': principle_list[i - fileid[0]],
                        '实验步骤': measure_list[i - fileid[0]],
                        '实验结论': conclusion_list[i - fileid[0]]}
            jsonData = await self.textList_embeding(embedding_name=embedding_name, jsonlist=jsonList)
            # print(f'{jsonData}\n')
            record = db.query(Electromagentic).filter(Electromagentic.id == i).first()
            record.materials_embedding = jsonData['实验材料_embedding']
            record.principle_embedding = jsonData['实验原理_embedding']
            record.methods_embedding = jsonData['实验方法_embedding']
            record.measure_embedding = jsonData['实验步骤_embedding']
            record.conclusion_embedding = jsonData['实验结论_embedding']
            db.commit()
            print(f'{i}向量化完毕')

    async def communication_model(self, qw_model_name: str, deepmind: bool = False, stream: bool = False):
        if deepmind:
            turn = 1
            while True:
                reasoning_content = ""  # 定义完整思考过程
                answer_content = ""  # 定义完整回复
                is_answering = False  # 判断是否结束思考过程并开始回复
                user_input = input("请输入：\n")
                if user_input == "bye":
                    break
                self.__messages.append({"role": "user", "content": user_input})
                completion = await self.client.chat.completions.create(
                    model=qw_model_name,  # 此处以 qwq-32b 为例，可按需更换模型名称
                    messages=self.__messages,
                    # QwQ 模型仅支持流式输出方式调用
                    stream=True,
                    # 解除以下注释会在最后一个chunk返回Token使用量
                    stream_options={
                        "include_usage": True
                    }
                )
                print("\n" + "=" * 20 + f"第{turn}思考过程" + "=" * 20 + "\n")
                async for chunk in completion:
                    if not chunk.choices:
                        print('\nUsage:')
                        print(chunk.usage)
                    else:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                            # hasattr(delta, reasoning_content) 就是判断delta对象中是否有名为reasoning_content的属性。
                            # 如果有则返回True，没有返回False。
                            print(delta.reasoning_content, end='', flush=True)
                            reasoning_content += delta.reasoning_content
                        else:
                            if delta.content != "" and is_answering is False:
                                print("\n" + "=" * 20 + f"第{turn}完整回复" + "=" * 20 + "\n")
                                is_answering = True
                                # 打印回复过程
                            print(delta.content, end='', flush=True)
                            answer_content += str(delta.content) if delta.content is not None else ""
                self.__messages.append({"role": "assistant", "content": answer_content})
        else:
            if stream:
                turn = 1
                while True:
                    answer_content = ""  # 定义完整回复
                    is_answering = False  # 判断是否结束思考过程并开始回复
                    user_input = input("请输入：\n")
                    if user_input == "bye":
                        break
                    self.__messages.append({"role": "user", "content": user_input})
                    completion = await self.client.chat.completions.create(
                        model=qw_model_name,  # 此处以 qwq-32b 为例，可按需更换模型名称
                        messages=self.__messages,
                        stream=True,
                        stream_options={
                            "include_usage": True
                        }
                    )
                    async for chunk in completion:
                        if not chunk.choices:
                            print('\nUsage:')
                            print(chunk.usage)
                        else:
                            delta = chunk.choices[0].delta
                            if delta.content != "" and is_answering is False:
                                print("\n" + "=" * 20 + f"第{turn}完整回复" + "=" * 20 + "\n")
                                is_answering = True
                                # 打印回复过程
                            print(delta.content, end='', flush=True)
                            answer_content += str(delta.content) if delta.content is not None else ""
                    turn += 1
                    self.__messages.append({"role": "assistant", "content": answer_content})
            else:
                while True:
                    turn = 1
                    user_input = input("请输入：\n")
                    if user_input == "bye":
                        break
                    self.__messages.append({"role": "user", "content": user_input})
                    completion = await self.client.chat.completions.create(
                        model=qw_model_name,  # 此处以 qwq-32b 为例，可按需更换模型名称
                        messages=self.__messages,
                    )
                    assistant_content = completion.choices[0].message.content
                    print("\n" + "=" * 20 + f"第{turn}完整回复" + "=" * 20 + "\n")
                    print(assistant_content)
                    turn += 1
                    self.__messages.append({"role": "assistant", "content": assistant_content})

    async def communication_format(self, qw_model_name: str, full_content: str):
        system = """
    总结输入文本的材料初始模型，并以json的格式输出，有build_composites,set_thickness,wl,三个键，
    它的值都是列表。build_composites代表复合材料膜的结构例如[TiO2,Ag,TiO2]，列表前面的代表顶层，后面的代表下层结构。
    set_thickness代表厚度（单位纳米）例如[[10,50],[50,90],[60,300]],列表中每个元素代表每层材料的厚度范围。
    wl代表波长范围（单位纳米）仅有三个元素开始,结尾和步长，例如[200,1200,1000],当没有提供具体的步长时你需要根据情况添加合适的数据(为满足拟合需要步长尽量等于（结束步长-开始步长））
    注意：波长的取值应覆盖所有波段，例如用户提供数据光学特性预测,在波段（20-80nm）,（400-600 nm）和 波段（700-2500 nm）都有涉及,你需要找到涉及波段的最小值和最大值,你的取值就全覆盖为[20,2500,2480]。
    输出的总格式{build_composites:[TiO2,Ag,TiO2]，set_thickness:[[10,50],[50,90],[60,300]],wl:[400,2500,2100]},根据输入内容的数据格式化输出。
       要求：
       1.  使用json格式返回数据。build_composites,set_thickness,wl,三个键，返回内容为列表形式
       2.  严格遵守用户要求，不要返回json以外的任何内容，便于用户后期调用
    """
        completion = await self.client.chat.completions.create(
            model=qw_model_name,
            messages=[
                {'role': 'system', 'content': f'{system}'},
                {'role': 'user', 'content': f'{full_content}'},
            ],
            response_format={"type": "json_object"}
        )
        # 获取生成的 JSON 内容
        json_content = completion.choices[0].message.content

        # 获取 token 使用情况
        total_tokens = completion.usage.total_tokens if hasattr(completion, 'usage') else None

        return json_content, total_tokens

    async def fit_format(self, qw_model_name: str, CM: CalculatedMaterials):
        """
        这是一个拟合的助手将上一个预测助手预测的材料进行拟合计算，
        :param qw_model_name: 模型名字
        :param CM：材料模型
        :return:
        """
        system = """
        你是一个数据拟合的助手，用户会提供拟合的参考数据给你，根据这些数据和意见，调用用户写好的拟合方法和调整拟合参数，来拟合数据以达到用户要求
        开始阶段(第一次输出阶段)：你拟合输入的初始参数是number_polyfit:list[int]=[3]（列表目前只有一个参数），它是拟合的必要参数可以控制拟合的效果,
                    选择的拟合方法是method="interpolite_composites",在result中说明你的选参数原因（第一次默认为‘默认值’）。
        要求：
         1. 以json的格式输出你的数据,例如:{'number_polyfit':[3]','method':'interpolite_composites','result':'默认值'} 多次反馈拟合阶段(后续阶段)：
         2.可以使用的拟合方法有["interpolite_composites","plot_composites"],你可以根据拟合结果的反馈选择合适的方法,。
         3.你只需要在method填入方法的名字即可，你也可以使用动态调整number_polyfit的值,来达到不同的拟合效果，
         4.result写明你选择参数的原因，

          以json的格式输出你的数据,例如:{'number_polyfit':[根据反馈调整参数,类型为int]','method':'根据反馈调整参数,类型为str','result':'返回你更改的原因,类型为str'}
        """

        fit_messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'开始拟合'},
        ]
        nt = 1
        while True:
            qw = await self.client.chat.completions.create(
                model=qw_model_name,
                messages=fit_messages,
                response_format={"type": "json_object"}
            )
            json_content = qw.choices[0].message.content
            print(json_content)
            fit_messages.append({'role': 'assistant', 'content': json_content})
            # 获取 token 使用情况
            total_tokens = qw.usage.total_tokens if hasattr(qw, 'usage') else None
            json_data = self.content_to_json(json_content)

            if json_data['fit_status'] == True:
                print(f'拟合完成，消耗的token为:{total_tokens}')
                break
            else:
                img_path, zipped = CM.calculate_fit_data(number_polyfit=json_data['number_polyfit'],
                                                         method=json_data['method'])
                zdict = {'fit_num': nt, 'data': zipped}
                print(zdict)
                fit_messages.append({'role': 'user', 'content': f'{json.dumps(zdict)}'})
        return fit_messages

    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    async def fit_evaluation(self, qw_model_name: str, img_path:str,zipped:dict):
        """
        这是一个拟合的助手将上一个预测助手预测的材料进行拟合计算，
        :param qw_model_name: 模型名字
        :param CM：材料模型
        :return:
        """
        system = """
               你是一个数据拟合的评论大师，我会提供一张拟合结果的图片和拟合误差指标，你需要根据它们来分析并评判拟合结果
                拟合误差指标：用户一般填入{'fit_num':1,'data':{'Ag': {'n_error': {'MSE': 0.010203918504604354, 'RMSE': np.float64(0.10101444700934789), 'MAE': 0.05930365626546965, 'R^2': 0.9705055119102391}, 'k_error': {'MSE': 0.010206575730128196, 'RMSE': np.float64(0.10102759885362117), 'MAE': 0.05986610414526992, 'R^2': 0.9988746610651629}}, 'SiO2': {'n_error': {'MSE': 0.00011581736666310157, 'RMSE': np.float64(0.010761847734617953), 'MAE': 0.007805012017237496, 'R^2': 0.9725129989062254}, 'k_error': {'MSE': 0.0, 'RMSE': np.float64(0.0), 'MAE': 0.0, 'R^2': 1.0}}, 'TiO2': {'n_error': {'MSE': 0.00016698613248391714, 'RMSE': np.float64(0.012922311421874848), 'MAE': 0.009970185066536194, 'R^2': 0.9803852902836107}, 'k_error': {'MSE': 0.0, 'RMSE': np.float64(0.0), 'MAE': 0.0, 'R^2': 1.0}}}}
                拟合误差指标解释：fit_num代表,第一次进行拟合反馈给你结果，
                       数据包含了三种材料（Ag、SiO2、TiO2）。
                       每种材料有两个属性：折射率（n）和消光系数（k）。
                       每个属性有四个误差指标：MSE（均方误差）、RMSE（均方根误差）、MAE（平均绝对误差）、R²（决定系数）。
                       MSE 和 RMSE：衡量预测值与真实值之间的平均误差大小，值越小越好。
                       MAE：衡量预测值与真实值之间的平均绝对误差，值越小越好。
                       R²：衡量模型对数据的拟合程度，值越接近1越好。
                       (1) Ag（银）
                       折射率（n）：
                       MSE = 0.0102, RMSE = 0.101, MAE = 0.059, R² = 0.971
                       R²<=0.98），说明模型拟合效果不达标，于散点原数据有较大差距。
                       RMSE 和 MAE 的值相对较大，说明预测值与真实值之间存在一定的误差，图像比较简单。
                       消光系数（k）：
                       MSE = 0.0102, RMSE = 0.101, MAE = 0.060, R² = 0.999
                       R² 非常接近1，说明模型对消光系数的拟合效果非常好，图像比较简单，波动较少。
                       RMSE 和 MAE 的值略高于折射率，但整体误差仍然较小。
                       (2) SiO2（二氧化硅）
                       折射率（n）：
                       MSE = 0.00012, RMSE = 0.011, MAE = 0.008, R² = 0.973
                       R² <=0.98，图像拟合曲线和散点贴合程度不够，需要重新拟合，图像简单。
                       拟合曲线为一次线性曲线不符合要求
                       RMSE 和 MAE 的值小，说明预测误差更小。
                       消光系数（k）：
                       MSE = 0.0, RMSE = 0.0, MAE = 0.0, R² = 1.0
                       所有误差指标均为0，R²为1，说明模型对消光系数的拟合是完美的，图像为一条直线。
                       (3) TiO2（二氧化钛）
                       折射率（n）：
                       MSE = 0.00017, RMSE = 0.013, MAE = 0.010, R² = 0.980
                       R² 较高，说明模型对折射率的拟合效果较好，R²= 0.980符合要求。
                       RMSE 和 MAE 的值比SiO2略大，但整体误差仍然较小，图像简单。
                       消光系数（k）：
                       MSE = 0.0, RMSE = 0.0, MAE = 0.0, R² = 1.0
                       所有误差指标均为0，R²为1，说明模型对消光系数的拟合是完美的，图像简单。
                要求：
                 1.以json的格式输出你的数据,例如上例你就需要返回{'Ag':{'n':{'analysis':你的分析,拟合效果评价，图像整体形状 类型为str,'evaluation':False 类型为bool},'k':{'analysis':你的分析,拟合效果评价，图像整体形状 类型为str,'evaluation':Ture 类型为bool}},
                 'TiO2':{'n':{'analysis':你的分析,拟合效果评价，图像整体形状 类型为str,'evaluation':Ture 类型为bool},'k':{'analysis':你的分析,拟合效果评价，图像整体形状 类型为str,'evaluation':True 类型为bool}},
                 'SiO2':{'n':{'analysis':你的分析,拟合效果评价，图像整体形状 类型为str,'evaluation':False 类型为bool},'k':{'analysis':你的分析,拟合效果评价，图像整体形状 类型为str,'evaluation':Ture 类型为bool}}}
                 其中anlaysis的值为你对该材料的n或者k的拟合效果的分析，evaluaion为你最后的判断拟合的好为True,拟合的不好为False
                 2.上传的依据一般以图片的分析为主要判断（重点看与原数据的贴合程度，有较高要求），图中散点为原数据点，线为拟合数据，以拟合误差指标为辅助，需要看每个材料的n或k的整条曲线的拟合效果，客观的做出判断，拟合完美才行，R²需要>=0.98。
                 3.以中文回复用户
                 4.如果拟合曲线(红色拟合曲线)是一次的线性曲线是不可行的，直接在'evaluation'为False
                 5.除了json格式的数据以外不要有任何其他的返回内容
               """
        base64_image = self.encode_image(image_path=img_path)
        zdict = {'fit_num': 1, 'data': zipped}
        fit_messages = [
                        {
                            "role": "system",
                            "content": [{"type":"text","text": system}]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                },
                                {"type": "text", "text":f'拟合误差指标{zdict},结合图片进行评价'},
                            ],
                        }]
        qw = await self.client.chat.completions.create(
            model=qw_model_name,
            messages=fit_messages,
            response_format={"type": "json_object"}
        )
        json_content = qw.choices[0].message.content

        fit_messages.append({'role': 'assistant', 'content': json_content})
        # 获取 token 使用情况
        total_tokens = qw.usage.total_tokens if hasattr(qw, 'usage') else None
        json_data = self.content_to_json(json_content)
        print(json_data)
        return json_data,total_tokens

    async def fit_Agent(self,qw_model_name: str,qw_model_vl_name: str,CM: CalculatedMaterials):
        system = """
                你是一个数据拟合的助手，用户会提供拟合的参考数据给你，根据这些数据和意见，调用用户写好的拟合方法和调整拟合参数，来拟合数据以达到用户要求
                开始阶段(第一次输出阶段)：用户输入：开始拟合。你拟合输入的初始参数是number_polyfit:list[int]=[3]（列表目前只有一个参数），它是拟合的必要参数可以控制拟合的效果,
                            选择的拟合方法是method="interpolite_composites",在result中说明你的选参数原因（第一次默认为‘默认值’）。
                反馈阶段（两次或者以上拟合）：用户一般会输入拟合次数，需要重新拟合材料为[材料1，材料2，...]列表形式，根据你选用的拟合方法和参数用户建议和评价（json形式的数据）
                用户的评价键为材料，材料的值为n(折射率)，k(消光系数)，后面的值为"analysis"用户的建议，"evaluation"用户的评价，一般评价为True，就是好，重点影响你的是评价不好的，
                你要更加评价不好的材料n或者k的"analysis"来选择合适的拟合方法和拟合系数
                例如Q:开始第3次拟合，需要重新拟合材料为[SiO2,Au]列表形式，根据你选用的拟合方法和参数,用户拟合后建议和评价为
                    {
                    "Au": {
                        "n": {
                            "analysis": "Au的折射率（n）拟合曲线整体呈现非线性趋势，与原数据点的贴合程度较好，但R²值为0.9597，略低于0.98的标准，且RMSE和MAE值相对较大，表明预测值与真实值之间存在一定的误差。",
                            "evaluation": False
                        },
                        "k": {
                            "analysis": "Au的消光系数（k）拟合曲线与原数据点的贴合程度非常好，R²值高达0.9986，接近1，RMSE和MAE值较小，表明模型对消光系数的拟合效果非常理想。",
                            "evaluation": True
                        }
                    },
                    "SiO2": {
                        "n": {
                            "analysis": "SiO2的折射率（n）拟合曲线整体呈现非线性下降趋势，与原数据点的贴合程度较好，R²值为0.9725，略低于0.98的标准，但RMSE和MAE值较小，表明预测误差较小。",
                            "evaluation": False
                        },
                        "k": {
                            "analysis": "SiO2的消光系数（k）拟合曲线与原数据点完全重合，所有误差指标均为0，R²值为1，表明模型对消光系数的拟合是完美的。",
                            "evaluation": True
                        }
                    }}
                    A:{'number_polyfit':[3]','method':'plot_composites','result':'第二次拟合的结果看来（因为这是第3次拟合，上一次为第2次），用户对Au的折射率n的评价（evaluation）为False,根据解释（analysis）R²值为0.9597，略低于0.98的标准。用户对SiO2的折射率n的评价（evaluation）为False,根据解释（analysis）R²值为0.9725，略低于0.98的标准,更换拟合方法plot_composites拟合结束为3'}
                要求：
                 1. 第一次拟合阶段：以json的格式输出你的数据,{'number_polyfit':[3]','method':'interpolite_composites','result':'默认值'}
                 2.可以使用的拟合方法有["interpolite_composites","plot_composites"],你可以根据拟合结果的反馈选择合适的方法,。
                 3.你只需要在method填入方法的名字即可，你也可以使用动态调整number_polyfit的值,来达到不同的拟合效果，
                 4.result写明你选择参数的原因，
                 5. 多次反馈拟合阶段(后续阶段)，以json的格式输出你的数据,例如:{'number_polyfit':[根据反馈调整参数,类型为int]','method':'根据反馈调整参数,类型为str','result':'返回你更改的原因,类型为str'}
            
                """
        fit_messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'开始拟合'},
        ]
        nt = 1
        total_tokens = 0
        sub_composites = []
        while True:
            qw = await self.client.chat.completions.create(
                model=qw_model_name,
                messages=fit_messages,
                response_format={"type": "json_object"}
            )
            json_content = qw.choices[0].message.content
            print(json_content)
            fit_messages.append({'role': 'assistant', 'content': json_content})
            # 获取 token 使用情况
            json_data1 = self.content_to_json(json_content)
            img_path,zipped=CM.calculate_fit_data(number_polyfit=[json_data1['number_polyfit']],
                                                  method=json_data1['method'],
                                                  sub_composites=sub_composites)
            # 拟合评价
            json_data,total_tokens1 = await self.fit_evaluation(qw_model_name=qw_model_name,img_path=img_path,zipped=zipped)
            total_tokens = total_tokens + total_tokens1 + qw.usage.total_tokens if hasattr(qw, 'usage') else None
            sub_composites=[]
            evaluation={}
            for i in json_data:
                for j in json_data[i]:
                    if not json_data[i][j]['evaluation']:
                        sub_composites.append(i)
            sub_composites=list(set(sub_composites))
            for i in sub_composites:
                evaluation[i]=json_data[i]
            if len(sub_composites)==0:
                break
            else:
                nt+=1
                fit_messages.append({'role': 'user',
                                     'content': f'开始第{nt}次拟合，需要重新拟合材料为{sub_composites}，根据你选用的拟合方法和参数，用户拟合建议和评价为{evaluation}'})


    async def analysis_model(self, qw_model_name: str,
                             qw_model_format_name: str,
                             qw_model_fit_name: str,
                             qw_model_evaluation_name: str,
                             full_content: str,
                             deepmind: bool = False):
        system = """
你是一个材料工程师，负责开发二维纳米复合薄膜材料，你会根据用户的要求构建一个材料的初始模型（该初始模型有层状材料组成成分，
每个组成成分的厚度）,你会分析初始模型的构成机理，并预测在具体波段的透过率，吸收率，反射率等特性。
要求，
1.厚度是一个范围值，用户会进行进化差分运算 出最优解，你需要给出每个材料的厚度范围可以用[最小厚度，最大厚度]表示
2.使用的厚度和波长单位都是纳米
3.需要有初始模型设计（材料选择，厚度范围），结构构成机理分析，光学预测，总结等方面
4.每层材料不要给多种可能的材料，仅给你预测的一种材料，便于用户进行计算。
示例：
Q:帮我预测一个可见光高透过，红外高反射和吸收的复合薄膜结构
A:为了设计一个在可见光波段（400-700 nm）具有高透过率，而在红外波段（700-2500 nm）具有高反射率和吸收率的复合薄膜结构，我们可以采用以下策略：
    初始模型设计
    材料选择：
    可见光高透过层：使用透明的氧化物材料，如二氧化硅（SiO₂），其在可见光波段具有高透过率。
    红外反射层：使用金属材料，如银（Ag）或铝（Al），这些金属对红外波段有较高的反射率。
    红外吸收层：使用半导体材料，如氧化铟锡（ITO）或氮化钛（TiN），这些材料在红外波段具有一定的吸收能力。
    厚度范围：
    SiO₂层：[50, 200] nm （用于可见光高透过）
    Ag层：[10, 50] nm （用于红外反射）
    ITO层：[20, 100] nm （用于红外吸收）
    结构构成机理分析
    可见光高透过：SiO₂层的主要功能是确保可见光波段的高透过率。由于其低折射率和低吸收系数，SiO₂能够在不显著衰减光强度的情况下让可见光通过。
    红外高反射：Ag层能够有效地反射红外光。这是因为金属在红外波段通常表现出高反射率，且Ag的光学性能特别适合用于反射红外辐射。
    红外高吸收：ITO层可以吸收部分红外光，从而减少红外光的透过。ITO作为一种透明导电氧化物，在红外波段具有一定的吸收特性，这有助于增强整体结构的红外吸收能力。
    光学特性预测
    透过率：
    在可见光波段（400-700 nm），主要由SiO₂层决定，透过率预计可达到80%-95%。
    在红外波段（700-2500 nm），透过率会显著降低，主要是因为Ag层的高反射和ITO层的吸收作用。
    反射率：
    在可见光波段，反射率较低，大部分光透过。
    在红外波段，Ag层将导致反射率显著增加，预计可达80%-95%。
    吸收率：
    在可见光波段，吸收率较低，主要由SiO₂层的低吸收特性决定。
    在红外波段，吸收率会因ITO层的存在而增加，预计可达到10%-30%。
    总结
    该复合薄膜结构通过合理选择材料及其厚度范围，可以在可见光波段实现高透过率，同时在红外波段实现高反射率和吸收率。用户可以通过进一步的进化差分运算优化各层的具体厚度，以获得最佳性能。
        """
        log = ""
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'构建的初始材料模型需要满足{full_content}'},
        ]
        turn = 1
        if deepmind:
            while True:
                reasoning_content = ""  # 定义完整思考过程
                answer_content = ""  # 定义完整回复
                completion = await self.client.chat.completions.create(
                    model=qw_model_name,  # 此处以 qwq-32b 为例，可按需更换模型名称
                    messages=messages,
                    # QwQ 模型仅支持流式输出方式调用
                    stream=True,
                    # 解除以下注释会在最后一个chunk返回Token使用量
                    stream_options={
                        "include_usage": True
                    }
                )
                async for chunk in completion:
                    if not chunk.choices:
                        log = log + '\nUsage:' + str(chunk.usage) + '\n'
                    else:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                            # hasattr(delta, reasoning_content) 就是判断delta对象中是否有名为reasoning_content的属性。
                            # 如果有则返回True，没有返回False。
                            reasoning_content += delta.reasoning_content
                        else:
                            answer_content += str(delta.content) if delta.content is not None else ""
                messages.append({"role": "assistant", "content": answer_content})
                log = log + "\n" + "=" * 20 + f"第{turn}思考过程" + "=" * 20 + "\n" + reasoning_content + "\n" + "=" * 20 + f"第{turn}回复" + "=" * 20 + "\n" + answer_content

                # 开始计算，获取json数据
                json_content, total_tokens = self.communication_format(qw_model_name=qw_model_format_name,
                                                                       full_content=answer_content)
                json_content1 = self.content_to_json(json_content)
                set_thickness = [(i[0] + i[1]) / 2 for i in json_content1['set_thickness']]

                # 创建材料模型对象
                CM = CalculatedMaterials(build_composites=['build_composites'],
                                         set_thickness=set_thickness,
                                         wl=json_content1['wl'])




                img1_path = f'../content/image_temp/fit_image/{solution_fit}'



                # 开始拟合
                # fit_message = self.fit_format(qw_model_name=qw_model_fit_name,
                #                               CM=CM)
                fit_massage = self.fit_evaluation(qw_model_name=qw_model_fit_name, img_path=img1_path, zipped=zipped)


                log = log + "\n" + f"第{turn}拟合材料" + "\n" + f'{fit_massage}' + '\n'

                # 光学性能计算
                optimal_thickness, R, T, A, img_url = CM.calculate_tmm_DE(json_content1['set_thickness'])

                # 判断材料模型是否满足预期
                json_content2, total_tokens2 = self.evaluation_model(qw_model_name=qw_model_evaluation_name,
                                                                     full_content=full_content, path=img_url)
                evaluation_json = self.content_to_json(json_content2)
                log = log + '\n' + f"第{turn}评估" + '\n' + f'{evaluation_json}' + '\n'
                if evaluation_json['evaluation'] or turn == 5:
                    break
                else:
                    file_object = await self.client.files.create(file=Path(img_url), purpose="file-extract")
                    messages.append({'role': 'user', 'content': f'fileid://{file_object.id}'})
                    advice = evaluation_json['advice']
                    messages.append(
                        {'role': 'user', 'content': f'{advice},根据以上建议和计算的图像结果重新预测材料基础模型'})
                    turn += 1
                print(log, end='', flush=True)
        else:
            while True:
                answer_content = ""  # 定义完整回复
                completion = await self.client.chat.completions.create(
                    model=qw_model_name,  # 此处以 qwq-32b 为例，可按需更换模型名称
                    messages=messages
                )
                assistant_content = completion.choices[0].message.content
                log = log + "\n" + "=" * 20 + f"第{turn}完整回复" + "=" * 20 + "\n" + assistant_content
                turn += 1
                messages.append({"role": "assistant", "content": assistant_content})
                # 开始计算，获取json数据
                json_content, total_tokens = self.communication_format(qw_model_name=qw_model_format_name,
                                                                       full_content=answer_content)
                json_content1 = self.content_to_json(json_content)
                set_thickness = [(i[0] + i[1]) / 2 for i in json_content1['set_thickness']]

                # 创建材料模型对象
                CM = CalculatedMaterials(build_composites=['build_composites'],
                                         set_thickness=set_thickness,
                                         wl=json_content1['wl'])

                # 开始拟合
                fit_message = self.fit_format(qw_model_name=qw_model_fit_name,
                                              CM=CM)
                log = log + "\n" + f"第{turn}拟合材料" + "\n" + f'{fit_message}' + '\n'

                # 光学性能计算
                optimal_thickness, R, T, A, img_url = CM.calculate_tmm_DE(json_content1['set_thickness'])

                # 判断材料模型是否满足预期
                json_content2, total_tokens2 = self.evaluation_model(qw_model_name=qw_model_evaluation_name,
                                                                     full_content=full_content, path=img_url)
                evaluation_json = self.content_to_json(json_content2)
                log = log + '\n' + f"第{turn}评估" + '\n' + f'{evaluation_json}' + '\n'
                if not evaluation_json['evaluation'] and turn != 5:
                    file_object = await self.client.files.create(file=Path(img_url), purpose="file-extract")
                    messages.append({'role': 'user', 'content': f'fileid://{file_object.id}'})
                    advice = evaluation_json['advice']
                    messages.append(
                        {'role': 'user', 'content': f'{advice},根据以上建议和计算的图像结果重新预测材料基础模型'})
                    turn += 1
                    print(log, end='', flush=True)
                else:
                    break

    async def evaluation_model(self, qw_model_name: str, full_content: str, path: str):
        system = """
        你是一个复合薄膜材料工程师助手，现在用户会构建一个初始的复合材料模型，对这个材料的光学性能有一定的性能期望，他会通过计算
        得到材料的光学性能参数在不同波长下的透过率，吸收率和反射率。你需要根据用户提供的计算数据判断，是否符合用户对材料的光学性能
        期望，如果基本满足条件你就在'evaluation'键的值填写True,不满足就填False。你可以在'advice'键的值中写你的建议,尽量详细点
        如果不好给出修改建议。以json的格式输出内容。输出形式为{'evaluation'：类型：bool,'advice'：你的建议，类型：str}
        """
        file_object = await self.client.files.create(file=Path(path), purpose="file-extract")
        completion = await self.client.chat.completions.create(
            model=qw_model_name,
            messages=[
                {'role': 'system', 'content': f'{system}'},
                {'role': 'user', 'content': f'fileid://{file_object.id}'},
                {'role': 'user', 'content': f'用户预期为：{full_content}'},
            ],
            response_format={"type": "json_object"}
        )
        # 获取生成的 JSON 内容
        json_content = completion.choices[0].message.content
        # 获取 token 使用情况
        total_tokens = completion.usage.total_tokens if hasattr(completion, 'usage') else None

        return json_content, total_tokens
