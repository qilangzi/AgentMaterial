from openai import OpenAI
import json


class QwenModel:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        pass

    def material_model_prediction(self, qw_model_name:str,system:str, prompt:str):
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        completion = client.chat.completions.create(
            model=qw_model_name,  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt}
            ]
        )
        return (completion.choices[0].message.content)

    #阅读论文的模型
    def material_readpaper(self,paper:str,qw_model_name:str):
        # system = """
        #         你是一个论文实验总结助手，返回用户3个部分，**实验判断**，**原文摘录**，**总结日志**。
        #         **实验判断**是分析提供的文本有没有实验部分，仅返回Y/N。
        #         **原文摘录**是将你认为是实验部分的内容返回给用户，不要改动原文，直接摘录该部分，
        #         **总结日志**是你的实验的总结。
        #         一般的实验部分标签是Experimental Section，你可以根据它的位置进行定位。
        #
        #         要求：
        #         1.	使用json格式返回数据。3个部分作为键，返回内容为值
        #         2.	严格遵守用户要求，不要返回json以外的任何内容，便于用户后期调用
        #         3.	如果实验判断为N, 原文摘录，总结日志都为
        #         4.  除了摘录原文外，其他成分用中文回答
        #         """
        system ="""
        你是一个论文实验总结助手，总结该论文中的实验原理，实验材料，实验方法，实验步骤，实验结论。
                要求：
                1.	使用json格式返回数据。5个部分作为键，返回内容为值
                2.	严格遵守用户要求，不要返回json以外的任何内容，便于用户后期调用
        """
        prompt = f"""
                # 原文：{paper}
                """
        try:
            odj=self.material_model_prediction(qw_model_name,system,prompt)
            positions1 = odj.find('{')
            positions2 = odj.find('}')
            od = odj[positions1:positions2 + 1]
            xd = json.loads(od)
            return xd
        except Exception as e:
            print(e)

    def text_embeding(self,embedding_name:str,text:str):
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        embedding = client.embeddings.create(
            model=embedding_name,
            input=text
        )
        return embedding.data[0].embedding

