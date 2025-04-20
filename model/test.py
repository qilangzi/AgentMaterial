import qwenModel
import asyncio
import os
async def main2():
    qw=qwenModel.QwenModel(api_key="sk-568bd13551dd42ae9c623bd04504ba02",base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    data_1={}
    data_2={}
    data_3={}
    data_4={}
    for i in os.listdir("../content/document/input"):
            path=os.path.join("../content/document/input",i)
            content,total_tokens1=await qw.pdf_reader(path=path,qw_model_name="qwen-long")
            data_1[i]={'data':content,'takes':total_tokens1}
            print(f'json数据为:\n{content}\n消耗的token为:{total_tokens1}\n\n\n\n\n')
            xj, takens = await qw.formatContent(full_content=content, qw_model_name="qwen-plus")
            print(f'json数据为:\n{xj}\n消耗的token为:{takens}\n\n\n\n\n')
            jsonList = qw.content_to_json(full_content=xj)
            data_2[i] = {'data': jsonList, 'takes': takens}
            jsonList.pop('论文DIO')
            jsonList.pop('关键字')
            text_embeding=await qw.textList_embeding(embedding_name='text-embedding-v3', jsonlist=jsonList)
            print(f'json数据为:\n{text_embeding}\n\n\n\n\n\n')
            data_3[i] = text_embeding
            data_4[i] = jsonList.update(text_embeding)
    await qw.client.close()
    # print(f"{data_1}\n\n\n")
    # print(f"{data_2}\n\n\n")
    # print(f"{data_3}\n\n\n")
    # print(f"{data_4}\n\n\n")
    return data_1,data_2,data_3,data_4

if __name__ == '__main__':
    asyncio.run(main2())