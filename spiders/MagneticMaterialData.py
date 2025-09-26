import requests
from lxml import etree
url='https://www.cryst.ehu.es/magndata/index.php?index=0.1'
headers={
"User-Agent": 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)'
                }
response = requests.get(url,headers=headers)
c=response.status_code
# print(c)
data=response.text
# print(data)

single_tree = etree.HTML(data)
mcif_value = single_tree.xpath("//form/input[@name='mcif']/@value")
mcif = mcif_value[1]

action_url='https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-magndata'
form_data = {
    "mcif": mcif,          # 提取的mcif值
    "choose": "Get_mirreps",  # 提交按钮值
    "mode": "irreps"       # 新增的mode参数
}
post_response = requests.post(action_url, data=form_data,headers=headers,cookies=response.cookies)
post_response.encoding = "utf-8"
# print("POST请求响应状态码：", post_response.status_code)
print("POST请求响应内容：", post_response.text)
ptext=post_response.text
single_tree = etree.HTML(ptext)
data_value = single_tree.xpath('//meta[@http-equiv="REFRESH"]/@content')[0][6:]
print(data_value)


