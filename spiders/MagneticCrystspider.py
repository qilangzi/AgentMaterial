import json

from bs4 import BeautifulSoup
import requests
from lxml import etree
import pandas as pd
import re
from tqdm import tqdm
import base64
from io import BytesIO
from database.connection import get_db
from database.modles import *
from sqlalchemy import and_
from sqlalchemy import func
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
logging.basicConfig(
    level=logging.INFO,  # 记录INFO及以上级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
    filename='spiders/project_log.log',  # 保存的日志文件名
    filemode='a'  # 追加模式，保留历史日志
)

class MagneticCrystSider:
    def __init__(self):
        self.url = 'https://www.cryst.ehu.es/magndata/search.php?show_db=1'
        self.headers = {
            "User-Agent": 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)'
        }
        self.magnetic_elements = ['la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb',
                                  'lu']
        self.resout_data = {}
        self.all_data = []
        self.useful_elements = []
        # self.d_b=get_db()
        # self.elements_id: any
        # self.elements_name: any
        # self.parent_space_group_url: any
        # self.propagation_vector: any
        # self.transition_temperature: any
        # self.experimental_data: any
        # self.lattice_parameters_of_the_magnetic_unit_cell: any
        # self.bns_magnetic_space_group: any
        self.false_elements = []




    def process_with_multithread(self, put_db):
        # 1. 线程安全的数据库会话（避免多线程共享同一个会话）
        # 注意：SQLAlchemy 会话不是线程安全的，每个线程需单独创建
        def get_thread_safe_db():
            return next(get_db())  # 假设self.get_db()是数据库会话生成器
        def process_item(item):
            db = None
            try:
                db = get_thread_safe_db()
                existing = db.query(MagneticCryst).filter(
                    MagneticCryst.material_id == item['elements_id']
                ).first()
                if not existing:
                    # 调用主处理函数（数据库交互）
                    self.touch_data(item, put_db=put_db, db=db)
                    return f"成功处理: {item['elements_id']}"
                else:
                    return f"已存在: {item['elements_id']}---{item['elements_name']}"
            except Exception as e:
                if db:
                    db.rollback()
                return f"处理失败 {item['elements_id']}: {str(e)}"
            finally:
                if db:
                    db.close()

        max_threads = min(5, len(self.useful_elements))
        results = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # 提交所有任务
            futures = {executor.submit(process_item, item): item for item in self.useful_elements}

            # 跟踪进度并收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc='多线程处理中'):
                results.append(future.result())
        # for res in results:
        #     print(res)

    def delete_sub(self, html_content, tags_to_remove: list):
        """
        删除HTML标签中的<sub>标签
        """
        for tag in tags_to_remove:
            # 构建正则表达式，匹配开标签和闭标签
            pattern = re.compile(f'<\/?{tag}>', re.IGNORECASE)
            html_content = pattern.sub('', html_content)

        return html_content

    def web_1(self):
        response = requests.get(self.url, headers=self.headers)
        html_data = response.text.replace('<sub>', '').replace('</sub>', '')
        tree = etree.HTML(html_data)
        nu1 = tree.xpath(r"//a[contains(@href, 'index.php?index=') and @class='blue']")
        nuu = [[nu.xpath(r"./@href"), nu.xpath("./text()[1]")] for nu in nu1]
        for i in nuu:
            self.resout_data['elements_id'] = i[0][0].split('index=')[1]
            self.resout_data['elements_url'] = i[0]
            self.resout_data['elements_name'] = i[1][0]
            self.all_data.append(self.resout_data)
            self.resout_data = {}


    def startspider(self, put_db=False):
        self.web_1()
        # 筛选有用元素（保持原有逻辑）
        added_indexes = set()  # 记录已添加的index
        useful_elements = []

        for item in self.all_data:
            compound = item['elements_name'].lower()
            if any(n in compound for n in self.magnetic_elements):
                # 用index作为唯一标识去重
                if item['elements_url'][0] not in added_indexes:
                    self.useful_elements.append(item)
                    added_indexes.add(item['elements_url'][0])
        # for i in self.all_data:
        #     for j in self.magnetic_elements:
        #         low_n = i['elements_name'][0].lower()
        #         if j in low_n:
        #             self.useful_elements.append(i)

        self.process_with_multithread(put_db)

    def web_3(self, single_tree, response_1):

        form_action = single_tree.xpath("//form[@method='post']/@action")
        action_url = fr"https://www.cryst.ehu.es{form_action[0]}"
        mcif_value = single_tree.xpath("//form/input[@name='mcif']/@value")
        mcif = mcif_value[1]
        form_data = {
            "mcif": mcif,  # 提取的mcif值
            "choose": "Get_mirreps",  # 提交按钮值
            "mode": "irreps"  # 新增的mode参数
        }
        post_response = requests.post(action_url, data=form_data, headers=self.headers, cookies=response_1.cookies,timeout=60)
        post_response.encoding = "utf-8"
        ptext = post_response.text
        single_tree = etree.HTML(ptext)
        data_value = single_tree.xpath('//meta[@http-equiv="REFRESH"]/@content')[0][6:]
        return data_value

    def picture_download(self,image_url):
        data=requests.get(image_url,headers=self.headers)
        # 发送请求下载图片
        response = requests.get(image_url)
        response.raise_for_status()  # 检查请求是否成功

        # 将图片内容转换为base64
        image_bytes = response.content
        base64_str = base64.b64encode(image_bytes).decode('utf-8')

        return base64_str

    def save_base64_image(self,base64_str, output_path):
        """
        将base64编码字符串解码并保存为图片，使用with语句处理文件

        参数:
            base64_str: 图片的base64编码字符串
            output_path: 保存图片的路径

        返回:
            bool: 保存成功返回True，否则返回False
        """
        try:
            # 解码base64字符串
            image_bytes = base64.b64decode(base64_str)

            # 使用with语句安全处理文件写入
            with BytesIO(image_bytes) as img_buffer, open(output_path, 'wb') as f:
                # 从缓冲区读取并写入文件
                f.write(img_buffer.read())

            print(f"图片已成功保存至: {output_path}")
            logging.info(f"图片已成功保存至: {output_path}")
            return True

        except Exception as e :
            print(f"保存base64图片时出错: {e}")
            logging.error(f"保存base64图片时出错: {e}")
            return False

    def search_name(self,single_tree: etree):
        # 假设single_tree是lxml解析的对象，先获取所有符合条件的form标签
        form_tags = single_tree.xpath("//td/form[@target='_blank']")

        # 取第一个form标签并转换为字符串
        form_str = etree.tostring(form_tags[0], encoding='utf-8', method='html').decode('utf-8')

        # 使用BeautifulSoup解析
        soup = BeautifulSoup(form_str, 'html.parser')

        # 删除所有sub和small标签
        for tag in soup.find_all(['sub', 'small']):
            tag.unwrap()  # 完全移除标签及其内容
        new_name = [h2.get_text() for h2 in soup.find_all('h2')]
        return new_name[0]

    def search_table(self,target_table_string: str):
        table = []
        # cleaned_target_table_string = delete_sub(target_table_string, ['sub','small'])
        cleaned_target_table = BeautifulSoup(target_table_string, 'html.parser')
        for tag in cleaned_target_table.find_all(['sub', 'font', 'b']):
            tag.unwrap()
        tr_table = cleaned_target_table.find_all('tr')
        headers1 = [th.text for th in tr_table[0].find_all('th')]
        table.append(headers1)
        data = []
        for row in tr_table[1:]:
            row_data = [td.text for td in row.find_all('td')]
            data.append(row_data)
        table.append(data)
        return table

    def index_incomm_set(self,html_data_1: str):
        single_tree = etree.HTML(html_data_1)
        new_url = single_tree.xpath("//head/meta[@http-equiv='REFRESH']/@content")
        if new_url:
            url = new_url[0].split('url=')[1]
            return url
        else:
            return False

    def seach_vectors(self,single_tree: etree):
        Propagation_vector = single_tree.xpath(
            "//b[contains(text(), 'Propagation vector:')]/following-sibling::text()[1]")
        if Propagation_vector:
            # print(Propagation_vector)
            return Propagation_vector
        else:
            Propagation_vector = single_tree.xpath(
                "//br[preceding::b[contains(text(), 'Propagation vector(s):')] and following::b[contains(text(), 'Transition Temperature:')]]")
            if Propagation_vector:
                target_brs = Propagation_vector[:-2]
                Propagation_vector = [br.xpath("./following-sibling::text()[1]") for br in target_brs]
            else:
                Propagation_vector = single_tree.xpath(
                    "//br[preceding::b[contains(text(), 'Propagation vector(s):')] and following::b[contains(text(), 'Experiment Temperature:')]]")
                target_brs = Propagation_vector[:-1]
                Propagation_vector = [br.xpath("./following-sibling::text()[1]") for br in target_brs]
                # print(Propagation_vector)
            return Propagation_vector
            # print(Propagation_vector)




    def touch_data(self, element,db,put_db=False):
        single_url = r'https://www.cryst.ehu.es/magndata/' + f'{element["elements_url"][0]}'
        response_1 = requests.get(single_url, headers=self.headers)
        html_data_1 = response_1.text
        # print(html_data_1)
        new_url = self.index_incomm_set(html_data_1)
        print(new_url)
        if new_url:
            new_url = r'https://www.cryst.ehu.es/magndata/' + f'{new_url}'
            response_1 = requests.get(new_url, headers=self.headers)
            html_data_1 = response_1.text
            single_url=new_url

        if put_db:
            try:
                dbMaterials=MagneticCryst(
                    material_id=element['elements_id'],
                    material_url=single_url,
                )
                db.add(dbMaterials)
                db.commit()
            except Exception as e:
                print(f"id={element['elements_id']}保存数据库失败:{e}")
                logging.error(f"id={element['elements_id']}保存数据库失败:{e}")
        db_ids = db.query(MagneticCryst).filter(MagneticCryst.material_id == element['elements_id']).first()
        single_tree = etree.HTML(html_data_1)
        new_name = self.search_name(single_tree)
        img_url_1 = single_tree.xpath("//td[@style='text-align:center;']/a/@href")
        # /../..：从 <i> 向上回溯两次，到达共同的父节点 <td>（第一次 .. 到 <br>，第二次 .. 到 <td>）。
        parent_space_group_url = single_tree.xpath(
            "//b[contains(text(), 'Parent space group')]/following::a[@class='blue'][1]/@href")
        Propagation_vector = self.seach_vectors(single_tree)
        Transition_Temperature = single_tree.xpath(
            "//b[contains(text(), 'Transition Temperature:')]/following-sibling::text()[1]")
        Experiment_Temperature = single_tree.xpath(
            "//b[contains(text(), 'Experiment Temperature:')]/following-sibling::text()[1]")
        Lattice_parameters_of_the_magnetic_unit_cell = self.seach_latticePParameters(single_tree)
        BNS_Magnetic_Space_Group = single_tree.xpath(
            "//b[contains(text(), 'BNS Magnetic Space Group:')]/following::a[@class='blue'][1]/@href")
        img_url_2=[]
        for i in img_url_1:
            imgUrl=fr'https://www.cryst.ehu.es/magndata/{i}'
            img_url_2.append(imgUrl)
            # img_base64 = self.picture_download(imgUrl)
            # img_url_2[imgUrl] = img_base64
        element['new_name'] = new_name
        element['img_url'] = {'url':img_url_2}
        element['parent_space_group_url'] = parent_space_group_url[0] if parent_space_group_url else None
        element['Propagation_vector'] = json.dumps(Propagation_vector)
        element['Transition_Temperature'] = Transition_Temperature[0] if Transition_Temperature else None
        element['Experiment_Temperature'] = Experiment_Temperature[0] if Experiment_Temperature else None
        element['Lattice_parameters_of_the_magnetic_unit_cell'] = Lattice_parameters_of_the_magnetic_unit_cell[0] if Lattice_parameters_of_the_magnetic_unit_cell else None
        element['BNS_Magnetic_Space_Group'] = BNS_Magnetic_Space_Group[0] if BNS_Magnetic_Space_Group else None
        if put_db:
            try:
                db_ids.material_name = element['new_name'],
                db_ids.materImageUrl=json.dumps(img_url_2),
                db_ids.parentSpaceGroupUrl=element['parent_space_group_url']
                db_ids.propagationvector = element['Propagation_vector']
                db_ids.transitionTemperature=element['Transition_Temperature']
                db_ids.experimentTemperature=element['Experiment_Temperature']
                db_ids.latticeParameters=element['Lattice_parameters_of_the_magnetic_unit_cell']
                db_ids.bnsMagneticSpaceGroup=element['BNS_Magnetic_Space_Group']
                db.commit()
            except Exception as e:
                print(f"id={element['elements_id']}-2保存数据库失败:{e}")
                logging.error(f"id={element['elements_id']}-2保存数据库失败:{e}")
        # Magnetic atoms
        target_table1 = single_tree.xpath(
            "//font[contains(text(), 'Magnetic atoms')]/following::table[@class='sample'][1]")
        if not target_table1:
            print(f"id={element['elements_id']}未找到# Magnetic atoms目标表格，请检查HTML结构或XPath")
            logging.error(f"id={element['elements_id']}未找到# Magnetic atoms目标表格，请检查HTML结构或XPath")
            element['magnetic_atoms'] = {}
        else:
            target_table_string = etree.tostring(target_table1[0], encoding='utf-8', method='html').decode('utf-8')
            table_1 = self.search_table(target_table_string)
            element['magnetic_atoms'] = {'headers1': table_1[0], 'data': table_1[1]}
            # cleaned_target_table_string = self.delete_sub(target_table_string, ['sub', 'small'])
            # target_table = etree.HTML(cleaned_target_table_string)
            # # 2. 提取表头（表格第一行的<th>）
            # headers1 = target_table.xpath(".//tr[1]/th/text()")
            # headers1[-2:] = ['|M|']
            # # 3. 提取数据行（表格tbody中除第一行外的<tr>）
            # rows = target_table.xpath(".//tr[position() > 1]")
            # # 4. 遍历行，提取单元格文本
            # data = []
            # for row in rows:
            #     cell_texts = row.xpath("./td/font/b/text()")
            #     m_tag_x = cell_texts[-7] + '_x' if cell_texts[-7] == 'm' else cell_texts[-7]
            #     m_tag_y = cell_texts[-6] + '_y' if cell_texts[-6] == ',m' else cell_texts[-6]
            #     m_tag_z = cell_texts[-5] + '_z' if cell_texts[-5] == ',m' else cell_texts[-5]
            #     m_tags = m_tag_x + m_tag_y + m_tag_z
            #     cell_texts[-7:-4] = [m_tags]
            #     data.append(cell_texts)
            # element['magnetic_atoms'] = {'headers1': headers1, 'data': data}


        magenticAtom1={}
        p1_tags = single_tree.xpath("//p/b/font[@color='red']/text()")
        if p1_tags:
            for i1 in p1_tags:
                if 'Set of atoms in the unit cell related by symmetry with the magnetic atom' in i1:
                    target_table1 = single_tree.xpath(
                        f"//font[contains(text(), '{i1}')]/following::table[@class='sample'][1]")
                    target_table_string = etree.tostring(target_table1[0], encoding='utf-8', method='html').decode('utf-8')
                    table_1 = self.search_table(target_table_string)
                    magenticAtom1[f'{i1}'] = {'headers1': table_1[0], 'data': table_1[1]}
                # cleaned_target_table_string = self.delete_sub(target_table_string, ['sub', 'small'])
                # target_table = etree.HTML(cleaned_target_table_string)
                # headers2 = target_table.xpath(".//tr[1]/th/text()")
                # rows2 = target_table.xpath(".//tr[position() > 1]")
                # data2 = []
                # for row in rows2:
                #     cell_texts = row.xpath("./td/text()")
                #     data2.append(cell_texts)
                # magenticAtom1[f'{i1}'] = {'headers2': headers2, 'data2': data2}
        else:
            print(f"id={element['elements_id']}未找到# magenticAtom1目标表格，请检查HTML结构或XPath")
            logging.error(f"id={element['elements_id']}未找到# magenticAtom1目标表格，请检查HTML结构或XPath")
        element['magnetic_atoms1'] = magenticAtom1


        # Non-magnetic atoms
        target_table = single_tree.xpath(
            "//b[contains(text(), 'Non-magnetic atoms')]/following::table[@class='sample'][1]")
        if not target_table:
            print(f"id={element['elements_id']}未找到Non-magnetic atoms目标表格，请检查HTML结构或XPath")
            logging.error(f"id={element['elements_id']}未找到Non-magnetic atoms目标表格，请检查HTML结构或XPath")
            element['non_magnetic_atoms']= {}
        else:
            target_table_string = etree.tostring(target_table1[0], encoding='utf-8', method='html').decode('utf-8')
            table_1 = self.search_table(target_table_string)
            element['non_magnetic_atoms'] = {'headers1': table_1[0], 'data': table_1[1]}
            # target_table = target_table[0]  # 获取表格节点
            # # 2. 提取表头（表格第一行的<th>）
            # headers1 = target_table.xpath(".//tr[1]/th/text()")
            # # 3. 提取数据行（表格tbody中除第一行外的<tr>）
            # rows = target_table.xpath(".//tr[position() > 1]")
            # # 4. 遍历行，提取单元格文本
            # data = []
            # for row in rows:
            #     cell_texts = row.xpath("./td/text()")
            #     data.append(cell_texts)
            # element['non_magnetic_atoms'] = {'headers1': headers1, 'data': data}

        nonMagneticAtom1={}
        p_tags = single_tree.xpath("//p/text()")
        if p_tags:
            for i_1 in p_tags:
                if 'Set of atoms in the unit cell related by symmetry with the atom' in i_1:
                    target_table = single_tree.xpath(f"//p[contains(text(), '{i_1}')]/following::table[@class='sample'][1]")
                    target_table_string = etree.tostring(target_table[0], encoding='utf-8', method='html').decode('utf-8')
                    table_1 = self.search_table(target_table_string)
                    nonMagneticAtom1[f'{i_1}'] = {'headers1': table_1[0], 'data': table_1[1]}
        else:
            print(f"id={element['elements_id']}未找到# nonMagneticAtom1目标表格，请检查HTML结构或XPath")
            logging.error(f"id={element['elements_id']}未找到# nonMagneticAtom1目标表格，请检查HTML结构或XPath")
                # target_table = target_table[0]  # 获取表格节点
                # # 2. 提取表头（表格第一行的<th>）
                # headers1 = target_table.xpath(".//tr[1]/th/text()")
                # # 3. 提取数据行（表格tbody中除第一行外的<tr>）
                # rows = target_table.xpath(".//tr[position() > 1]")
                # # 4. 遍历行，提取单元格文本
                # data = []
                # for row in rows:
                #     cell_texts = row.xpath("./td/text()")
                #     data.append(cell_texts)
                #
                # nonMagneticAtom1[f'{i_1}'] = {'headers1': headers1, 'data': data}
        element['non_magnetic_atoms1']=nonMagneticAtom1
        if put_db:
            try:
                # db=next(get_db())
                db_ids.tableMagneticAtoms=json.dumps(element['magnetic_atoms1']),
                db_ids.tableMagneticAtom = json.dumps(element['magnetic_atoms']),
                db_ids.tableNoMagneticAtom=json.dumps(element['non_magnetic_atoms']),
                db_ids.tableNoMagneticAtoms=json.dumps(element['non_magnetic_atoms1']),
                db.commit()
            except:
                print(f"id={element['elements_id']}保存数据库失败-3")
                logging.info(f"id={element['elements_id']}保存数据库失败-3")
        try:
            element['Get_mirreps'] = self.web_3(single_tree, response_1)
            if put_db:
                try:
                    db_ids.getmirrepsurl = element['Get_mirreps'],
                    db.commit()
                except:
                    print(f"id={element['elements_id'] }保存数据库失败-4")
                    logging.info(f"id={element['elements_id']}保存数据库失败-4")
        except Exception as e:
            print(f"id={element['elements_id']}Get_mirreps失败-{e}")
            logging.info(f"id={element['elements_id']}Get_mirreps失败-{e}")

    def repair_vectors(self):
        db=next(get_db())
        existing_list = db.query(MagneticCryst).filter(
            MagneticCryst.propagationvector == None
        ).all()
        if existing_list:
            for i in existing_list:
                print(i.material_id)
                single_url = i.material_url
                response_1 = requests.get(single_url, headers=self.headers)
                html_data_1 = response_1.text
                single_tree = etree.HTML(html_data_1)
                Propagation_vector = self.seach_vectors(single_tree)
                if Propagation_vector:
                    i.propagationvector = json.dumps(Propagation_vector)
                    db.commit()
                    print(f"id={i.material_id}保存数据库成功")
                else:
                    print(f"id={i.material_id}保存数据库失败")
        db.close()

    def repair_getmirrepsurl(self):
        db=next(get_db())
        existing_list = db.query(MagneticCryst).filter(
            MagneticCryst.getmirrepsurl == None
        ).all()
        if existing_list:
            for i in existing_list:
                print(i.material_id)
                single_url = i.material_url
                getmirrepsurl=None
                try:
                    response_1 = requests.get(single_url, headers=self.headers)
                    html_data_1 = response_1.text
                    single_tree = etree.HTML(html_data_1)
                    getmirrepsurl = self.web_3(single_tree, response_1)
                except Exception as e:
                    print(f"id={i.material_id}Get_mirreps失败-{e}")
                if getmirrepsurl:
                    i.getmirrepsurl = getmirrepsurl
                    db.commit()
                    print(f"id={i.material_id}保存数据库成功")
                else:
                    print(f"id={i.material_id}保存数据库失败")
        db.close()

    def repair_img(self):
        db=next(get_db())
        existing_list = db.query(MagneticCryst).filter(
            MagneticCryst.materImageUrl== '[]'
        ).all()
        if existing_list:
            for i in existing_list:
                print(i.material_id)
                single_url = i.material_url
                response_1 = requests.get(single_url, headers=self.headers)
                html_data_1 = response_1.text
                single_tree = etree.HTML(html_data_1)
                img_url = single_tree.xpath('//td[@style="text-align:center;"]/a/@href')
                if img_url:
                    img_url_2 = []
                    for i1 in img_url:
                        imgUrl = fr'https://www.cryst.ehu.es/magndata/{i1}'
                        img_url_2.append(imgUrl)
                    i.materImageUrl = json.dumps(img_url_2)
                    db.commit()
                    print(f"id={i.material_id}保存数据库成功")
                else:
                    print(f"id={i.material_id}保存数据库失败")
        db.close()

    def repair_table(self):
        db = next(get_db())
        # 检查这4个列是否全为None
        rows_to_delete = db.query(MagneticCryst).filter(
            MagneticCryst.tableMagneticAtoms == None,
            MagneticCryst.tableMagneticAtom == None,
            MagneticCryst.tableNoMagneticAtom == None,
            MagneticCryst.tableNoMagneticAtoms == None
        ).all()

        if rows_to_delete:
            for row in rows_to_delete:
                db.delete(row)
            db.commit()
            print(f"已删除 {len(rows_to_delete)} 行数据")
        else:
            print("没有符合删除条件的行")

    def seach_latticePParameters(self,single_tree):
        Lattice_parameters_of_the_magnetic_unit_cell = single_tree.xpath(
            "//b[contains(text(), 'Lattice parameters of the magnetic unit cell:')]/following-sibling::text()[1]")
        if Lattice_parameters_of_the_magnetic_unit_cell:
            LL=Lattice_parameters_of_the_magnetic_unit_cell
        else:
            Lattice_parameters_of_the_magnetic_unit_cell = single_tree.xpath(
                "//b[contains(text(), 'Lattice parameters of the basic unit cell:')]/following-sibling::text()[1]")
            LL = Lattice_parameters_of_the_magnetic_unit_cell
        return LL


    def repair_latticeParameters(self):
        db = next(get_db())
        existing_list = db.query(MagneticCryst).filter(
            MagneticCryst.latticeParameters == None
        ).all()
        if existing_list:
            for i in existing_list:
                print(i.material_id)
                single_url = i.material_url
                response_1 = requests.get(single_url, headers=self.headers)
                html_data_1 = response_1.text
                single_tree = etree.HTML(html_data_1)
                latticeParameters = self.seach_latticePParameters(single_tree)
                if latticeParameters:
                    i.latticeParameters = latticeParameters[0] if latticeParameters else None
                    db.commit()
                    print(f"id={i.material_id}保存数据库成功")
                else:
                    print(f"id={i.material_id}保存数据库失败")
        db.close()
