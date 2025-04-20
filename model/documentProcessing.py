import math
import PyPDF2
import os


class documentProcessing:
    def __init__(self):
        # self.path: str = path
        self.__inputpath: str = r"/content/document/input"
        self.__outputpath: str = r"/content/document/output"
        pass

    def ext_from_pdf(self, path: str):
        """
        读取pdf，返回str
        """
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
        return text

    def text_from_pdf_peg(self, path, ped=1):
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            mylen = range(math.ceil(len(reader.pages) / ped))

            text = {}
            for page_num in mylen:
                page = reader.pages[page_num * 2:page_num * 2 + ped]

                page1 = ['' + i.extract_text().replace('\n', '') or "" for i in page]
                text[f'第{page_num}部分'] = page1
        return text

    def batches_text_pdf(self):
        """
        批量读取pdf文件，以json格式返回
        :return: {文件名：论文内容}
        """
        result_datas = {}
        for i in os.listdir(self.__inputpath):
            input_path = os.path.join(self.__inputpath, i)
            output_path = os.path.join(self.__outputpath, i)
            ext_text = self.ext_from_pdf(input_path)
            result_datas[i] = ext_text
            os.rename(input_path, output_path)
        return result_datas

