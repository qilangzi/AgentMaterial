import math
import PyPDF2


class documentProcessing:
    def __init__(self):
        # self.path: str = path
        self.__inputpath: str = r"/content/document/input"
        self.__outputpath: str = r"/content/document/output"
        pass
    def ext_from_pdf(self, path: str):
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
        return text

    def text_from_pdf_peg(self,path,ped=1):
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            mylen=range(math.ceil(len(reader.pages)/ped))
            # print(mylen)
            text={}
            for page_num in mylen:
                page = reader.pages[page_num*2:page_num*2+ped]
                # print(len(page))
                page1=[''+i.extract_text().replace('\n','') or "" for i in page]
                text[f'第{page_num}部分']=page1
        return text

