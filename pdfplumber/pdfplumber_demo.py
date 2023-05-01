import pdfplumber
import pandas as pd
import os
import glob

def save_pdf_to_excel(path):
    #     print('文件名为：',path.split('/')[-1].split('.')[0] + '.xlsx')
    pdf = pdfplumber.open(path)
    pages = pdf.pages
    if len(pages) > 1:
        tables = []
        for each in pages:
            table = each.extract_table()
            tables.extend(table)
    else:
        tables = each.extract_table()
    data = pd.DataFrame(tables[1:], columns=tables[0])

    file_name = path.split('/')[-1].split('.')[0] + '.xlsx'
    data.to_excel("/Volumes/ExtremeSSD/git/demo/pdfplumber/excel_file/{}".format(file_name), index=False)
    return '保存成功！'

# main函数入口
if __name__ == "__main__":
    # 提取单个pdf文件
    pdf = pdfplumber.open("pdfplumber/pdf_file/新手爸妈攻略③-生产全攻略.pdf")
    pages = pdf.pages
    if len(pages) > 1:
        tables = []
        for each in pages:
            table = each.extract_table()
            tables.extend(table)
    else:
        tables = pages.extract_table()
    data = pd.DataFrame(tables[1:], columns=tables[0])
    # print(data)
    data.to_excel("pdfplumber/excel_file/新手爸妈攻略③-生产全攻略.xlsx", index=False)

    # 批量提取多个pdf文件
    path = r'pdfplumber/pdf_file'
    for f in glob.glob(os.path.join(path, "*.pdf")):
        res = save_pdf_to_excel(f)
        print(res)


