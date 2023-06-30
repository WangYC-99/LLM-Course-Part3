import jsonlines
with open("/zhangpai25/wyc/data-main/模型预测结果/10b_dialog_new.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        print(item)

import xlrd
excel_path = "./data.xlsx"
#打开文件，获取excel文件的workbook（工作簿）对象
excel = xlrd.open_workbook(excel_path,encoding_override="utf-8")

#获取sheet对象
sheet = excel.sheets()[0]
sheet_row_mount = sheet.nrows
sheet_col_mount = sheet.ncols
print("row number: {0} ; col number: {1}".format(sheet_row_mount, sheet_col_mount))

# 按照行为单位输出
for i in range(1, sheet_row_mount):
    print(sheet.row_values(i))

#按照单元格为单位输出
for x in range(1, sheet_row_mount):
    y = 0
    while y < sheet_col_mount:
        print(sheet.cell_value(x,y), end = "")
        print(" ", end = "")
        y += 1
    print(" ")

for x in range(1, sheet_row_mount):
    y = 0
    while y < sheet_col_mount:
        sheet.cell_value(x,y)
        print(" ", end = "")
        y += 1
    print(" ")