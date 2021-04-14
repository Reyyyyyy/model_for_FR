import os
import xlwt

#获取数据路径
p_imgs = os.listdir(r'C:\Users\tensorflow\Desktop\核聚变课题组\正样本')
n_imgs = os.listdir(r'C:\Users\tensorflow\Desktop\核聚变课题组\负样本')

#创建一个workbook并设置编码
workbook = xlwt.Workbook(encoding = 'utf-8')
#创建一个worksheet
worksheet = workbook.add_sheet('My worksheet')
#写入数据
#参数对应:行, 列, 值
for idx,p in enumerate(p_imgs):
    if '无' in p:
        name = p.split('无')[0][0:-1]
    else:
        name = p.split('.')[0]
    worksheet.write(idx,0,label=name)
    worksheet.write(idx,1,label='1')
for idx,n in enumerate(n_imgs):
    name = n.split('.')[0]
    worksheet.write(len(p_imgs)+idx,0,label=name)
    worksheet.write(len(p_imgs)+idx,1,label='0')
#保存
workbook.save('Data.xls')
