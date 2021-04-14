import pandas as pd
import shutil
import os
import numpy as np

#总共有51个数据需要更正
#有47个正样本需要改成负样本
#有4个负样本需要改成正样本
fix = []

with open(r'更改.xlsx','rb') as f:
    data = pd.read_excel(f,index_col=0)

for idx,each in enumerate(np.array(data)):
    if each[1] == 0 or each[1] == 1:
        if each[0] != each[1]:
            if each[0] == 1:
                print(idx+2,' ','正样本->负样本',' '+data.index[idx])
                fix.append(('正样本','负样本',data.index[idx]))
            else:
                print(idx+2,' ','负样本->正样本',' '+data.index[idx])
                fix.append(('负样本','正样本',data.index[idx]))

path = r'C:\Users\tensorflow\Desktop\核聚变课题组'
for each in fix:
    try:
        name = each[2]+'.jpg'
        shutil.move(path+'\\'+each[0]+'\\'+name , path+'\\'+each[1])
    except:
        continue
