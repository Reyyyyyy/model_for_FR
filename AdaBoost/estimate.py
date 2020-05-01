import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import pickle

def get_test_batch():
    x_test = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_batches.npy',allow_pickle=True)
    y_test = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_labels.npy',allow_pickle=True)

    return x_test,y_test

def decolor(imgs):
    tmp_batch = np.zeros((imgs.shape[0],1,224,224))
    for index,each in enumerate(imgs):
        img = Image.fromarray(each)
        gray_img = img.convert('L')
        img_array = np.array(gray_img).reshape(1,224,224)
        tmp_batch[index] = img_array
    return tmp_batch.reshape(imgs.shape[0],224,224)

#获取数据
x_test,y_test = get_test_batch()
#载入模型
with open('model','rb') as f:
    model = pickle.loads(f.read())
#将x和y降维
x_test = decolor(x_test)
tmp = np.zeros((798,224*224))
for i in range(x_test.shape[0]):
    tmp[i] = x_test[i].flatten()
x_test = tmp

tmp = np.zeros((y_test.shape[0]))
for i in range(798):
    tmp[i] = y_test[i].argmax()
y_test = tmp
#预测
res = model.predict(x_test)
#评估
y_pred = res
y_true = y_test

TP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
FP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
TN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
FN = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))

accuracy = (TP + TN)/(TN + TP + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F_score = (2*precision*recall)/(precision+recall)
N_score = TN / (TN + FP)

print('Accuracy:',accuracy)
print('Precison:',precision)
print('Recall:',recall)
print('F-Score:',F_score)
print('N-Score:',N_score)











