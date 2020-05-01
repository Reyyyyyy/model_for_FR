import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#模型训练时长为80秒左右

def get_train_batch():
    x_train = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\train_labels.npy',allow_pickle=True)

    return x_train,y_train

def decolor(imgs):
    tmp_batch = np.zeros((imgs.shape[0],1,224,224))
    for index,each in enumerate(imgs):
        img = Image.fromarray(each)
        gray_img = img.convert('L')
        img_array = np.array(gray_img).reshape(1,224,224)
        tmp_batch[index] = img_array
    return tmp_batch.reshape(imgs.shape[0],224,224)

#获取数据
x_train,y_train = get_train_batch()
#将x和y降维
x_train = decolor(x_train)
tmp = np.zeros((2304,224*224))
for i in range(x_train.shape[0]):
    tmp[i] = x_train[i].flatten()
x_train = tmp
tmp = np.zeros((y_train.shape[0]))
for i in range(2304):
    tmp[i] = y_train[i].argmax()
y_train = tmp
#定义一个弱分类器
weak_clf = DecisionTreeClassifier()
#定义AdaBoost分类器
clf = AdaBoostClassifier(n_estimators=200, base_estimator=weak_clf, learning_rate=0.01)
#计时
time_start = time.time()
#训练
clf.fit(x_train, y_train)
#结束计时
time_stop = time.time()
#评估
y_pred = clf.predict(x_train)
accuracy = sum(y_pred==y_train)/y_pred.shape[0]
print('训练结束，用时:'+str(int(time_stop-time_start))+'秒。')
print('训练集的准确率:',accuracy)
#保存模型
model = pickle.dumps(clf) #dumps针对bytes
with open('model','wb') as f:
    f.write(model)









