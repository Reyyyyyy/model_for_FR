import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers as reg
from keras import optimizers
from keras.models import Sequential
from keras.utils import plot_model

#超参数
units_1 = 128  #LSTM内部神经元个数
units_2 = 128
units_3 = 256
units_4 = 256
dp_1 = 0.1 #丢弃的比例
dp_2 = 0.1
dp_3 = 0.2
dp_4 = 0.2
a_1 = 0.001#正则化惩罚系数
a_2 = 0.001
a_3 = 0.002
a_4 = 0.002
time_steps = 224  #输入序列的总长度
input_dim = 224 #序列中各个元素的维度
epochs = 10
batch_size  = 96
lr = 0.001

def get_train_batch():
    x_train = np.load(r'D:\核聚变课题组数据\干净数据\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'D:\核聚变课题组数据\干净数据\train_labels.npy',allow_pickle=True)
    return x_train,y_train

def get_test_batch():
    x_test = np.load(r'D:\核聚变课题组数据\干净数据\test_batches.npy',allow_pickle=True)
    y_test = np.load(r'D:\核聚变课题组数据\干净数据\test_labels.npy',allow_pickle=True)
    return x_test,y_test

def decolor(imgs):
    tmp_batch = np.zeros((imgs.shape[0],1,224,224))
    for index,each in enumerate(imgs):
        img = Image.fromarray(each)
        gray_img = img.convert('L')#＞1的都置1，反之置0
        img_array = np.array(gray_img).reshape(1,224,224)
        tmp_batch[index] = img_array
    return tmp_batch.reshape(imgs.shape[0],224,224)/255
    
x_train,y_train = get_train_batch()
x_test,y_test = get_test_batch()

x_train = decolor(x_train)
x_test = decolor(x_test)

#定义优化器
adam = optimizers.Adam(learning_rate=lr)

#建筑模型
model = Sequential()
model.add(LSTM(units=units_1, input_shape=(time_steps,input_dim),kernel_regularizer=reg.l2(a_1),return_sequences=True))
model.add(Dropout(dp_1))
model.add(LSTM(units=units_2,kernel_regularizer=reg.l2(a_2),return_sequences=True))
model.add(Dropout(dp_2))
model.add(LSTM(units=units_3,kernel_regularizer=reg.l2(a_3),return_sequences=True))
model.add(Dropout(dp_3))
model.add(LSTM(units=units_4,kernel_regularizer=reg.l2(a_4)))
model.add(Dropout(dp_4))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

#可视化模型
plot_model(model, to_file='model.png')

#训练模型
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#保存模型
model.save_weights("weights")
with open("model.json", "w") as f:
    f.write(model.to_json())
    
#评估模型
loss, accuracy = model.evaluate(x_test,y_test)
y_pred = model.predict_classes(x_test)
y_true = np.argmax(y_test,axis=1)

TP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
FP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
TN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
FN = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F_score = (2*precision*recall)/(precision+recall)
N_score = TN / (TN + FP)

print('\n')        
print('loss: ', loss)
print('Accuracy: ', accuracy)
print('Precison:',precision)
print('Recall:',recall)
print('F-Score:',F_score)
print('N-score:',N_score)

input()


