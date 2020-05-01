from tensorflow.python.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#获取数据
x_train = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\train_batches.npy',allow_pickle=True)
x_test = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_batches.npy',allow_pickle=True)

#归一化数据,不然binary_crossentropy的损失值会为负
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#定义输入
input_img = Input(shape=(224,224,3))

#---编码---
#Conv1
x = Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(input_img)
x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
#Conv2
x = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
#Conv3
x = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(x)
encoded =  MaxPooling2D(pool_size=(2,2),padding='same')(x)

#---解码---
#Deconv1
x = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(encoded)
x = UpSampling2D(size=(2,2))(x)
#Deconv2
x = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(x)
x = UpSampling2D(size=(2,2))(x)
#Deconv3
x = Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(x)
x = UpSampling2D(size=(2,2))(x)
decoded = Conv2D(filters=3,kernel_size=(3,3),activation='sigmoid',padding='same')(x)

#声明并编译模型
autoencoder = Model(inputs=input_img,outputs=decoded)
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

#训练模型
autoencoder.fit(x_train,x_train,epochs=50,batch_size=48,shuffle=True,validation_data=(x_test,x_test))

#保存模型
autoencoder.save_weights("weights")
with open("autoencoder.json", "w") as f:
    f.write(autoencoder.to_json())








