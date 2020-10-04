import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from PIL import Image
import os
import random

def cut_img(img_array):
    if img_array.shape[0] ==1584 or img_array.shape[0] ==1581:
        img_array = img_array[196:1420,371:2612]
    if img_array.shape[0] ==2134 or img_array.shape[0] ==2145:
        img_array = img_array[240:1905,500:3524]
        
    resizing_img = Image.fromarray(img_array)
    resized_img = resizing_img.resize((224,224),Image.ANTIALIAS)
    img = np.array(resized_img)
    return img

def decolor(imgs):
    tmp_batch = np.zeros((imgs.shape[0],1,224,224))
    for index,each in enumerate(imgs):
        img = Image.fromarray(each)
        gray_img = img.convert('L')
        img_array = np.array(gray_img).reshape(1,224,224)
        tmp_batch[index] = img_array
    return tmp_batch.reshape(imgs.shape[0],224,224)

def predict(img):
    categories = ['正','负']
    pred = model.predict_classes(img)#pred是一个列表，形式类似于[1,1,1,0,0,0]
    name = categories[pred[0]]
    return name

#载入模型
model = model_from_json(open('model.json').read())
model.load_weights('weights')

#载入数据
os.chdir(r'D:\核聚变课题组\正样本')
imgs = os.listdir(r'D:\核聚变课题组\正样本')
#打乱数据
random.shuffle(imgs)

#处理数据
xs=[]
for i,img_name in enumerate(imgs):
    img = Image.open(img_name)
    img_array = cut_img(np.array(img))
    xs.append(img_array)
    if len(xs)==30:
        break
images = decolor(np.array(xs))

#预测
for idx,img in enumerate(images):
    name = predict(img.reshape(1,224,224))
    print('你输入的第{}张图片是:  '.format(idx+1),name)
