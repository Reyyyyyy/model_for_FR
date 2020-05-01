from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
#负样本都是(2134, 4000, 3)
#正样本有的是(2134, 4000, 3),有的是(1584, 3000, 3)

train_batches = []
train_labels  = []

test_batches = []
test_labels  = []

def get_imgs(name):
    os.chdir('C:\\Users\\tensorflow\\Desktop\\核聚变课题组\\{PN}样本'.format(PN=name))
    imgs = os.listdir(r'C:\Users\tensorflow\Desktop\核聚变课题组\{PN}样本'.format(PN=name))
    return imgs

def cut_img(img_array):
    if img_array.shape[0] ==1584 or img_array.shape[0] ==1581:
        img_array = img_array[196:1420,371:2612]
    if img_array.shape[0] ==2134 or img_array.shape[0] ==2145:
        img_array = img_array[240:1905,500:3524]
    return img_array
    
p_imgs = get_imgs('正')
for img_name in p_imgs:
    img = Image.open(img_name)
    img_array = cut_img(np.array(img))
    resizing_img = Image.fromarray(img_array)
    resized_img = resizing_img.resize((224,224),Image.ANTIALIAS)
    img_array = np.array(resized_img)
    if len(train_batches) < 1152:
        train_batches.append(img_array)
        train_labels.append(np.array([1,0]))
        print('Train&Positive: {num}'.format(num=str(len(train_batches))))
    else:
        test_batches.append(img_array)
        test_labels.append(np.array([1,0]))
        print('Test&Positive: {num}'.format(num=str(len(test_batches))))

n_imgs = get_imgs('负')
for img_name in n_imgs:
    img = Image.open(img_name)
    img_array = cut_img(np.array(img))
    resizing_img = Image.fromarray(img_array)
    resized_img = resizing_img.resize((224,224),Image.ANTIALIAS)
    img_array = np.array(resized_img)
    if len(train_batches)-1152 < 1152:
        train_batches.append(img_array)
        train_labels.append(np.array([0,1]))
        print('Train&Negative: {num}'.format(num=str(len(train_batches)-1152)))
    else:
        test_batches.append(img_array)
        test_labels.append(np.array([0,1]))
        print('Test&Negative: {num}'.format(num=str(len(test_batches)-551)))

#打乱数据
train = shuffle(train_batches,train_labels)
test  = shuffle(test_batches,test_labels)

#把dataset以npy格式保存
os.chdir(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset')
np.save('train_batches.npy',train[0])
np.save('train_labels.npy',train[1])
np.save('test_batches.npy',test[0])
np.save('test_labels.npy',test[1])





