from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
#正样本共张，负样本共张
#train_batches(P:N = 1200:1200)  test_batches(P:N = 209:364)

#定义样本数量
train_pos = 1200
train_neg = 1200
test_pos  = 209
test_neg  = 364
#定义列表容器
train_batches = []
train_labels  = []
test_batches = []
test_labels  = []

def get_imgs(name):
    os.chdir(r'D:\核聚变课题组\样本图片\{PN}样本'.format(PN=name))
    imgs = os.listdir(r'D:\核聚变课题组\样本图片\{PN}样本'.format(PN=name))
    return imgs

def cut_img(img_array):
    #基于对数组的切片算法
    if img_array.shape[0] ==1584 or img_array.shape[0] ==1581:
        img_array = img_array[196:1420,371:2612]
    if img_array.shape[0] ==2134 or img_array.shape[0] ==2145:
        img_array = img_array[240:1905,500:3524]
    return img_array

def is_Deep(img_array):
    #DEEP
    if int(np.mean(img_array[:,:,0]))*int(np.mean(img_array[:,:,1])) <= np.mean(img_array[:,:,2]):
        return True
    #BIG
    elif np.mean(img_array[:,-100:,0])*np.mean(img_array[:,-100:,1]) <= np.mean(img_array[:,-100:,2]):
        return False
    #NORMAL
    else:
        return None

def cut_blue(big_blue,stride=10):
    for i in range(300):
        cut_line = (big_blue.shape[1]-50)-i*stride
        box = big_blue[(big_blue.shape[0]//5):(big_blue.shape[0]//5+100),cut_line:,:]
        if np.mean(big_blue[(big_blue.shape[0]//5):(big_blue.shape[0]//5+100),cut_line,:]) >= 55:
            break      
    return big_blue[:,:cut_line,:]

p_imgs = get_imgs('正')
for img_name in p_imgs:
    img = Image.open(img_name)
    img_array = cut_img(np.array(img))
    #first cut
    img_array = cut_img(np.array(img))
    #cut blue
    res = is_Deep(img_array)
    if res == True:
        #print('Deep Blue!')
        pass
    else:
        #print('Big Blue!')
        img_array = cut_blue(img_array)    
    resizing_img = Image.fromarray(img_array)
    resized_img = resizing_img.resize((224,224),Image.ANTIALIAS)
    img_array = np.array(resized_img)
    if len(train_batches) < train_pos:
        train_batches.append(img_array)
        train_labels.append(np.array([1,0]))
        print('Train-Positive: {num}'.format(num=str(len(train_batches))))
    else:
        test_batches.append(img_array)
        test_labels.append(np.array([1,0]))
        print('Test-Positive: {num}'.format(num=str(len(test_batches))))

n_imgs = get_imgs('负')
for img_name in n_imgs:
    img = Image.open(img_name)
    img_array = cut_img(np.array(img))
    #first cut
    img_array = cut_img(np.array(img))
    #cut blue
    res = is_Deep(img_array)
    if res == True:
        #print('Deep Blue!')
        pass
    else:
        #print('Big Blue!')
        img_array = cut_blue(img_array)    
    resizing_img = Image.fromarray(img_array)
    resized_img = resizing_img.resize((224,224),Image.ANTIALIAS)
    img_array = np.array(resized_img)
    if len(train_batches)-train_pos < train_neg:
        train_batches.append(img_array)
        train_labels.append(np.array([0,1]))
        print('Train-Negative: {num}'.format(num=str(len(train_batches)-train_pos)))
    else:
        test_batches.append(img_array)
        test_labels.append(np.array([0,1]))
        print('Test-Negative: {num}'.format(num=str(len(test_batches)-test_pos)))

#打乱数据
train = shuffle(train_batches,train_labels)
test  = shuffle(test_batches,test_labels)

#把dataset以npy格式保存
os.chdir(r'D:\核聚变课题组\dateset')
np.save('train_batches.npy',train[0])
np.save('train_labels.npy',train[1])
np.save('test_batches.npy',test[0])
np.save('test_labels.npy',test[1])





