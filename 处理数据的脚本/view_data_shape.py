import os
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

imgs = os.listdir(r'D:\核聚变课题组\样本图片\正样本')
os.chdir(r'D:\核聚变课题组\样本图片\正样本')

def get_shapes(imgs):
    shapes = []
    for img in imgs:
        if img.shape not in shapes:
            shapes.append(img.shape)
    return shapes

def cut_img(img_array):
    if img_array.shape[0] ==1584 or img_array.shape[0] ==1581:
        img_array = img_array[196:1420,371:2612]
    if img_array.shape[0] ==2134 or img_array.shape[0] ==2145:
        img_array = img_array[240:1905,500:3524]
    return img_array

#classify deep blue and big blue
def is_Deep(img_array):
    #DEEP
    if int(np.mean(img_array[:,:,0]))*int(np.mean(img_array[:,:,1])) <= np.mean(img_array[:,:,2]):
        return True
    #BIG
    elif np.mean(img_array[:,-100:,0])*np.mean(img_array[:,-100:,1]) <= np.mean(img_array[:,-100:,2]):
        return False
    else:
        return None
#second cut
def cut_blue(big_blue,stride=20):
    for i in range(30):
        if np.mean(big_blue[:,(big_blue.shape[1]-50)-i*stride:,:]) >= 50:
            break
    cut_line = (big_blue.shape[1]-50)-i*stride
            
    return big_blue[:,:cut_line,:]

for img in imgs:
    img = Image.open(img)
    img_array = np.array(img)
        
    print('裁剪前:',img_array.shape)
    plt.subplot(1,2,1)
    plt.imshow(img_array)
    plt.title('裁剪前')
    #plt.axis('off')

    #first cut
    img_array = cut_img(np.array(img))
    #cut blue
    res = is_Deep(img_array)
    if res == True:
        print('Deep Blue!')
    else:
        print('Big Blue!')
        img_array = cut_blue(img_array)    
    print('裁剪后:',img_array.shape)
    plt.subplot(1,2,2)
    plt.imshow(img_array)
    plt.title('裁剪后')
    #plt.axis('off')
    
    plt.show()
    print('\n')
#正样本尺寸分布：(1584,3000,3), (2134,4000,3), (2145,4000,3) ,(1581,3000,3)
#负样本尺寸分布：(1584,3000,3), (2134,4000,3)
        
#(1584,3000,3)和(1581,3000,3)裁剪尺寸( 196~1420 ,371~ 2612)
#(2134,4000,3)和(2145,4000,3)裁剪尺寸( 240~1905 ,500~ 3524)


