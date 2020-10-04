import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os

#超参数
neurons_1 = 128
neurons_2 = 256

#引入vgg16
vgg16 = (np.load('vgg16.npy',allow_pickle=True,encoding='bytes')).tolist()

def cut_img(img_array):
    if img_array.shape[0] ==1584 or img_array.shape[0] ==1581:
        img_array = img_array[196:1420,371:2612]
    if img_array.shape[0] ==2134 or img_array.shape[0] ==2145:
        img_array = img_array[240:1905,500:3524]
    return img_array

def get_weight(shape,a=0.001):
    w = tf.Variable(tf.random.truncated_normal(shape=shape,stddev=0.04),dtype=tf.float32)
    return w
    
def conv2d(x,w,b,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,strides=2):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,strides,strides,1],padding='SAME')

def VGG16(x,fine_tuning=False):
    
    conv1_1 = conv2d(x,vgg16[b'conv1_1'][0],vgg16[b'conv1_1'][1])
    conv1_2 = conv2d(conv1_1,vgg16[b'conv1_2'][0],vgg16[b'conv1_2'][1])
    mp1 = maxpool2d(conv1_2)
    conv2_1 = conv2d(mp1,vgg16[b'conv2_1'][0],vgg16[b'conv2_1'][1])
    conv2_2 = conv2d(conv2_1,vgg16[b'conv2_2'][0],vgg16[b'conv2_2'][1])
    mp2 = maxpool2d(conv2_2)
    conv3_1 = conv2d(mp2,vgg16[b'conv3_1'][0],vgg16[b'conv3_1'][1])
    conv3_2 = conv2d(conv3_1,vgg16[b'conv3_2'][0],vgg16[b'conv3_2'][1])
    conv3_3 = conv2d(conv3_2,vgg16[b'conv3_3'][0],vgg16[b'conv3_3'][1])
    mp3 = maxpool2d(conv3_3)
    conv4_1 = conv2d(mp3,vgg16[b'conv4_1'][0],vgg16[b'conv4_1'][1])
    conv4_2 = conv2d(conv4_1,vgg16[b'conv4_2'][0],vgg16[b'conv4_2'][1])
    conv4_3 = conv2d(conv4_2,vgg16[b'conv4_3'][0],vgg16[b'conv4_3'][1])
    mp4 = maxpool2d(conv4_3)
    if fine_tuning:
        conv5_1 = conv2d(mp4,weights['vgg_w1'],biases['vgg_b1'])
        conv5_2 = conv2d(conv5_1,weights['vgg_w2'],biases['vgg_b2'])
        conv5_3 = conv2d(conv5_2,weights['vgg_w3'],biases['vgg_b3'])     

    else:
        conv5_1 = conv2d(mp4,vgg16[b'conv5_1'][0],vgg16[b'conv5_1'][1])
        conv5_2 = conv2d(conv5_1,vgg16[b'conv5_2'][0],vgg16[b'conv5_2'][1])
        conv5_3 = conv2d(conv5_2,vgg16[b'conv5_3'][0],vgg16[b'conv5_3'][1])
        
    mp5 = maxpool2d(conv5_3)
    xed = mp5
    return xed
    
def Tail(x,weights,biases,fine_tuning=False):
    xed = VGG16(x,fine_tuning)
    flatten = tf.reshape(xed,[-1,7*7*512])
    
    fc1 = tf.nn.relu(tf.matmul(flatten,weights['wf1'])+biases['bf1'])
    fc2 = tf.nn.relu(tf.matmul(fc1,weights['wf2'])+biases['bf2'])
    
    out = tf.matmul(fc2,weights['out']) + biases['out']
    return out

weights={'vgg_w1':tf.Variable(vgg16[b'conv5_1'][0]),
         'vgg_w2':tf.Variable(vgg16[b'conv5_2'][0]),
         'vgg_w3':tf.Variable(vgg16[b'conv5_3'][0]),
         'wf1':get_weight([7*7*512,neurons_1]),\
         'wf2':get_weight([neurons_1,neurons_2])/np.sqrt((neurons_1)/2),\
         'out':get_weight([neurons_2,2])/np.sqrt((neurons_2)/2)}
         


biases={'vgg_b1':tf.Variable(vgg16[b'conv5_1'][1]),
        'vgg_b2':tf.Variable(vgg16[b'conv5_2'][1]),
        'vgg_b3':tf.Variable(vgg16[b'conv5_3'][1]),
        'bf1':tf.Variable(tf.zeros([neurons_1])),\
        'bf2':tf.Variable(tf.zeros([neurons_2])),\
        'out':tf.Variable(tf.zeros([2]))}

x = tf.placeholder(tf.float32,[None,224,224,3])

pred = Tail(x,weights,biases)
saver = tf.train.Saver(weights.update(biases))

if __name__ == '__main__':
    pointer = 0
    gap = 24
    sign='正'
    while(1):
        with tf.Session() as sess:
            saver.restore(sess,r'D:\核聚变课题组\use_transfer_learning_vgg16\my_model.ckpt')
            xs=[]
            #predict
            sign = random.choice(['正','负'])
            os.chdir(r'D:\核聚变课题组\{}样本'.format(sign))
            imgs = os.listdir(r'D:\核聚变课题组\{}样本'.format(sign))
            random.shuffle(imgs)
            try:
                for i,img_name in enumerate(imgs[pointer:pointer+24]):
                    img = Image.open(img_name)
                    img_array = cut_img(np.array(img))
                    resizing_img = Image.fromarray(img_array)
                    resized_img = resizing_img.resize((224,224),Image.ANTIALIAS)
                    img_array = np.array(resized_img)
                    xs.append(img_array)
                    if len(xs)==24:
                        break
                images = np.array(xs)
      
                print(sign+'({})'.format(gap),':images:',sess.run(tf.argmax(pred,axis=1),feed_dict={x:images}))
                print('')

                pointer += gap    
            except:
                print('Done!')
            

