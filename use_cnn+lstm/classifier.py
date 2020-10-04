import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os

#有时模型数据太多，保存时程序会卡崩，导致ckpt文件因此损坏，只能重新训练模型再保存

#超参数
units = 256
view_1 = 5
view_2 = 3
num_filter_1 = 36
num_filter_2 = 1
fc_neuron_num = 768

def cut_img(img_array):
    if img_array.shape[0] ==1584 or img_array.shape[0] ==1581:
        img_array = img_array[196:1420,371:2612]
    if img_array.shape[0] ==2134 or img_array.shape[0] ==2145:
        img_array = img_array[240:1905,500:3524]
    return img_array

def lstm_cell(units):
    cell = rnn.LSTMCell(units)#activation默认为tanh
    return cell

def conv2d(x,w,b,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)

    return tf.nn.relu(x)

def maxpool2d(x,strides=2):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,strides,strides,1],padding='SAME')

def net(x,weights,biases):
    conv_1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv_1 = maxpool2d(conv_1)
    conv_2 = conv2d(conv_1,weights['wc2'],biases['bc2'])
    conv_2 = maxpool2d(conv_2)
    
    flatten = tf.unstack(tf.reshape(conv_2,[-1,56,56]),56,1)

    lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell(units) for _ in range(2)])
    outputs,_ = rnn.static_rnn(lstm_layers,flatten,dtype="float32")

    fc = tf.matmul(outputs[-1],weights['wf']) + biases['bf']
    res = tf.matmul(fc,weights['out']) + biases['out']

    return res
    
weights={'wc1':tf.Variable(tf.random.truncated_normal([view_1,view_1,3,num_filter_1],stddev=0.02)),
         'wc2':tf.Variable(tf.random.truncated_normal([view_2,view_2,num_filter_1,num_filter_2],stddev=0.02)/np.sqrt(num_filter_1/2)),
         'wf':tf.Variable(tf.random.truncated_normal([units,fc_neuron_num],stddev=0.04)),
         'out':tf.Variable(tf.random.truncated_normal([fc_neuron_num,2],stddev=1/192)/np.sqrt(fc_neuron_num/2))
         }

biases={'bc1':tf.Variable(tf.zeros([num_filter_1])),
        'bc2':tf.Variable(tf.zeros([num_filter_2])),
        'bf':tf.Variable(tf.zeros([fc_neuron_num])),
        'out':tf.Variable(tf.zeros([2])),
        }    

x = tf.placeholder(tf.float32,[None,224,224,3])
pred = net(x,weights,biases)
init = tf.global_variables_initializer()
saver = tf.train.Saver(weights.update(biases))

if __name__ == '__main__':
    pointer = 0
    gap = 24
    sign='正'
    while(1):
        with tf.Session() as sess:
            saver.restore(sess,r'D:\核聚变课题组\use_cnn+lstm\my_model.ckpt')
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
                
