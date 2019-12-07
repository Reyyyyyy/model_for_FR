import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import json
import random
from sklearn.utils import shuffle

#模型精度：0.488→0.77→0.99

#使用GPU
#tf.device('/gpu:1')需要配置GPU环境

#超参数
lr = 0.0001
batch_size = 24
epochs = 3
view_1 = 5
view_2 = 5
view_3 = 5
#view_4 = 3
#view_5 = 3
num_filter_1 = 32
num_filter_2 = 64
num_filter_3 = 64
#num_filter_4 = 64
#num_filter_5 = 64
fc_neuron_num_1 = 1024
#fc_neuron_num_out = 512
#------
use_bn = True
use_dropout = True
dropout = 0.5#每个元素被保留的概率
keep_prob = tf.placeholder(tf.float32)

def get_train_batch():
    x_train = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\train_labels.npy',allow_pickle=True)
    return x_train,y_train

def get_test_batch():
    x_test = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_batches.npy',allow_pickle=True)
    y_test = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_labels.npy',allow_pickle=True)
    return x_test,y_test
    
def conv2d(x,w,b,use_bn,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    x = tf.layers.batch_normalization(x,training=use_bn)

    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,k,k,1],padding='SAME')
    return x

def conv_net(x,weights,biases,use_bn,use_dropout):
    #convs = []
    #activations = []
    #x = tf.reshape(x,[-1,32,32,3])
    
    conv_1 = conv2d(x,weights['wc1'],biases['bc1'],use_bn)
    #convs.append(conv_1)
    #activations.append(conv_1)
    conv_1 = maxpool2d(conv_1)
    #lrn1 = tf.nn.lrn(conv_1,4,bias=1,alpha=0.001/9.0, beta=0.75)
    
    conv_2 = conv2d(conv_1,weights['wc2'],biases['bc2'],use_bn)
    #convs.append(conv_2)
    #activations.append(conv_2)
    conv_3 = conv2d(conv_2,weights['wc3'],biases['bc3'],use_bn)
    #convs.append(conv_3)
    #activations.append(conv_3)
    conv_3 = maxpool2d(conv_3)
    #lrn2 = tf.nn.lrn(conv_2,4,bias=1,alpha=0.001/9.0, beta=0.75)
    
    #conv_3 = conv2d(conv_2,weights['wc3'],biases['bc3'])
    '''
    conv_4 = conv2d(conv_3,weights['wc4'],biases['bc4'])
    conv_4 = maxpool2d(conv_4)

    conv_5 = conv2d(conv_4,weights['wc5'],biases['bc5'])
    conv_5 = maxpool2d(conv_5)
    ''' 
    flatten = tf.reshape(conv_3,[-1,56*56*num_filter_3])

    fc1 = tf.nn.relu(tf.matmul(flatten,weights['wf1'])+biases['bf1'])
    fc1= tf.layers.batch_normalization(fc1)
    #activations.append(fc1)
    if use_dropout:
        fc1 = tf.nn.dropout(fc1,keep_prob)
    '''
    fc2 = tf.nn.relu(tf.matmul(fc1,weights['wf2']) + biases['bf2'])
    if use_dropout:
        fc2 = tf.nn.dropout(fc2,keep_prob)
    '''
    out = tf.matmul(fc1,weights['out']) + biases['out']

    return out

weights={'wc1':tf.Variable(tf.random.truncated_normal([view_1,view_1,3,num_filter_1],stddev=0.02)),
         'wc2':tf.Variable(tf.random.truncated_normal([view_2,view_2,num_filter_1,num_filter_2],stddev=0.02)/np.sqrt(num_filter_1/2)),
         'wc3':tf.Variable(tf.random.truncated_normal([view_3,view_3,num_filter_2,num_filter_3],stddev=0.02)/np.sqrt(num_filter_2/2)),
         #'wc4':tf.Variable(tf.random.truncated_normal([view_4,view_4,num_filter_3,num_filter_4],stddev=0.05)/np.sqrt(num_filter_3/2)),
         #'wc5':tf.Variable(tf.random.truncated_normal([view_5,view_5,num_filter_4,num_filter_5],stddev=0.05)/np.sqrt(num_filter_4/2)),
         'wf1':tf.Variable(tf.random.truncated_normal([56*56*num_filter_3,fc_neuron_num_1],stddev=0.04)/np.sqrt(num_filter_3/2)),
         #'wf2':tf.Variable(tf.random.truncated_normal([fc_neuron_num_1,fc_neuron_num_out],stddev=0.04)/np.sqrt(fc_neuron_num_1/2)),
         'out':tf.Variable(tf.random.truncated_normal([fc_neuron_num_1,2],stddev=1/192)/np.sqrt(192/2))
         }

biases={'bc1':tf.Variable(tf.zeros([num_filter_1])),
        'bc2':tf.Variable(tf.zeros([num_filter_2])+0.1),
        'bc3':tf.Variable(tf.zeros([num_filter_3])),
        #'bc4':tf.Variable(tf.zeros([num_filter_4])),
        #'bc5':tf.Variable(tf.zeros([num_filter_5])),
        'bf1':tf.Variable(tf.zeros([fc_neuron_num_1])+0.1),
        #'bf2':tf.Variable(tf.zeros([fc_neuron_num_out])+0.1),
        'out':tf.Variable(tf.zeros([2])),
        } 
x = tf.placeholder(tf.float32,[None,224,224,3])
y = tf.placeholder(tf.float32,[None,2])

pred = conv_net(x,weights,biases,use_bn,use_dropout)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred+h,labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1)),dtype=tf.float32))
init = tf.global_variables_initializer()

train_x,train_y = get_train_batch()
test_x,test_y = get_test_batch()

if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        
        #Train
        for step in range(int(epochs*1920/batch_size)):
            pointer = step*batch_size%1920
            x_batch = train_x[pointer:pointer+batch_size]
            y_batch = train_y[pointer:pointer+batch_size]
            
            sess.run(optimizer,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            loss = sess.run(cost,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            acc = sess.run(accuracy,feed_dict={x:x_batch,y:y_batch,keep_prob:1.0})
            print('loss:',loss)
            print('accuracy:',acc,'\n')
            #print(sess.run(pred,feed_dict={x:x_batch,y:y_batch,keep_prob:1.0}))

        #Evaluate
        use_dropout = False
        use_bn = False
        avg_acc_test = 0
        
        for i in range(int(test_x.shape[0]/5)):
            acc_test = sess.run(accuracy,feed_dict={x:test_x[i*5:(i+1)*5],y:test_y[i*5:(i+1)*5],keep_prob:dropout})
            print('Test accuracy: ',acc_test)
            avg_acc_test += acc_test
            
        avg_acc_test = avg_acc_test/int(test_x.shape[0]/5)
        print('Done! Average accuracy of test data is: ',avg_acc_test)
        

        #保存模型变量，注意json不接受numpy的array,要变成list
        with open ('weights.json','w') as f:
            ws = {}
            for name,w in weights.items():
                ws[name] = sess.run(w).tolist()
            json.dump(ws,f)

        with open ('biases.json','w') as f:
            bs = {}
            for name,b in biases.items():
                bs[name] = sess.run(b).tolist()
            json.dump(bs,f)

