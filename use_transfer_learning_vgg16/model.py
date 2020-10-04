import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#引入vgg16
vgg16 = (np.load('vgg16.npy',allow_pickle=True,encoding='bytes')).tolist()

#超参数
neurons_1 = 128
neurons_2 = 256
a_1 = 0.002  #正则化惩罚系数
a_2 = 0.002
a_3 = 0.0001
lr=0.001
batch_size=96
epochs=5
kp1=0.9
kp2=0.9
keep_prob_1=tf.placeholder(tf.float32)
keep_prob_2=tf.placeholder(tf.float32)

def get_train_batch():
    x_train = np.load(r'D:\核聚变课题组数据\干净数据\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'D:\核聚变课题组数据\干净数据\train_labels.npy',allow_pickle=True)
    return x_train,y_train

def get_test_batch():
    x_test = np.load(r'D:\核聚变课题组数据\干净数据\test_batches.npy',allow_pickle=True)
    y_test = np.load(r'D:\核聚变课题组数据\干净数据\test_labels.npy',allow_pickle=True)
    return x_test,y_test

def get_weight(shape,a=0.001):
    w = tf.Variable(tf.random.truncated_normal(shape=shape,stddev=0.04),dtype=tf.float32)
    tf.add_to_collection('L2_loss',tf.contrib.layers.l2_regularizer(a)(w))
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
    fc1 = tf.nn.dropout(fc1,keep_prob_1)
    
    fc2 = tf.nn.relu(tf.matmul(fc1,weights['wf2'])+biases['bf2'])
    fc2 = tf.nn.dropout(fc2,keep_prob_2)
    
    out = tf.matmul(fc2,weights['out']) + biases['out']
    return out

weights={'vgg_w1':tf.Variable(vgg16[b'conv5_1'][0]),
         'vgg_w2':tf.Variable(vgg16[b'conv5_2'][0]),
         'vgg_w3':tf.Variable(vgg16[b'conv5_3'][0]),
         'wf1':get_weight([7*7*512,neurons_1],a_1),\
         'wf2':get_weight([neurons_1,neurons_2],a_2)/np.sqrt((neurons_1)/2),\
         'out':get_weight([neurons_2,2],a_3)/np.sqrt((neurons_2)/2)}
         


biases={'vgg_b1':tf.Variable(vgg16[b'conv5_1'][1]),
        'vgg_b2':tf.Variable(vgg16[b'conv5_2'][1]),
        'vgg_b3':tf.Variable(vgg16[b'conv5_3'][1]),
        'bf1':tf.Variable(tf.zeros([neurons_1])),\
        'bf2':tf.Variable(tf.zeros([neurons_2])),\
        'out':tf.Variable(tf.zeros([2]))}

x = tf.placeholder(tf.float32,[None,224,224,3])
y = tf.placeholder(tf.float32,[None,2])

pred = Tail(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
cost += tf.add_n(tf.get_collection('L2_loss'))#加入正则化项
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1)),dtype=tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver(weights.update(biases))

train_x,train_y = get_train_batch()
test_x,test_y = get_test_batch()

if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        n_data = train_x.shape[0]
        #Train
        for step in range(int(epochs*n_data/batch_size)):
            pointer = step*batch_size%n_data
            x_batch = train_x[pointer:pointer+batch_size]
            y_batch = train_y[pointer:pointer+batch_size]

            if x_batch.shape[0] == 0:
                continue
            
            sess.run(optimizer,feed_dict={x:x_batch,y:y_batch,keep_prob_1:kp1,keep_prob_2:kp2})
            loss = sess.run(cost,feed_dict={x:x_batch,y:y_batch,keep_prob_1:1.0,keep_prob_2:1.0})
            acc = sess.run(accuracy,feed_dict={x:x_batch,y:y_batch,keep_prob_1:1.0,keep_prob_2:1.0})
            print('loss:',loss)
            print('accuracy:',acc,'\n')
            
        #Evaluate
        test_batch = 9
        avg_acc_test = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(int(test_x.shape[0]/test_batch)):
            y_pred = sess.run(tf.argmax(pred,axis=1),feed_dict={x:test_x[i*test_batch:(i+1)*test_batch],keep_prob_1:1.0,keep_prob_2:1.0})
            y_true = np.argmax(test_y[i*test_batch:(i+1)*test_batch],axis=1)

            TP += np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
            FP += np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
            TN += np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
            FN += np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
            
            acc_test = sess.run(accuracy,feed_dict={x:test_x[i*test_batch:(i+1)*test_batch],y:test_y[i*test_batch:(i+1)*test_batch],keep_prob_1:1.0,keep_prob_2:1.0})
            print('Test accuracy: ',acc_test)
            avg_acc_test += acc_test
            
        avg_acc_test = avg_acc_test/int(test_x.shape[0]/test_batch)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_score = (2*precision*recall)/(precision+recall)
        N_score = TN / (TN + FP)
        print('Done!')
        print('Accuracy:',avg_acc_test)
        print('Precison:',precision)
        print('Recall:',recall)
        print('F-Score:',F_score)
        print('N-Score:',N_score)
        #Save
        saver.save(sess,'my_model.ckpt')

