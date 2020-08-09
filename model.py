import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#超参数
units = 256
view_1 = 5
view_2 = 5
num_filter_1 = 24
num_filter_2 = 1
fc_neuron_num = 1024
learning_rate = 0.001
batch_size = 32
epochs = 5

def get_train_batch():
    x_train = np.load(r'D:\核聚变去噪数据\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'D:\核聚变去噪数据\train_labels.npy',allow_pickle=True)

    return x_train,y_train

def get_test_batch():
    x_test = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_batches.npy',allow_pickle=True)
    y_test = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_labels.npy',allow_pickle=True)

    return x_test,y_test

def decolor(imgs):
    tmp_batch = np.zeros((imgs.shape[0],1,224,224))
    for index,each in enumerate(imgs):
        img = Image.fromarray(each)
        gray_img = img.convert('L')#＞1的都置1，反之置0
        img_array = np.array(gray_img).reshape(1,224,224)
        tmp_batch[index] = img_array
    return tmp_batch.reshape(imgs.shape[0],224,224,1)/255

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
    
weights={'wc1':tf.Variable(tf.random.truncated_normal([view_1,view_1,1,num_filter_1],stddev=0.02)),
         'wc2':tf.Variable(tf.random.truncated_normal([view_2,view_2,num_filter_1,num_filter_2],stddev=0.02)/np.sqrt(num_filter_1/2)),
         'wf':tf.Variable(tf.random.truncated_normal([units,fc_neuron_num],stddev=0.04)),
         'out':tf.Variable(tf.random.truncated_normal([fc_neuron_num,2],stddev=1/192)/np.sqrt(fc_neuron_num/2))
         }

biases={'bc1':tf.Variable(tf.zeros([num_filter_1])),
        'bc2':tf.Variable(tf.zeros([num_filter_2])),
        'bf':tf.Variable(tf.zeros([fc_neuron_num])),
        'out':tf.Variable(tf.zeros([2])),
        }    
    
x = tf.placeholder(tf.float32,[None,224,224,1])
y = tf.placeholder(tf.float32,[None,2])

pred = net(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1)),dtype=tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver(weights.update(biases))

train_x,train_y = get_train_batch()
test_x,test_y = get_test_batch()

train_x = decolor(train_x)
test_x = decolor(test_x)

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
            
            sess.run(optimizer,feed_dict={x:x_batch,y:y_batch})
            loss = sess.run(cost,feed_dict={x:x_batch,y:y_batch})
            acc = sess.run(accuracy,feed_dict={x:x_batch,y:y_batch})
            print('loss:',loss)
            print('accuracy:',acc,'\n')
        #Evaluate
        test_batch = 76
        avg_acc_test = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(int(test_x.shape[0]/test_batch)):
            y_pred = sess.run(tf.argmax(pred,axis=1),feed_dict={x:test_x[i*test_batch:(i+1)*test_batch]})
            y_true = np.argmax(test_y[i*test_batch:(i+1)*test_batch],axis=1)

            TP += np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
            FP += np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
            TN += np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
            FN += np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
            
            acc_test = sess.run(accuracy,feed_dict={x:test_x[i*test_batch:(i+1)*test_batch],y:test_y[i*test_batch:(i+1)*test_batch]})
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
