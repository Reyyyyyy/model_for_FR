import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#超参数
units = 256
view_1 = 5
view_2 = 3
num_filter_1 = 36
num_filter_2 = 1
fc_neuron_num = 768
learning_rate = 0.001
batch_size = 64
epochs = 6
dropout = 0.7#每个元素被保留下来的概率

def get_train_batch():
    x_train = np.load(r'D:\核聚变课题组数据\干净数据\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'D:\核聚变课题组数据\干净数据\train_labels.npy',allow_pickle=True)

    return x_train,y_train

def get_test_batch():
    x_test = np.load(r'D:\核聚变课题组数据\干净数据\test_batches.npy',allow_pickle=True)
    y_test = np.load(r'D:\核聚变课题组数据\干净数据\test_labels.npy',allow_pickle=True)

    return x_test,y_test

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
    drop = tf.nn.dropout(fc,keep_prob)
    res = tf.matmul(drop,weights['out']) + biases['out']

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
y = tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32)

pred = net(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1)),dtype=tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver(weights.update(biases))

train_x,train_y = get_train_batch()
test_x,test_y = get_test_batch()

#建图，可视化模型
for w in weights.values():
    tf.summary.histogram('Weights',w)
for b in biases.values():
    tf.summary.histogram('Biases',b)
tf.summary.scalar('cross-entropy',cost)
merged_summary_op = tf.summary.merge_all()

if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter('graphs',sess.graph)
        n_data = train_x.shape[0]
        #Train
        for step in range(int(epochs*n_data/batch_size)):
            pointer = step*batch_size%n_data
            x_batch = train_x[pointer:pointer+batch_size]
            y_batch = train_y[pointer:pointer+batch_size]

            if x_batch.shape[0] == 0:
                continue
            
            sess.run(optimizer,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            loss = sess.run(cost,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            acc = sess.run(accuracy,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            print('loss:',loss)
            print('accuracy:',acc,'\n')
            #将数据传入tensorboard
            summary_str = sess.run(merged_summary_op,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            summary_writer.add_summary(summary_str,step)

        summary_writer.close()
        #Evaluate
        test_batch = 9
        avg_acc_test = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(int(test_x.shape[0]/test_batch)):
            y_pred = sess.run(tf.argmax(pred,axis=1),feed_dict={x:test_x[i*test_batch:(i+1)*test_batch],keep_prob:1.0})
            y_true = np.argmax(test_y[i*test_batch:(i+1)*test_batch],axis=1)

            TP += np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
            FP += np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
            TN += np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
            FN += np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
            
            acc_test = sess.run(accuracy,feed_dict={x:test_x[i*test_batch:(i+1)*test_batch],y:test_y[i*test_batch:(i+1)*test_batch],keep_prob:1.0})
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
