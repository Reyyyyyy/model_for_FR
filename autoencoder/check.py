from tensorflow.python.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

#载入数据
imgs = np.load(r'C:\Users\tensorflow\Desktop\核聚变课题组\dateset\test_batches.npy',allow_pickle=True)

#载入模型
autoencoder = model_from_json(open('autoencoder.json').read())
autoencoder.load_weights('weights')

#可视化模型效果
for img in imgs:
    clean_img = autoencoder.predict(img.reshape(1,224,224,3))
    
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('原图')

    plt.subplot(1,2,2)
    plt.imshow(clean_img.reshape(224,224,3))
    plt.title('降噪后的图片')

    plt.pause(1)
    
    
