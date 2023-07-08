import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import os

train_filenames = ['C:\\Users\\ThinkPad\\Pictures\\' + filename for filename in os.listdir('C:\\Users\\ThinkPad\\Pictures') if filename.find('.jpg') > 0 and filename.find('.png') < 0]
print(train_filenames)
#第一步：获取原始数据
image = open('C:\\Users\\ThinkPad\\Pictures\\just do it (2).jpg','rb').read()
len_pic = len(train_filenames)

data=pd.read_csv('DataSet/titanic.csv')

#第二步：定义record文件
tfrecord_file='titanic_train.tfrecords'
writer=tf.io.TFRecordWriter(tfrecord_file)
#第三步：每一次写入一条样本记录
for i in range(len(data)):

    image_file = open(train_filenames[i%len_pic], 'rb')
    read_image = image_file.read()
    features=tf.train.Features(feature={'Age':tf.train.Feature(float_list=tf.train.FloatList(value=[data['Age'][i]])),
                                        'Sex':tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if data['Sex'][i]=='male' else 0])),
                                        'Pclass':tf.train.Feature(int64_list=tf.train.Int64List(value=[data['Pclass'][i]])),
                                        'Parch':tf.train.Feature(int64_list=tf.train.Int64List(value=[data['Parch'][i]])),
                                        'SibSp':tf.train.Feature(int64_list=tf.train.Int64List(value=[data['SibSp'][i]])),
                                        'Fare':tf.train.Feature(float_list=tf.train.FloatList(value=[data['Fare'][i]])),
                                        'Survived':tf.train.Feature(int64_list=tf.train.Int64List(value=[data['Survived'][i]])),
                                        'Name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(data['Name'][i],encoding='utf8')])),
                                        #'Name': tf.train.Feature(bytes_list=tf.train.BytesList(value=data['Name'][i])),
                                        'Image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[read_image]))
                                        })
    #每一条样本的特征，将一系列特征组织成一条样本
    example=tf.train.Example(features=features)
    #将每一条样本写入到tfrecord文件
    writer.write(example.SerializeToString())
    image_file.close()
#第四步：写入后关闭文件
writer.close()
