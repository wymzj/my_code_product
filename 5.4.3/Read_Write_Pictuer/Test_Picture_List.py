import os
import tensorflow as tf
##版本tensorflow 2.1
classes = {'up', 'down'} #预先自己定义的类别，根据自己的需要修改
writer = tf.io.TFRecordWriter("train.tfrecords")  #train表示转成的tfrecords数据格式的名字
train_filenames = ['C:\\Users\\ThinkPad\\Pictures\\' + filename for filename in os.listdir('C:\\Users\\ThinkPad\\Pictures')
                   if filename.find('.jpg') > 0 and filename.find('.png') < 0]
print(train_filenames)
for index, name in enumerate(classes):#按不同类别的目录进行检索标注
    for img_name in train_filenames:
        img = open(img_name,'rb')
        img_raw = img.read()
        print(img_raw)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
        img.close()
writer.close()
