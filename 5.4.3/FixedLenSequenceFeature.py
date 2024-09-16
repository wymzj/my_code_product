import tensorflow as tf
import numpy as np
##版本tensorflow 2.1

#1重点理解标准和完整的tf.io.parse_sequence_example####################################################
def generate_tfrecords(tfrecod_filename):
    sequences = [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5],
                 [1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]
    labels = [1, 2, 3, 4, 5, 1, 2, 3, 4]

    with tf.io.TFRecordWriter(tfrecod_filename) as f:
        for feature, label in zip(sequences, labels):
            # 创建一个feature数组
            frame_feature = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), feature))
            frame_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[feature[i]])) for i in range(len(feature))]
            print(frame_feature)
            example = tf.train.SequenceExample(
                context=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}),
                feature_lists=tf.train.FeatureLists(feature_list={'sequence': tf.train.FeatureList(feature=frame_feature)
                })
            )
            f.write(example.SerializeToString())

def single_example_parser(serialized_example):
    context_features = {
        "label": tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "sequence": tf.io.FixedLenSequenceFeature([], dtype=tf.int64,allow_missing=False,default_value=None)#后两个参数都是默认值
    }
    context_sequence_parsed = tf.io.parse_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return context_sequence_parsed

tfrecord_filename = 'parse_sequence_example.tfrecord'
get_data = generate_tfrecords(tfrecord_filename)

raw_dataset = tf.data.TFRecordDataset(tfrecord_filename,buffer_size=60,num_parallel_reads=4)
get_data_list = raw_dataset.map(single_example_parser)
def my_print_1():
    print(get_data_list)
    for tuple_dictionary in get_data_list:
        print(tuple_dictionary)
        print(tuple_dictionary[0]['label'])
        print(tuple_dictionary[1]['sequence'])
        print(tuple_dictionary[2])
my_print_1()

#2重点理解allow_missing=True#############################################################################
"""
生成的解析单个SequenceExample或Example的Tensor具有静态shape：[None] + shape和指定的dtype。
解析batch_size大小的多个Example的结果Tensor具有静态shape：[batch_size, None] + shape和指定的dtype。
来自不同示例的批处理中的条目将使用default_value填充到批处理中存在的最大长度。
要将稀疏输入视为密集，请提供allow_missing=True；否则，任何缺少此功能的示例的解析函数都将失败。
allow_missing：是否允许功能列表项中缺少此功能。仅可用于解析SequenceExample而不用于解析Examples。
"""
def build_tf_example(record):
    return tf.train.Example(features=tf.train.Features(feature=record)).SerializeToString() #注意不是 SequenceExample

def serialize_tf_record(features, targets):
    record = {
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=features.shape)),
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features.flatten())),
        'targets': tf.train.Feature(int64_list=tf.train.Int64List(value=targets)),
    }
    return build_tf_example(record)

def deserialize_tf_record(record):
    tfrecord_format = {
        'shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'features': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True,default_value=0),
        'targets': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
    }
    features_tensor = tf.io.parse_single_example(record, tfrecord_format)
    return features_tensor

features = np.zeros((3, 5, 7))
targets = np.ones((4,), dtype=int)
print(deserialize_tf_record(serialize_tf_record(features, targets)))

#重点理解参数：default_value###############################################################################
default_value_example_file = "default_value_example.tfrecord"
writer = tf.io.TFRecordWriter(default_value_example_file)
def build_tf_example_3(record):
    return writer.write(tf.train.Example(features=tf.train.Features(feature=record)).SerializeToString())

def serialize_tf_record_3(features, targets):
    record = {
        'ft': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        'targets': tf.train.Feature(int64_list=tf.train.Int64List(value=targets)),
    }
    return build_tf_example_3(record)

def serialize_tf_record_4(features, targets):
    record = {
        'targets': tf.train.Feature(int64_list=tf.train.Int64List(value=targets)),
    }
    return build_tf_example_3(record)
features_1 = [1.,2.,3.,4.,5.,6.,7.,8.,9.]
targets = np.ones((4,), dtype=int)
serialize_tf_record_3(features_1, targets)
serialize_tf_record_4(features_1, targets)

def single_example_parser(serialized_example):
    tfrecord_format = {
        'ft': tf.io.FixedLenFeature([9], dtype=tf.float32, default_value=tf.ones([9],dtype=tf.float32)), ##生点关注代码行
        'targets': tf.io.FixedLenFeature((4), dtype=tf.int64),
    }
    context_sequence_parsed = tf.io.parse_example(serialized_example,tfrecord_format)
    return context_sequence_parsed

raw_dataset = tf.data.TFRecordDataset(default_value_example_file)
get_data_list = raw_dataset.map(single_example_parser)

for tuple_dictionary in get_data_list:
    print(tuple_dictionary['ft'])







