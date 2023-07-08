# coding:utf-8
'''
Inception module
'''

import tensorflow.compat.v1 as tf


def inception4(inputs,
               sub_chs,
               stride,
               is_training,
               scope='inception'):
    '''
    Figure 4

    sub_ch1: 1x1
    sub_ch2: 1x1  > 3x3
    sub_ch3: 1x1  > 5x5
    sub_ch4: pool > 1x1

    '''
    x = inputs
    [sub_ch1, sub_ch2, sub_ch3, sub_ch4] = sub_chs
    sub = []

    # 1x1
    sub1 = tf.layers.Conv2D(sub_ch1, [1, 1], stride, padding='SAME')(x)
    sub1 = tf.layers.BatchNormalization()(sub1, training=is_training)
    sub1 = tf.nn.relu(sub1)
    sub.append(sub1)

    # 1x1 > 3x3
    sub2 = tf.layers.Conv2D(sub_ch2[0], [1, 1], padding='SAME')(x)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub2 = tf.layers.Conv2D(sub_ch2[1], [3, 3], stride, padding='SAME')(sub2)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub.append(sub2)

    # 1x1 > 5x5
    sub3 = tf.layers.Conv2D(sub_ch3[0], [1, 1], padding='SAME')(x)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [5, 5], stride, padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub.append(sub3)

    # pool > 1x1
    if sub_ch4[1] == None:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], stride, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], stride, padding='SAME')(x)
        else:
            raise ValueError
    else:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], 1, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], 1, padding='SAME')(x)
        else:
            raise ValueError
        sub4 = tf.layers.Conv2D(sub_ch4[1], [1, 1], stride, padding='SAME')(sub4)
        sub4 = tf.layers.BatchNormalization()(sub4, training=is_training)
        sub4 = tf.nn.relu(sub4)
    sub.append(sub4)

    x = tf.concat(sub, axis=-1)
    return x


def inception5(inputs,
               sub_chs,
               stride,
               is_training,
               scope='inception'):
    '''
    Figure 5

    sub_ch1: 1x1
    sub_ch2: 1x1  > 3x3
    sub_ch3: 1x1  > 3x3 > 3x3
    sub_ch4: pool > 1x1

    '''
    x = inputs
    [sub_ch1, sub_ch2, sub_ch3, sub_ch4] = sub_chs
    sub = []

    # 1x1
    sub1 = tf.layers.Conv2D(sub_ch1, [1, 1], stride, padding='SAME')(x)
    sub1 = tf.layers.BatchNormalization()(sub1, training=is_training)
    sub1 = tf.nn.relu(sub1)
    sub.append(sub1)

    # 1x1 > 3x3
    sub2 = tf.layers.Conv2D(sub_ch2[0], [1, 1], padding='SAME')(x)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub2 = tf.layers.Conv2D(sub_ch2[1], [3, 3], stride, padding='SAME')(sub2)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub.append(sub2)

    # 1x1  > 3x3 > 3x3
    sub3 = tf.layers.Conv2D(sub_ch3[0], [1, 1], padding='SAME')(x)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [3, 3], 1, padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [3, 3], stride, padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub.append(sub3)

    # pool > 1x1
    if sub_ch4[1] == None:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], stride, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], stride, padding='SAME')(x)
        else:
            raise ValueError
    else:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], 1, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], 1, padding='SAME')(x)
        else:
            raise ValueError
        sub4 = tf.layers.Conv2D(sub_ch4[1], [1, 1], stride, padding='SAME')(sub4)
        sub4 = tf.layers.BatchNormalization()(sub4, training=is_training)
        sub4 = tf.nn.relu(sub4)
    sub.append(sub4)

    x = tf.concat(sub, axis=-1)
    return x


def inception6(inputs,
               n,
               sub_chs,
               stride,
               is_training,
               scope='inception'):
    '''
    Figure 6

    sub_ch1: 1x1
    sub_ch2: 1x1  > 1xn > nx1
    sub_ch3: 1x1  > 1xn > nx1 > 1xn > nx1
    sub_ch4: pool > 1x1

    '''
    x = inputs
    [sub_ch1, sub_ch2, sub_ch3, sub_ch4] = sub_chs
    sub = []

    # 1x1
    sub1 = tf.layers.Conv2D(sub_ch1, [1, 1], stride, padding='SAME')(x)
    sub1 = tf.layers.BatchNormalization()(sub1, training=is_training)
    sub1 = tf.nn.relu(sub1)
    sub.append(sub1)

    # 1x1 > 1xn > nx1
    sub2 = tf.layers.Conv2D(sub_ch2[0], [1, 1], padding='SAME')(x)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub2 = tf.layers.Conv2D(sub_ch2[1], [1, n], padding='SAME')(sub2)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub2 = tf.layers.Conv2D(sub_ch2[1], [n, 1], stride, padding='SAME')(sub2)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub.append(sub2)

    # 1x1  > 1xn > nx1 > 1xn > nx1
    sub3 = tf.layers.Conv2D(sub_ch3[0], [1, 1], padding='SAME')(x)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [1, n], padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [n, 1], padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [1, n], padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [n, 1], stride, padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub.append(sub3)

    # pool > 1x1
    if sub_ch4[1] == None:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], stride, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], stride, padding='SAME')(x)
        else:
            raise ValueError
    else:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], 1, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], 1, padding='SAME')(x)
        else:
            raise ValueError
        sub4 = tf.layers.Conv2D(sub_ch4[1], [1, 1], stride, padding='SAME')(sub4)
        sub4 = tf.layers.BatchNormalization()(sub4, training=is_training)
        sub4 = tf.nn.relu(sub4)
    sub.append(sub4)

    x = tf.concat(sub, axis=-1)
    return x


def inception7(inputs,
               sub_chs,
               stride,
               is_training,
               scope='inception'):
    '''
    Figure 7

    sub_ch1: 1x1
    sub_ch2: 1x1  > 3x3
    sub_ch3: 1x1  > 3x3 > 3x3
    sub_ch4: pool > 1x1

    '''
    x = inputs
    [sub_ch1, sub_ch2, sub_ch3, sub_ch4] = sub_chs
    sub = []

    # 1x1
    sub1 = tf.layers.Conv2D(sub_ch1, [1, 1], stride, padding='SAME')(x)
    sub1 = tf.layers.BatchNormalization()(sub1, training=is_training)
    sub1 = tf.nn.relu(sub1)
    sub.append(sub1)

    # 1x1 > 1x3 and 3x1
    sub2 = tf.layers.Conv2D(sub_ch2[0], [1, 1], padding='SAME')(x)
    sub2 = tf.layers.BatchNormalization()(sub2, training=is_training)
    sub2 = tf.nn.relu(sub2)
    sub21 = tf.layers.Conv2D(sub_ch2[1], [1, 3], stride, padding='SAME')(sub2)
    sub21 = tf.layers.BatchNormalization()(sub21, training=is_training)
    sub21 = tf.nn.relu(sub21)
    sub.append(sub21)
    sub22 = tf.layers.Conv2D(sub_ch2[1], [3, 1], stride, padding='SAME')(sub2)
    sub22 = tf.layers.BatchNormalization()(sub22, training=is_training)
    sub22 = tf.nn.relu(sub22)
    sub.append(sub22)

    # 1x1  > 3x3 > 1x3 and 3x1
    sub3 = tf.layers.Conv2D(sub_ch3[0], [1, 1], padding='SAME')(x)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub3 = tf.layers.Conv2D(sub_ch3[1], [3, 3], 1, padding='SAME')(sub3)
    sub3 = tf.layers.BatchNormalization()(sub3, training=is_training)
    sub3 = tf.nn.relu(sub3)
    sub31 = tf.layers.Conv2D(sub_ch3[1], [1, 3], stride, padding='SAME')(sub3)
    sub31 = tf.layers.BatchNormalization()(sub31, training=is_training)
    sub31 = tf.nn.relu(sub31)
    sub.append(sub31)
    sub32 = tf.layers.Conv2D(sub_ch3[1], [1, 3], stride, padding='SAME')(sub3)
    sub32 = tf.layers.BatchNormalization()(sub31, training=is_training)
    sub32 = tf.nn.relu(sub32)
    sub.append(sub32)

    # pool > 1x1
    if sub_ch4[1] == None:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], stride, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], stride, padding='SAME')(x)
        else:
            raise ValueError
    else:
        if sub_ch4[0] == 'max':
            sub4 = tf.layers.MaxPooling2D([3, 3], 1, padding='SAME')(x)
        elif sub_ch4[0] == 'avg':
            sub4 = tf.layers.AveragePooling2D([3, 3], 1, padding='SAME')(x)
        else:
            raise ValueError
        sub4 = tf.layers.Conv2D(sub_ch4[1], [1, 1], stride, padding='SAME')(sub4)
        sub4 = tf.layers.BatchNormalization()(sub4, training=is_training)
        sub4 = tf.nn.relu(sub4)
    sub.append(sub4)

    x = tf.concat(sub, axis=-1)
    return x


def aux_classifier(inputs):
    '''
    Figure 8
    '''
    x = inputs
    x = tf.layers.AveragePooling2D([5, 5], 3)(x)
    x = tf.layers.Conv2D(128, [1, 1])(x)
    x = tf.layers.Conv2D(1024, [5, 5])(x)
    logits = tf.layers.Conv2D(1000, [1, 1])(x)
    return logits


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [192, 28, 28, 3])
    y4 = inception4(x, [64, [96, 128], [16, 32], ['max', 32]], 2, is_training=True, scope='1')
    y5 = inception5(x, [64, [96, 128], [16, 32], ['max', 32]], 2, is_training=True, scope='1')
    y6 = inception6(x, 3, [64, [96, 128], [16, 32], ['max', 32]], 2, is_training=True, scope='1')
    y7 = inception7(x, [64, [96, 128], [16, 32], ['max', 32]], 2, is_training=True, scope='1')
