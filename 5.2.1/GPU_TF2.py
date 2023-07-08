import tensorflow as tf
import os
import timeit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
print('GPU', tf.test.is_gpu_available())
print('GPU:', tf.config.list_physical_devices('GPU'))
with tf.device('/gpu:0'):
    a = tf.constant([1.1,2.2,3.3],shape=[3],name='a')
    b = tf.constant([1.1,2.2,3.3],shape=[3],name='b')
    c = tf.constant([1.1,2.2,3.3],shape=[3],name='c')
with tf.device('/gpu:0'):
    d = a + b + c
print(d)

with tf.device('/gpu:0'):
	cpu_a = tf.random.normal([10000, 1000])
	cpu_b = tf.random.normal([1000, 2000])
	print(cpu_a.device, cpu_b.device)

with tf.device('/gpu:0'):
	gpu_a = tf.random.normal([10000, 1000])
	gpu_b = tf.random.normal([1000, 2000])
	print(gpu_a.device, gpu_b.device)

def cpu_run():
	with tf.device('/gpu:0'):
		c = tf.matmul(cpu_a, cpu_b)
	return c

def gpu_run():
	with tf.device('/gpu:0'):
		c = tf.matmul(gpu_a, gpu_b)
	return c


# warm up
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warmup:', cpu_time, gpu_time)


cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)

