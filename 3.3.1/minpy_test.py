import tensorflow as tf
import numpy as np
#tf.enable_eager_execution(execution_mode=tf.contrib.eager.SYNC)
print(tf.__version__)
"""
# 设定学习率
learning_rate = 0.1
# 训练迭代次数
train_steps = 1000
# 构造训练数据
#train_X = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.799],[6.182],[7.59],[2.167],[7.042],[10.791],[5.313],[7.997],[5.654],[9.27],[3.1]],dtype = np.float32)
#train_Y = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.827],[3.465],[1.65],[2.904],[2.42],[2.94],[1.3]],dtype = np.float32)

train_X = np.array([[3.3],[4.4],[5.5],[6.71],[6.93]],dtype = np.float32)
train_Y = np.array([[6.6],[8.8],[11.0],[13.42],[13.86]],dtype = np.float32)

def linear_regression(data_x, data_y):
    X = data_x
    Y_label = data_y
    # 定义模型参数
    w = tf.Variable(tf.random_normal([1, 1]),name = "weight")
    b = tf.Variable(tf.zeros([1]), name = "bias")
    # 构建模型Y = weight*X + bias
    Y = tf.add(tf.matmul(X, w), b)
    # 定义损失函数
    loss = tf.reduce_sum(tf.square(Y-Y_label))/train_Y.shape[0]
    print(loss)

    return loss

optimizer = tf.train.AdadeltaOptimizer(learning_rate= learning_rate)

# 训练1000次
for i in range(0, train_steps):
    # 在动态图机制下，minimize要求接收一个函数
    #loss = network(train_X, train_Y)
    optimizer.minimize((lambda :linear_regression(train_X, train_Y)))

import tensorflow as tf
import numpy as np
"""
# 一些参数
learning_rate = 0.01  # 学习率
training_steps = 10000  # 训练次数
display_step = 50  # 训练50次输出一次

# 训练数据
#X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
X = np.array([3.3,4.4,5.5,6.71,6.93],dtype = np.float32)
Y = np.array([6.6,8.8,11.0,13.42,13.86],dtype = np.float32)
n_samples = X.shape[0]

# 随机初始化权重和偏置
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# 线性回归函数
def linear_regression(x):
    return W*x + b

# 损失函数
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * n_samples)

# 优化器采用随机梯度下降(SGD)
optimizer = tf.optimizers.SGD(learning_rate)

# 计算梯度，更新参数
def run_optimization():
    # tf.GradientTape()梯度带，可以查看每一次epoch的参数值
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred,Y)
        # 计算梯度
        gradients = g.gradient(loss, [W, b])
    # 更新W，b
    optimizer.apply_gradients(zip(gradients, [W, b]))

# 开始训练
for step in range(1, training_steps+1):
    run_optimization()
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))

"""
print(train_X.shape)
x = tf.placeholder(tf.float32, [train_Y.shape[0],None]) # 占位符
y = tf.placeholder(tf.float32, [train_Y.shape[0],None])

X = train_X
Y_label = train_Y
# 定义模型参数
w = tf.Variable(tf.random_normal([1, 1]),name = "weight")
b = tf.Variable(tf.zeros([1]), name = "bias")
# 构建模型Y = weight*X + bias
Y = tf.add(tf.matmul(X, w), b)
# 定义损失函数
loss = tf.reduce_sum(tf.square(Y-Y_label))/train_Y.shape[0]
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
predict = optimizer.minimize(loss)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(0, train_steps):
        print(sess.run([predict,loss], feed_dict={x: X, y: Y_label}))
    print(sess.run([w,b]))
"""



