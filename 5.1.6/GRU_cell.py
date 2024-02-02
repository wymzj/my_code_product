import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

#   设置相关底层配置
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#   只取10000个单词，超过10000的按生僻词处理
total_words = 10000
max_sentencelength = 121         #   每个句子最大长度
batchsize = 2000
embedding_len = 100             #   将单词从原来的的一个数扩充为100维的向量

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)         #   numweord为单词种类个数
print(y_train)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen = max_sentencelength)        #   把句子长度限制为定长
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen = max_sentencelength)

db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.shuffle(1000).batch(batch_size=batchsize,drop_remainder=True)   #   设置drop参数可以把最后一个batch如果与前面的batch长度不一样，就丢弃掉
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.batch(batch_size=batchsize,drop_remainder=True)
print(type(db_test))
class MyRnn(tf.keras.Model):
    def __init__(self,units):
        super(MyRnn,self).__init__()

        self.embedding = tf.keras.layers.Embedding(
           total_words,
           embedding_len,
           input_length = max_sentencelength)

        self.rnn = tf.keras.Sequential([
            tf.keras.layers.LSTM(units, return_sequences=True, unroll=True),
            tf.keras.layers.LSTM(units, unroll=True),
            #tf.keras.layers.GRU(units),# return_sequences=True, unroll=True),
            #tf.keras.layers.GRU(units, unroll=True),
        ])
        #   fc , [b,80,100] =>[b,64]=>[b,1]
        self.outlayer = tf.keras.layers.Dense(1)


    def __call__(self, inputs, training = None):

        # [b,80]
        x = inputs
        x = self.embedding(x)

        #   [b,80,100] = [b,64]
        x = self.rnn(x)

        # out:[b,64] => [b,1]
        x = self.outlayer(x)
        prob = tf.sigmoid(x)
        return prob

if __name__ == '__main__':
    units = 64
    epochs = 40
    lr = 1e-2
    model = MyRnn(units)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss= tf.losses.BinaryCrossentropy(),     #   二分类的loss函数
                  metrics=['accuracy'])
    model.fit(db_train,epochs=epochs,validation_data=db_test)
    model.evaluate(db_test)