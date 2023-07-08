import os
import io
import csv
import requests
import jieba
import string
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import json



class MyModel(tf.keras.Model):
    def __init__(self,batch_size=60,rnn_size=100,data_dir=None,csv_file=None):
        super(MyModel, self).__init__()
        print(sys.version)
        print(tf.__version__)

        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.embedding_size = 300  # 嵌套在长度为50的词向量中

        self.text_data = []
        self.vocab_processor = tf.keras.preprocessing.text.Tokenizer(num_words=100,split=" ",filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')#sequences_to_matrix(self.text_data, mode="binary")

        self.data_train = []
        self.data_target = []
        if (data_dir is not None and csv_file is not None):
            self.read_csv(data_dir,csv_file)
        text_data_train = [x[1] for x in self.text_data]
        text_data_target = [x[0] for x in self.text_data]
        text_data_train = [''.join(c for c in x if c not in string.punctuation) for x in text_data_train]
        text_data_train = [' '.join(jieba.cut(x)) for x in text_data_train]
        text_data_train = [' '.join(x.split()) for x in text_data_train]

        self.vocab_processor.fit_on_texts(text_data_train)
        json.dump(self.vocab_processor.word_index, open('./vocab.dict', 'w'))
        pad_word = tf.keras.preprocessing.sequence.pad_sequences(
            self.vocab_processor.texts_to_sequences(text_data_train), maxlen=100)
        text_data_train = tf.convert_to_tensor(pad_word)
        text_data_target = tf.convert_to_tensor([0 if x == '0' else 1 for x in text_data_target])

        ix_cutoff = int(len(text_data_target) * 0.80)
        x_train, x_test = text_data_train[:ix_cutoff], text_data_train[ix_cutoff:]
        y_train, y_test = text_data_target[:ix_cutoff], text_data_target[ix_cutoff:]

        db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.db_train = db_train.shuffle(1000).batch(batch_size=self.batch_size,drop_remainder=True)  # 设置drop参数可以把最后一个batch如果与前面的batch长度不一样，就丢弃掉
        db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.db_test = db_test.batch(batch_size=self.batch_size, drop_remainder=True)

        self.cell = tf.keras.layers.RNN(tf.keras.layers.GRUCell((self.rnn_size)))  # state_size = rnn_size = 10

        self.embedding = tf.keras.layers.Embedding(
            len(self.vocab_processor.word_index),
            self.embedding_size,
            input_length=200)
        self.outlayer = tf.keras.layers.Dense(1)

        self.dropout = tf.keras.layers.Dropout(0.2)
    def read_csv(self,data_dir,csv_file):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if os.path.isfile(os.path.join(data_dir, csv_file)):
            save_file_name = os.path.join(data_dir,csv_file)  # cosmetics-notitle.csv  temp_spam_data.csv SYJ_corpus.csv
            if os.path.isfile(save_file_name):
                with open(save_file_name, 'r',encoding='utf-8') as temp_output_file:
                    reader = csv.reader(temp_output_file)
                    for row in reader:
                        if len(row[0]) == 1 and len(row[1]) > 0:
                            self.text_data.append(row)

    def deal_with_data(self,input_text):
        if len(input_text) == 0:
            return None
        text_data_train = [''.join(c for c in x if c not in string.punctuation) for x in input_text]
        text_data_train = [' '.join(jieba.cut(x)) for x in text_data_train]
        text_data_train = [' '.join(x.split()) for x in text_data_train]

        pad_word = tf.keras.preprocessing.sequence.pad_sequences(self.vocab_processor.texts_to_sequences(text_data_train),maxlen=100)
        text_data_train = tf.convert_to_tensor(pad_word)

        return text_data_train

    def call(self, inputs, training = None):
        output_em = self.embedding(inputs)
        output_b = self.cell(output_em)  #state [250  25  10] [250,10]
        output_f = self.dropout(output_b)
        out = self.outlayer(output_f)
        logits_out = tf.sigmoid(out)
        return logits_out

if __name__ == '__main__':
    epochs = 2
    learning_rate = 1e-2
    model = MyModel(60,100,'temp','cosmetic-food-medical-apparatus.csv')

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                  loss= tf.losses.BinaryCrossentropy(),     #二分类的loss函数
                  metrics=['accuracy'])
    model.fit(model.db_train,epochs=epochs,validation_data=model.db_test)
    model.evaluate(model.db_test)
    model.save("./RPC/server/check_rnn_interface_long_class.tf")
    #model.save_weights("./temp/check_rnn_interface_long_weight.tf")
