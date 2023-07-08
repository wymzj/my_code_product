import tensorflow as tf
from tensorflow.keras.models import load_model
from concurrent import futures
import grpc
import os
import sys

current_fonder_path = os.path.split(os.path.realpath(__file__))[0]

protocal_path = os.path.join(current_fonder_path,"..","example")

sys.path.append(protocal_path)
import  data_pb2,data_pb2_grpc

model = load_model('check_rnn_interface_long_class.tf')

class CalServicer(data_pb2_grpc.CalServicer):
  def Predict_Corpus(self, request, context):   # Multiply函数的实现逻辑
      print("Predict_Corpus service called")
      predict_input = []
      for i in request.values:
          w = []
          for t in i.voc:
              w.append(t)
          predict_input.append(w)
      print(predict_input)
      pad_word = tf.keras.preprocessing.sequence.pad_sequences(predict_input, maxlen=100)
      text_data_train = tf.convert_to_tensor(pad_word)
      data_list = model.predict(text_data_train)

      send_corpus = data_pb2.ResultFeature()
      for column in range(len(data_list)):
          s = send_corpus.results.add()
          s.bytes_result.extend(data_list[column])

      return data_pb2.ResultFeature(results=send_corpus.results)

#print(text_data_train)

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
  data_pb2_grpc.add_CalServicer_to_server(CalServicer(),server)
  server.add_insecure_port("[::]:50051")
  server.start()
  print("grpc tensorflow server start...")
  server.wait_for_termination()

if __name__ == '__main__':
  serve()
