from concurrent import futures
import grpc
import os
import sys
current_fonder_path = os.path.split(os.path.realpath(__file__))[0]
print (current_fonder_path)
protocal_path = os.path.join(current_fonder_path,"..","example")
print (protocal_path)
sys.path.append(protocal_path)
import  data_pb2,data_pb2_grpc

data_list = [[0.86],[0.02]]
send_corpus = data_pb2.ResultFeature()
for column in range(len(data_list)):
    s = send_corpus.results.add()
    s.bytes_result.extend(data_list[column])

class CalServicer(data_pb2_grpc.CalServicer):
  def Add(self, request, context):   # Add函数的实现逻辑
    print("Add function called")
    return data_pb2.ResultReply(number=request.number1 + request.number2)

  def Multiply(self, request, context):   # Multiply函数的实现逻辑
    print("Multiply service called")
    return data_pb2.ResultReply(number=request.number1 * request.number2)

  def Predict_Corpus(self, request, context):   # Multiply函数的实现逻辑
      print("Predict_Corpus service called")
      s = []
      for i in request.values:
          l = []
          for t in i.voc:
              l.append(t)
          s.append(l)
      print(s)
      return data_pb2.ResultFeature(results=send_corpus.results)



def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
  data_pb2_grpc.add_CalServicer_to_server(CalServicer(),server)
  server.add_insecure_port("[::]:50051")
  server.start()
  print("grpc server start...")
  server.wait_for_termination()

if __name__ == '__main__':
  serve()
