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

class CalServicer(data_pb2_grpc.CalServicer):
  def Add(self, request, context):
    return data_pb2.ResultReply(number=request.num1 + request.num2)

  def subtract(self, request, context):
    return data_pb2.ResultReply(number=request.num1 - request.num2)

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  data_pb2_grpc.add_CalServicer_to_server(CalServicer(),server)
  server.add_insecure_port("[::]:50051")
  server.start()
  server.wait_for_termination()

if __name__ == '__main__':
  serve()
