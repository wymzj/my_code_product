import os
import sys
import grpc
import jieba
import string
import json
current_fonder_path = os.path.split(os.path.realpath(__file__))[0]
protocal_path = os.path.join(current_fonder_path,"..","example")
sys.path.append(protocal_path)
import  data_pb2,data_pb2_grpc


def run(n, m):
  channel = grpc.insecure_channel('localhost:50051') # 连接上gRPC服务端
  stub = data_pb2_grpc.CalStub(channel)
  response = stub.Add(data_pb2.AddRequest(num1=n, num2=m))
  print(f"{n} + {m} = {response.number}")
  response = stub.subtract(data_pb2.subtractRequest(num1=n, num2=m))
  print(f"{n} - {m} = {response.number}")

if __name__ == "__main__":
  run(201, 402)
