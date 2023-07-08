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


voc_dict = json.load(open('./vocab.dict','r'))
print(len(voc_dict))

text = ['我爱北京天安门，天安门上太阳长，我爱伟大领袖毛主席，毛主席带领我们向前进。上面的代码无不让人惊叹其简捷和优美。',
        '最后该给大家补充一个上面遗漏的知识点，在BasicLSTMCell方法中有个参数reuse描述是否在现有scope中重用共享变量。如果为False，在现有scope已经具有给定变量则会引发错误，布尔类型。',
        '国家食品药品监督管理总局去年在全国范围内组织抽检了婴幼儿类化妆品1011批次，抽样检验项目合格样品1010批次，仅上海丽婴房婴童用品有限公司委托上海华妮透明美容香皂有限公司生产的宝贝可爱婴儿润肤露1批次不合格',
        '王艳铭是大坏蛋吃药疯了，吃错药了，正在办理食品加工生产许可证，预包装食品和茶叶，保证质量合格没有毒，到国家食品监督管理局备案和被查，买化妆品，保健品食品加工']


text_data_train = [''.join(c for c in x if c not in string.punctuation) for x in text]
text_data_train = [' '.join(jieba.cut(x)) for x in text_data_train]

print(len(text_data_train))
predict_input = []
for i in range(len(text_data_train)):
    predict_input_mem =[]
    for n in text_data_train[i].split(' '):
        predict_input_mem.append(int(voc_dict.get(n,'0')))
    predict_input.append(predict_input_mem)

send_corpus = data_pb2.CorpusVoc()
for column in range(len(predict_input)):
    s = send_corpus.values.add()
    s.voc.extend(predict_input[column])

def run(request_message):
  channel = grpc.insecure_channel('localhost:50051') # 连接上gRPC服务端
  stub = data_pb2_grpc.CalStub(channel)
  response = stub.Predict_Corpus(data_pb2.CorpusVoc(values=request_message))
  print(f"Predict_Corpus = {response}")
if __name__ == "__main__":
  run(send_corpus.values)
