第1章 烂熟于心中的基础知识	1

1.1快速恶补必要的数学知识	1

1.1.1  线性代数提供了一种看待世界的抽象视角	2

1.1.2  概率统计是用面积度量世界万物的存在	20

1.1.3  微积分运算解决了一定条件下直线到曲线的矛盾	31

1.1.4  信息论的产物-交叉熵	38

1.2  Python语言没有你想像的那样简单	40

1.2.1  Python模块的导入和引用	40

1.2.2  Python那些奇怪符号的用法	48

1.2.3  仅Python才有的神奇代码	59

1.2.4  Python代码高级综合案例	62

1.3  选择TensroFlow1.X还是2.X的理由	72

1.3.1  TensorFlow技术概要	73

1.3.2  TensorFlow1.X与TensorFlow2.X有哪些区别	77

1.3.3  TensorFlow1.X和TensorFlow2.X手写数字识别	82

1.4 总结	88

第2章 模型工程化必备的技能	90

2.1  模型转换为云服务的桥梁-Docker	91

2.1.1  映像、容器和隔离	91

2.1.2  Docker Compose	101

2.1.3  大规模使用Docker	107

2.2  GIT代码管理让你融入优秀的团队	110

2.2.1  安装Git	110

2.2.2  代码数据仓库的搭建	116

2.2.3  GIT的流行仅因为多了一个“分支”	123

2.2.4  Git删除文件、文件重命名、去除提交和文件恢复	130

2.2.5  远程版本管理	134

2.3 总结	147

第3章 TensorFlow安装和配置	150

3.1 Windows开发环境配置	150

3.1.1  Anaconda	150

3.1.2  cudatoolkit和cudnn安装	154

3.1.3  Anaconda+Python+TensorFlow-GPU+Pycharm	158

3.2  Linux开发环境配置	162

3.2.1  构建TensorFlow运行环境三种方式	162

3.2.2  Virtualenv和Docker	165

3.2.3  Jupyter Notebook	167

3.2.4  Vim	175

3.3  Python常用的科学、算法和机器学习库	183

3.3.1  NumPy	183

3.3.2  Matplotlib	191

3.3.3  Pandas	197

3.3.4  Python SciPy	208

3.3.5  Scikit-learn	212

3.4  总结	220

第4章 云端部署TensorFlow模型	222

4.1  RPC原理	222

4.1.1  如何实现远程RPC调用	224

4.2  远程调用通讯机制	225

4.2.1  发布服务	227

4.2.2  使用Python实现RPC服务	230

4.2.3  JSON进行序化和反序化	248

4.3  TensorFlow Serving 发布服务	252

4.3.1	TensorFlow Serving安装和Docker示例	253

4.3.2  TensorFlow Serving的Docker环境	254

4.3.3  客户端远程调用TensorFlow Serving服务	260

4.3.4  TensorFlow Serving简化版实现	265

4.3.5  使用gRPC调用服务	270

4.4 总结	273

第5章  TensorFlow基础	274

5.1 TensorFlow基本框架概念	274

5.1.1  TensorFlow基本概念	277

5.1.2  动态模式进行简单线性回归训练	279

5.1.3  估算框架接口Estimators API	285

5.1.4  tf.keras接口	293

5.1.5  CNN卷积核的多样性	300

5.1.6  循环神经网络RNN	303

5.2  TensorFlow的GPU分配资源和策略	323

5.2.1 为整个程序指定GPU卡	323

5.2.2 个性化定制GPU资源	325

5.2.3 使用GPU分配策略	326

5.3 TensorFlow模型训练保存与加载	339

5.3.1  用静态数据流图保存、二次训练和加载	339

5.3.2  Build方式保存模型：	342

5.3.3  TensorFlow2.X训练模型的保存与加载	345

5.4  TFRecord	356

5.4.1  tf.Example数据类型	356

5.4.2  读取序化文件形成数据集	360

5.4.3  对图像进行序列化处理	361

5.4.4  对样本图像的批量复杂处理	363

5.4.5  VarLenFeature和FixedLenFeature区别	367

5.4.6  CSV文件转换为TFRecord	373

5.4.7  XML文件转换为TFRecord	374

5.5 总结	378

第6章 经典神经网络框架	380

6.1  AlexNet—AI潮起	380

6.1.1  AlexNet架构	380

6.1.2  AlexNet带来的新技术	382

6.1.3  AlexNet的TensorFlow2.0实现	384

6.2  VGGNet—更小的卷积造就更深的网络	390

6.2.1  VGGNet架构	390

6.2.2  VGGNet中的创新点	392

6.2.3  VGGNet的TensorFlow2.0实现	394

6.3  GoogleNet—走向更深更宽的网络	400

6.3.1  GoogleNet架构	400

6.3.2  GoogleNet中的创新点	401

6.3.3  GoogleNet v1的TensorFlow2.0实现	403

6.4  ResNet—残差网络独领风骚	409

6.4.1  ResNet架构	409

6.4.2  ResNet基于TensorFlow2.0的实现	412

6.5  SENet—视觉注意力机制的起点	417

6.5.1  SENet架构	417

6.5.2  SENet_ResNet的TensorFlow2.0实现	418

6.6 Self-Attention—自注意力机制	424

6.6.1  Self-Attention的原理	424

6.7  Vision Transformer—注意力引爆视觉任务	426

6.7.1  ViT的TensorFlow2.0实现	427

6.8 总结	429

第7章 目标检测	430

7.1  RCNN—在巨人的肩膀上前进	430

7.1.1  RCNN架构	430

7.2  SPPNET	433

7.2.1  空间金字塔池化层	433

7.2.2  SPPNet流程	435

7.3 Fast R-CNN	436

7.3.1  ROI Pooling	436

7.3.2  Fast R-CNN与R-CNN的不同	437

7.3.3  Fast R-CNN算法流程	437

7.4 Faster R-CNN	438

7.4.1  Region proposal network	438

7.4.2  Anchor	439

7.4.3  Faster RCNN流程	441

7.4.4  Faster RCNN的Tensorflow2.0实现	442

7.5 SSD目标检测算法	449

7.5.1  多尺度特征预测	449

7.5.2  边界框的定制	450

7.5.3  Dilation Convolution	451

7.5.4  SSD的训练过程与细节	452

7.5.5  SSD的优缺点	453

7.6 YOLO目标检测算法	455

7.6.1  YOLO V1	455

7.6.2  YOLO V2	460

7.6.3  YOLO V3	464

7.6.4  YOLO V4—Trick集大成者	467

7.7 总结	476

第8章 深度学习综合实践案例	479

8.1  肺结节检测系统研发案例	479

8.1.1  胸片数据标注	480

8.1.2  模型训练过程	490

8.2 食品、医药类相关新闻抓取案例	502
 
 
8.2.1  新闻语料样本和分词工具	502

8.2.2  GRU循环神经网络模型	504

8.2.3  gRPC远程调用训练后模型	510

8.3  做个孤夜独行的技术人	515


