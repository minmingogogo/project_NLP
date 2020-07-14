#! -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import tensorflow as tf

'''
bert 内存需要：
    实测12g显存batch size 32, max seq len 128有少量盈余，6g酌情减半吧，不过1060速度够呛
BERT实战（源码分析+踩坑）
https://zhuanlan.zhihu.com/p/58471554


https://zhuanlan.zhihu.com/p/104173382
1.获得当前主机上特定运算设备的列表
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
2.设置当前程序可见的设备范围
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
3.显存的使用:设置仅在需要时申请显存空间
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

4、注意

默认情况下，TensorFlow 将使用几乎所有可用的显存，以避免内存碎片化所带来的性能损失；（这也导致不设置显存空间时，会报错）

实际在训练中，如果显存过小，则不应设置过大的batchsize！以免报错。
'''

'''
cpu性能参数设置

'''    
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # gpu 资源
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 1
os.environ['KMP_BLOCKTIME'] = "1"
os.environ['KMP_SETTINGS'] = "1"
os.environ['KMP_AFFINITY'] = "granularity=fine,verbose,compat,1,0"
os.environ['OMP_NUM_THREADS'] = "8"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.Session(config=config)

mode = 0
maxlen = 128    
learning_rate = 5e-5
min_learning_rate = 1e-5

# config_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '../../kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

modeldir = 'D:/myGit/Algorithm/A_my_nlp_project/pre_train_model/bert/chinese_L-12_H-768_A-12 (hagongda)\chinese_L-12_H-768_A-12'
config_path = modeldir + '/bert_config.json'
checkpoint_path = modeldir + '/bert_model.ckpt'
dict_path = modeldir + '/vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)
'''
数据集说明
训练集&验证集：
在训练及验证数据发布阶段，我们会发布2w条左右的文本及其所标注事件类型和事件主体和5千条左右的验证文本及其所标注的事件类型和事件主体。
训练集每行4列，数据以“ ”分隔，格式为：文本id 文本内容 事件类型 事件主体，
测试集每行3列，数据以“ ”分隔，格式为：文本id 文本内容 事件类型。

测试集：
在测试数据发布阶段，我们将会再发布5千条左右的文本数据集，不含标注结果，作为测试。
每行3列，数据以“ ”分隔，格式为：文本id 文本内容 事件类型。

'''
train_data_dir = 'D:/myGit/Algorithm/A_my_nlp_project/bert_in_keras/data_download\ccks2019_event_entity_extract'
D = pd.read_csv(train_data_dir + '/event_type_entity_extract_train.csv', encoding='utf-8', header=None)
'''
#   文本内容分布
text_content =  pd.DataFrame(D[2].value_counts()).reset_index(drop = False)
text_content.columns = ['content','freq']
    content  freq
0        其他  3141
1      信批违规  2513
2   实控人股东变更  1827
3      交易违规  1732
4    涉嫌非法集资  1644
5      不能履职  1326
6      重组失败  1045
7      评级调整   861
8      业绩下滑   686
9      涉嫌违法   616
10     财务造假   592
11     涉嫌传销   500
12     涉嫌欺诈   396
13   资金账户风险   312
14     高管负面   215
15     资产负面   116
16     投诉维权    96
17     产品违规    62
18     提现困难    56
19     失联跑路    43
20     歇业停业    34
21   公司股市异常     2

#   事件类型分布
text_type =  pd.DataFrame(D[3].value_counts()).reset_index(drop = False)
text_type.columns = ['type','freq']
type : 应该是公司/股票名称 subject
'''
# D = pd.read_csv('../ccks2019_event_entity_extract/event_type_entity_extract_train.csv', encoding='utf-8', header=None)
D = D[D[2] != u'其他']
classes = set(D[2].unique())


train_data = []
for t,c,n in zip(D[1], D[2], D[3]):
    train_data.append((t, c, n))


if not os.path.exists('./random_order_train.json'):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('./random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('./random_order_train.json'))


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
additional_chars = set()
for d in train_data + dev_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

additional_chars.remove(u'，')


# D = pd.read_csv('../ccks2019_event_entity_extract/event_type_entity_extract_eval.csv', encoding='utf-8', header=None)
D = pd.read_csv(train_data_dir + '/event_type_entity_extract_eval.csv', encoding='utf-8', header=None)

test_data = []
for id,t,c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i
    return -1


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, c = d[0][:maxlen], d[1]
                text = u'___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[2]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                    s1[start] = 1
                    s2[end] = 1
                    x1, x2 = tokenizer.encode(first=text)
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1, X2, S1, S2], None
                        X1, X2, S1, S2 = [], [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True


x1_in = Input(shape=(None,)) # 待识别句子输入
x2_in = Input(shape=(None,)) # 待识别句子输入
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）

x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

x = bert_model([x1, x2])
''' 12个block 压缩到5个
https://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/
如何获取中间层的输出？

一种简单的方法是创建一个新的Model，使得它的输出是你想要的那个输出
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data

bert_model.inputs 
    Out[99]: 
    [<tf.Tensor 'Input-Token:0' shape=(None, None) dtype=float32>,
     <tf.Tensor 'Input-Segment:0' shape=(None, None) dtype=float32>]
bert_model.outputs 
    [<tf.Tensor 'Encoder-12-FeedForward-Norm/add_1:0' shape=(None, None, 768) dtype=float32>]
for l in bert_model.layers:
    print(l.name)
bert_model.inputs   

layer_name = 'Encoder-4-FeedForward-Norm'                                     
bert_model.get_layer(layer_name)



https://blog.csdn.net/qq_37974048/article/details/102727653?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
失败 K.function
tiny_bert_layers = K.function(bert_model.get_input_at(0),
                              [bert_model.get_layer(layer_name).output])

x = tiny_bert_layers([x1, x2])
'''

layer_name = 'Encoder-1-FeedForward-Norm'                                     


''' 抽取中间层的 写法 '''

tiny_bert_model = Model(inputs = bert_model.get_input_at(0),
                        outputs= [bert_model.get_layer(layer_name).output])
tiny_bert_model.summary()
num_l = len(tiny_bert_model.layers)
for i in range(num_l-8,num_l):
    # print(i)
    l = tiny_bert_model.layers[i]
    l.trainable = True
for l in tiny_bert_model.layers:
    l.trainable = False 
    
    
x = tiny_bert_model([x1, x2])
''' 增加自己训练部分  参数量减少 
D:\tf_enviroment\tf_2-1_py_37\Lib\site-packages\keras_transformer\transformer.py
D:\tf_enviroment\tf_2-1_py_37\Lib\site-packages\keras_bert\bert.py

transformed = get_encoders(
    encoder_num = transformer_num,
    input_layer = embed_layer,
    head_num =head_num,
    hidden_dim =feed_forward_dim,
    attention_activation=attention_activation,
    feed_forward_activation=feed_forward_activation,
    dropout_rate=dropout_rate,
)
'''
from keras_transformer import get_encoders, gelu
#   调整 多头维度
trans_in = Dense(128)(x)

transformed = get_encoders(
    encoder_num = 3,
    input_layer = trans_in,
    head_num = 8 ,
    hidden_dim = 128,# 输出层
    attention_activation=gelu,
    feed_forward_activation=gelu,
    dropout_rate=0.1,
)
x = transformed

'''
for l in tiny_bert_model.layers:
    print(l.name)
检查是否具有权重
tiny_bert_model.layers[2].get_weights()
'''
ps1 = Dense(1, use_bias=False)(x)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
ps2 = Dense(1, use_bias=False)(x)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

model = Model([x1_in, x2_in], [ps1, ps2])


train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
loss = loss1 + loss2

train_model.add_loss(loss)

# class CustomizedSchedule(
#     tf.keras.optimizers.schedules.LearningRateSchedule):
#     '''自定义学习率优化器
#     # lrate = (d_model ** -0.5) * min(step_num ** (-0.5),
# #                                 step_num * warm_up_steps **(-1.5))
#     '''
#     def __init__(self, d_model, warmup_steps = 400):
#         super(CustomizedSchedule, self).__init__()
        
#         self.d_model = tf.cast(d_model, tf.float32)
#         self.warmup_steps = warmup_steps
    
#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** (-1.5))        
#         arg3 = tf.math.rsqrt(self.d_model)        
#         return arg3 * tf.math.minimum(arg1, arg2)    
    
# learning_rate = CustomizedSchedule(768)    
# optimizer = tf.keras.optimizers.Adam(learning_rate,
#                                       beta_1 = 0.9,
#                                       beta_2 = 0.98,
#                                       epsilon = 1e-9)

train_model.compile(optimizer=Adam(learning_rate))
# train_model.compile(optimizer=optimizer)

train_model.summary()


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text_in, c_in):
    if c_in not in classes:
        return 'NaN'
    text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2  = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start:end+1].argmax() + start
    a = text_in[start-1: end]
    return a

checkpoint_dir = 'checkpoint/subject_extract'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.
        self.passed = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights(checkpoint_dir+'/best_model.weights')
        print ('acc: %.4f, best acc: %.4f\n' % (acc, self.best))
    def evaluate(self):
        A = 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
            s = ', '.join(d + (R,))
            # F.write(s.encode('utf-8') + '\n')
            F.write(s + '\n')
        F.close()
        return A / len(dev_data)


def test(test_data):
    F = open('result.txt', 'w')
    for d in tqdm(iter(test_data)):
        s = u'"%s","%s"\n' % (d[0], extract_entity(d[1], d[2]))
        # s = s.encode('utf-8')
        F.write(s)
        
        
    F.close()


evaluator = Evaluate()
train_D = data_generator(train_data,batch_size=64)


if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=10,
                              callbacks=[evaluator]
                             )
else:
    train_model.load_weights(checkpoint_dir+'/best_model.weights')


'''
记录模型差异
77层 multihead:
    参数量：
        Total params: 66,239,232
        Trainable params: 1,536
        Non-trainable params: 66,237,696    
    
3层 multihead:
    参数量：37,887,744 ；Trainable params: 1,536
Epoch 100/100
204/204 [==============================] - 26s 128ms/step - loss: 2.0070
1631it [00:06, 234.16it/s]
acc: 0.4169, best acc: 0.4169

        
Epoch 1/10
408/408 [==============================] - 27s 66ms/step - loss: 7.9582
1631it [00:07, 220.73it/s]
acc: 0.0570, best acc: 0.0570

Epoch 2/10
408/408 [==============================] - 27s 65ms/step - loss: 5.6837
1631it [00:06, 236.98it/s]
acc: 0.1600, best acc: 0.1600

Epoch 3/10
408/408 [==============================] - 26s 65ms/step - loss: 4.9679
1631it [00:06, 237.73it/s]
acc: 0.1931, best acc: 0.1931

Epoch 4/10
408/408 [==============================] - 26s 65ms/step - loss: 4.6363
1631it [00:06, 244.12it/s]
acc: 0.2103, best acc: 0.2103

Epoch 5/10
408/408 [==============================] - 27s 65ms/step - loss: 4.351708 [==>...........................] - ETA: 23s - loss: 4.4289
1631it [00:06, 235.24it/s]
acc: 0.2244, best acc: 0.2244

Epoch 6/10
408/408 [==============================] - 27s 65ms/step - loss: 4.0877
1631it [00:06, 234.73it/s]
acc: 0.2459, best acc: 0.2459

Epoch 7/10
408/408 [==============================] - 27s 65ms/step - loss: 3.8912
1631it [00:06, 235.58it/s]
acc: 0.2575, best acc: 0.2575

Epoch 8/10
408/408 [==============================] - 27s 65ms/step - loss: 3.7026
1631it [00:06, 238.53it/s]
acc: 0.2685, best acc: 0.2685

Epoch 9/10
408/408 [==============================] - 27s 65ms/step - loss: 3.5609A: 11s - loss: 3.5888
1631it [00:06, 238.70it/s]
acc: 0.2728, best acc: 0.2728

Epoch 10/10
408/408 [==============================] - 27s 66ms/step - loss: 3.4201
1631it [00:06, 234.36it/s]
acc: 0.2820, best acc: 0.2820    


'''
