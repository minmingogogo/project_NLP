# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:36:59 2020

@author: Scarlett

当前模块包含以下 tensorflow Model模块
block:
    Cnn : 单个卷积block
    multiKernel_cnn: 横向并列多核卷积
    multiLayer_cnn: 多层卷积
model:
    textcnn:Model 卷积操作可调整为任意卷积block
    resetnetfinetune : 图像预训练模型  Model 可设置某几层参与训练
更多卷积结构：
https://tensorflow.google.cn/api_docs/python/tf/keras/applications/ResNet50?hl=en

可直接作为model 中 一个 block，单独设置某几层训练或不训练
tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet',
    input_tensor=None, input_shape=None,
    pooling=None, classes=1000, **kwargs
)


"""
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import matplotlib as mpl
import matplotlib.pyplot as plt


class Cnn(L.Layer):
    def __init__(self,config):
        super(Cnn,self).__init__()
        self.config = config
        self.con_layer = L.Conv1D(filters=config['num_filters'],
                                  kernel_size=config['kernel_size'])                       
        self.batch_norm = L.BatchNormalization()
        self.activation = L.Activation('relu')
        self.pooling = L.GlobalMaxPool1D()
    def call(self,inputs):
        con = self.con_layer(inputs)
        con = self.batch_norm(con)
        con = self.activation(con)
        con = self.pooling(con)
        return con


class multiKernel_cnn(L.Layer):
    def __init__(self,config):
        super(multiKernel_cnn,self).__init__()    
        self.config = config
        try:
            self.kernel_size_list = config['kernel_size_list']
        except:
            self.kernel_size_list = [2,4]
            
        self.conv1d = L.Conv1D(filters = config['num_filters'],
                             kernel_size = config['kernel_size'],
                             activation = 'relu',
                             kernel_regularizer=K.regularizers.l2(0.0001)
                             )
        self.batch_norm = L.BatchNormalization()
        self.activation = L.Activation('relu')
        self.pooling = L.GlobalMaxPool1D()
        self.flatten = L.Flatten()

    def call(self,inputs):
        '''
        横向并列
        多个尺寸卷积叠加
        '''
        cons = []
        for size  in self.kernel_size_list:            
            con = self.conv1d(inputs)
            con = self.batch_norm(con)
            pool_size = int(con.shape[-2])
            pool = L.MaxPool1D(pool_size)(con)
            cons.append(pool)
        concat_layer = L.concatenate(cons)
        '''卷积后必须Flatten'''

        output = self.flatten(concat_layer)
        return output
  
    
class multiLayer_cnn(L.Layer):
    def __init__(self,config):
        super(multiLayer_cnn,self).__init__()    
        self.config = config
        try:
            self.kernel_size_list = config['kernel_size_list']
        except:
            self.kernel_size_list = [2,4]
            
        self.conv1d = L.Conv1D(filters = config['num_filters'],
                             kernel_size = config['kernel_size'],
                             activation = 'relu',
                             kernel_regularizer=K.regularizers.l2(0.0001)
                             )
        self.batch_norm = L.BatchNormalization()
        self.activation = L.Activation('relu')
        self.pooling = L.MaxPool1D(self.config['pool_size'],padding = 'same')
        self.flatten = L.Flatten()

    def call(self,inputs):
        '''
        深层卷积
        '''
        con = 0
        for _  in range(len(self.config['num_layers'])): 
            if _ == 0:
                con = self.conv1d(inputs)
            else:
                con = self.conv1d(con)
            con = self.batch_norm(con)
            con = self.pooling(con)
            
        '''卷积后必须Flatten'''
        output = self.flatten(con)
        return output

class textcnn(K.Model):
    def __init__(self,config,
                 embedding_matrix=None,
                 is_embedding_matrix=False,
                 trainable = False):
        super(textcnn,self).__init__()
        self.config = config

        if not is_embedding_matrix:
            self.embed = L.Embedding(config['vocab_size'],
                                      config['embed_size'],
                                      input_length =config['max_len'])
        
        else:
            self.embed = L.Embedding(config['vocab_size'],
                                      config['embed_size'],
                                      input_length =config['max_len'],
                                      embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
                                      trainable = trainable)
        self.con_layer = Cnn(config)
        self.mlp = L.Dense(config['hidden_dim'],activation = 'relu')
        self.softmax = L.Dense(config['num_classes'],activation = 'softmax')
        self.sigmoid = L.Dense(1,activation = 'sigmoid')       
    def call(self,inputs,training =True,is_softmax=True):
        input_embedding = self.embed(inputs)
        conv_out = self.con_layer(input_embedding)
        mlp_out = self.mlp(conv_out)
        if is_softmax:
            out = self.softmax(mlp_out)
        else:
            out = self.sigmoid(mlp_out)
        return out        
        

class restnet50_finetune(K.Model):
    def __init__(self,config,
                 embedding_matrix=None,
                 is_embedding_matrix=False,
                 trainable = False):
        super(restnet50_finetune,self).__init__()
        ''' 加载 resnet50 finetune'''
        self.config = config
        self.resnet50 = K.applications.ResNet50(include_top = False, #不包含网络顶层的全连接结构
                                           pooling = 'avg',
                                           weigths = 'imagenet')#   预训练数据名称
        self.softmax = L.Dense(config['num_classes'],activation = 'softmax')
    def call(self,inputs):
        #   设置retnet50 后五层可训练
        
        for layer in self.resnet50.layers[0:-5]:
            layer.trainable = False
        out = self.resnet50(inputs)
        out = self.softmax(out)
        return out
if __name__ == '__main__':
    config = {
        'epoch':10,
        'batch_size':16,
        'num_classes':96,        
        'max_len':80,
        'embed_size':256,
        'vocab_size':3196,
        #   普通层配置
        'mlp_units':64,
        'hidden_dim':512,
        'activation':'relu',
        #   卷积层配置
        'num_layers':3,   # 多层卷积，通道和卷积核应该调整，否则可能后面层参数不够
        'num_filters':512, #卷积输出通道
        'kernel_size':5,
        'kernel_size_list':[2,4] ,#考虑多尺寸卷积核
        'pool_size':5 # 池化尺寸
        }     

