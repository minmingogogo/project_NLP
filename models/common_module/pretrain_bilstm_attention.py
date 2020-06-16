# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:46:51 2020

@author: Scarlett
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np
import pandas as pd
import sklearn
import os
import json
from load_embedding_data import load_embedding_data




class Pretrain_biLstm_attention(K.models.Model):
    def __init__(self,config,
                 embedding_matrix=None,
                 is_embedding_matrix=False,
                 trainable = False):
        super(Pretrain_biLstm_attention,self).__init__()
        ''' 
        注意使用预训练模型需要对齐词表，
        vocabsize,embed_size 
        可能涉及重新生成训练数据，word2id,id2word
        
        参数子类api:调用方式
            config = {
                'epoch':10,
                'embed_size':256,
                'vocab_size':3196,
                'max_len':80,
                'mlp_units':64,
                'num_classes':96,
                'hidden_dim':512,
                'activation':'relu',
                'batch_size':16
                }
                    
            model = Pretrain_biLstm_attention(config)
            x = np.random.randint(0,1000,2000)
            x = x.reshape((-1,config['max_len']))
            y = np.random.randint(0,config['num_classes'],len(x))    
            y_pre = model(x)  
            
            层级参数获取：
            model.layers[0].get_weights()

        '''
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
                
        self.bilstm = L.Bidirectional(L.LSTM(config['hidden_dim'],
                                             return_sequences = True))
        self.attention = L.Attention()
        self.maxpooling=L.GlobalMaxPooling1D()
        self.concat = L.Concatenate()
        try:
            activation = config['activation']
        except:
            activation = 'relu'
        self.mlp = L.Dense(config['mlp_units'],activation = activation)
        self.softmax = L.Dense(config['num_classes'],activation = 'softmax')
        self.sigmoid = L.Dense(1,activation = 'sigmoid')
    def call(self,inputs,is_softmax=True,training=True):
        input_embedding = self.embed(x)
        bilstm_out = self.bilstm(input_embedding)
        attention_out = self.attention([bilstm_out,bilstm_out])
        pooling1 = self.maxpooling(bilstm_out)
        pooling2 = self.maxpooling(attention_out)
        #   合并lstm 和 attention 结果
        merge = self.concat([pooling1,pooling2])
        mlp_out = self.mlp(merge)
        if is_softmax:
            out = self.softmax(mlp_out)
        else:
            out = self.sigmoid(mlp_out)
        return out


# =============================================================================
# 下面这种为快速训练模式 model=> model.compile =>model.fit 即可
# =============================================================================
def bilstm_attention_Model():
    '''
    函数式写法
    model = K.Model(inputs=[],
                    outputs = [])

    '''

    input_layer = L.Input(shape = (config['max_len'],config['embed_size']))
    bilstm_out = L.Bidirectional(L.LSTM(units = config['hidden_dim'],
                                        return_sequences = True))(input_layer)
    attention_out = L.Attention()([bilstm_out,bilstm_out])
    pooling1 = L.GlobalMaxPooling1D()(bilstm_out)
    pooling2 = L.GlobalMaxPooling1D()(attention_out)
    output_layer = L.Concatenate()([pooling1,pooling2])
    model = K.Model(inputs = input_layer,
                    outputs = output_layer)
    model.summary()
#   加载 embedding_matrix 在实例化
embedding_matrix,word2id,id2word = load_embedding_data(config)
bilstm_attention = bilstm_attention_Model()
model = Sequential([
    L.Embedding(config['vocab_size'],
                config['embed_size'],
                weights = [embedding_matrix],
                trainable = False,
                input_length = config['max_len']
               ),
    bilstm_attention,
    L.Dense(64,activation = 'relu'),
    L.Dense(1,activation = 'sigmoid')
    ])

model.summary()

                
            
if __name__ == '__main__':
    config = {
        'epoch':10,
        'embed_size':256,
        'vocab_size':3196,
        'max_len':80,
        'mlp_units':64,
        'num_classes':96,
        'hidden_dim':512,
        'activation':'relu',
        'batch_size':16
        }
            
    # x = np.random.randint(0,1000,2000)
    # x = x.reshape((-1,config['max_len']))
    # y = np.random.randint(0,config['num_classes'],len(x))    
    # y_pre = model(x)

    # embedding_matrix,word2id,id2word = load_embedding_data(config)
    # model = Pretrain_biLstm_attention(config,
    #                                   embedding_matrix,
    #                                   is_embedding_matrix=True)
    # model(x)
    # embedding_layer = model.layers[0].get_weights()
    # print('隐藏层是否一致 ： ',embedding_layer[0]==embedding_matrix)
    







    