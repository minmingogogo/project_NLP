# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:03:38 2020

@author: Scarlett
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from models.common_module.CRF_Layer import CRF


class BiLSTM_CRF(L.Layer):    
    def __init__(self,config):
        super(BiLSTM_CRF,self).__init__()
        self.config = config
        self.embedding = L.Embedding(config['vocab_size'],
                                     config['embed_size'])  
        
        self.bilstm1 = L.Bidirectional(L.LSTM(config['hidden_units'],
                                             return_sequences = True,
                                             activation="tanh"),
                                             merge_mode='sum')
        self.bilstm2 = L.Bidirectional(L.LSTM(config['hidden_units'],
                                             return_sequences = True,
                                             activation="softmax"),
                                             merge_mode='sum')  # merge_mode='sum' 前向后向加和              
        self.crf = CRF(config['num_classes'],name = 'crf_layer')

    def call(self,inp,training=None):
        x = self.embedding(inp)
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        out = self.crf(x)
        return out
        

if __name__ == '__main__':
    
    config = {
        # 无关联参数配置
        'epoch':10,
        'batch_size':16,
        'embed_size':256,# 需要等于 attention_units 残差连接 
        # embedding_layer
        'vocab_size':3196,
        'max_len':40,
        'num_classes':18,
        'hidden_units':512
            }
        
    model = BiLSTM_CRF(config)


        