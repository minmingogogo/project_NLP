# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:32:34 2020

@author: Scarlett
"""

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from models.common_module.CRF_Layer import CRF
from models.common_module.transformer import transformer



class Ner(K.Model):
    """docstring for ner_model"""
    def __init__(self, config):
        super(Ner, self).__init__()
        self.config = config
        self.tf_layer = transformer(config)        
        self.bi_lstm = L.Bidirectional(L.LSTM(config['rnn_unit'], 
                                    return_sequences=True,
                                    return_state=False))        
        self.dropout = L.Dropout(config['rnn_dropout'])  

        self.dense_layer = L.Dense(config['tgt_size'])
        self.crf = CRF(config['tgt_size'],name = 'crf_layer')
                
    def call(self,seq_input,label_input=None,training=1,detail=False):
        output = self.tf_layer(seq_input,training)
        # output.shape (B,M,U)
        
        if self.config['rnn']:
            ''' 是否增加 bi-lstm block'''
            output = self.bi_lstm(output)
            if training:
                output = self.dropout(output)
        output = self.dense_layer(output)
        out = self.crf(output)
        
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
        #   bert
        'tgt_size':18,
        'num_layers':4,
        'head':8,
        'ffw_rate':4,
        'attention_dropout':0.2,
        'layer_norm_epsilon':1e-5,    
        #   lstm
        # 'hidden_units':512,
        'rnn':False,
        'rnn_unit':256,
        'rnn_dropout':0.2,        
            }
        
    model = Ner(config)    
    
 