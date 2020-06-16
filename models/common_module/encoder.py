# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:04:51 2020

@author: Scarlett
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

#%% Encoder for lstm    
class Encoder(K.Model):
    '''
    lstm encoder
    '''
    def __init__(self,config):
    # encoding_units ： encoder 中 循环神经网络的size 多大
        super(Encoder,self).__init__()
        self.batch_size = config['batch_size']
        self.encoding_units = config['encoding_units']
        self.embedding = L.Embedding(config['vocab_size'],
                                     config['embed_size'])
        self.gru = L.GRU(config['encoding_units'],
                        return_sequences = True,#输出所有状态
                        return_state= True, #   输出最后状态
                        recurrent_initializer = 'glorot_uniform')       
    def call(self,x,hidden):
        '''
        input.shape
            x :(B,M)
            hidden : (B,U) 首次为初始化0隐含状态/后续回自动更新调用
        output.shape
            output:all_hidden_state : (B,M,U)
            state:last_hidden_state : (B,1,U)
        '''
        x = self.embedding(x) # (B,M) =emb=> (B,M,E) 
        output,state = self.gru(x,initial_state = hidden)
        return output,state
    
    def initialize_hidden_state(self):
        #  output: (M,U)创建全是零的隐含状态，传给call
        return tf.zeros((self.batch_size,self.encoding_units))

if __name__ == '__main__':

    config = {
        'epoch':10,
        'embed_size':256,# 需要等于 encoding_units 才能拼接  
        'vocab_size':3196,
        'max_len':40,
        'batch_size':16,
        # 'num_classes':96,
        # #   target
        'tar_max_len':20,
        'tar_embed_size':256, #   
        'tar_vocab_size':500, #   
        #   rnn 
        'decoding_units':80,
        'encoding_units':256,# 需要等于 tar_embed_size  才能拼接  
        'num_layers':2,
        # 'mlp_units':64,
        'attention_units':128,# 线性变换
        'activation':'relu',
        'dropout':0.5,
        
        }   

    inp = np.random.randint(0,1000,
                            config['batch_size']*config['max_len']
                            ).reshape((-1,config['max_len']))
    tar = np.random.randint(0,200,2000).reshape((-1,config['tar_max_len']))    
    
    encoder = Encoder(config)
    hidden = encoder.initialize_hidden_state()
    output,state = encoder(inp,hidden)

    


#%% Encoder for transformer/bert
from models.common_module.multihead_attention import MultiHeadAttention
from models.common_module.position_embedding import sin_cos_pos_embedding
from models.common_module.create_masks import Create_masks


def feed_forward_network(dff,d_model):
    '''
    feedforward network 层 EncoderLayer,DecoderLayer 中都被用到
    d_model = 128,dff = 512
    '''
    
    #   dff : dim of feed forward network
    #   两层全连接网络结构，dff 是第一层，d_model 是第二层
    model = K.Sequential([
                    L.Dense(dff,activation = 'relu'),
                    L.Dense(d_model)])             
    return model

class EncoderLayer(L.Layer):
    '''
    网络结构
    block:
        x    ->   self attention -> dropout & add & normalize -> out1
        out1 ->   feed_forward   -> dropout & add & normalize -> out2
    '''
    def __init__(self,config):

        super(EncoderLayer,self).__init__()
        d_model = config['attention_units']
        num_heads = config['num_heads']
        dff = config['ffn_units']
        rate = config['dropout']
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(dff,d_model)
        
        self.layer_norm1 = L.LayerNormalization(epsilon = 1e-6)
        self.layer_norm2 = L.LayerNormalization(epsilon = 1e-6)
        
        self.dropout1 = L.Dropout(rate)
        self.dropout2 = L.Dropout(rate)
        
    def call(self, x,training,encoder_padding_mask):
        ''' add & normalize(layer normalize) & dropout '''        
        # x.shape:(B,M,E) 
        # attn_output.shape:(B,M,U)
        attn_output, _ = self.mha(x,x,x,encoder_padding_mask)
        # training：训练模式dropout,还是在推理模式下不dropout。
        attn_output = self.dropout1(attn_output,training = training)
        
        ''' 残差连接 '''
        out1 = self.layer_norm1(x + attn_output)
        #  ffn_output.shape : (B,M,U)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output,training = training)
        # out2.shape : (B,M,U)
        ''' 残差连接 '''
        out2 = self.layer_norm2(out1 + ffn_output)        
        return out2
    

class EncoderModel(L.Layer):
    '''
    x -> Emb (word_emb,pos_emb) -> dropout -> encoder_layers-> output
    '''
    def __init__(self,config):

        super(EncoderModel,self).__init__()        
        self.d_model = config['attention_units']
        self.max_length = config['max_len']
        self.num_layers = config['num_encoder_layer']
        rate = config['dropout']
        self.embedding = L.Embedding(config['vocab_size'],
                                     config['embed_size'])

        self.position_embedding = sin_cos_pos_embedding(config['embed_size'],
                                                        config['max_len'])        
        self.dropout = L.Dropout(rate)
        #   初始化 num_layers 个 numberlayer
        self.encoder_layers = [EncoderLayer(config)
                                for _ in range(config['num_encoder_layer'])]        
      
    def call(self,x,training,encoder_padding_mask):
        '''
        input:
            x (B,M)
        output :
            (B,M,U)
        '''
        #   1 校验 max_len 维度与config参数是否一致
        # input_seq_len = tf.shape(x)[1]
        # tf.debugging.assert_less_equal(
        #     input_seq_len, self.max_length,
        #     'input_seq_len should be less or equal to self.max_length') 
        #   2 model
        x = self.embedding(x) #(B,M,E)
        ''' 
        pos_emb
        #   x + position_embedding前,x 要缩放 dk**0.5
        原因：
        #   x 经过embedding 初始化时是从0-1之间均匀分布取到，缩放使得x 范围是0-d_model
        #乘以根号d_model,后面再multihead 再除回去 先放再缩
        '''
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        pos_emb = tf.cast(self.position_embedding,dtype = x.dtype)
        # pos_emb.shape(1,M,E),
            # 第一维度相加时候回自动复制batch_size 倍
            # pos_emb.dtype 必须等于x.dtype 否则不能相加
        x += pos_emb
        x = self.dropout(x,training=training)       
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x,training,encoder_padding_mask)            
            #x.shape:(batch_size,input_seq_len,d_model)
        return x

    
if __name__ == '__main__':    
    config = {
        #   train
        'epoch':10,
        'batch_size':16,
        #   input
        'max_len':16,
        #   embedding
        'vocab_size':3196,        
        'embed_size':256, # 要实现残差连接必须保证 等于 attention_units
        #   target
        'tar_max_len':20,
        'tar_embed_size':128, # 可以等于 attention_units
        'tar_vocab_size':3196,        
        #   attention 
        'attention_units':256,#要实现残差连接必须保证 等于embed_size
        'num_heads':8,
        #   feed_forward_network
        'ffn_units':512,
        #   norm & dropout
        'activation':'relu',
        'dropout':0.5,
        'num_encoder_layer':3,
        }   

    inp = np.random.randint(0,1000,
                            config['batch_size']*config['max_len']
                            ).reshape((-1,config['max_len']))
    tar = np.random.randint(0,200,
                            config['batch_size']*config['tar_max_len']
                            ).reshape((-1,config['tar_max_len']))    
    
    encoder = EncoderModel(config)
    encoder_output = encoder(inp,1,None)
    
    print(encoder_output.shape)

