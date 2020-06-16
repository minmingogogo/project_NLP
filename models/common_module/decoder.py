# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 00:47:08 2020

@author: Scarlett
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from models.common_module.attention import Attention
from models.common_module.encoder import Encoder

#%% decoder for lstm    
class Decoder(K.Model):
    def __init__(self,config):
        super(Decoder,self).__init__()
        self.embedding = L.Embedding(config['tar_vocab_size'],
                                     config['tar_embed_size'])
        self.gru = L.GRU(config['decoding_units'],
                        return_sequences = True,#输出所有状态
                        return_state= True, #   输出最后状态
                        recurrent_initializer = 'glorot_uniform') 
        #   输出成某个词
        self.fc = L.Dense(config['tar_vocab_size'])
        self.attention = Attention(config)
    def call(self,x,encoding_outputs,hidden):
        ''' decoder 每次生成一个词 
        x.shape (B,1)
        encoding_outputs.shape
        hidden.shape
        '''
        # encoding_outputs.shape (B,M,U) ,hidden.shape (B,1,U)
        # context_vector.shape:(B,1,U) ,attention_weights (B,M,O) =>(B,M,1)
        context_vector,attention_weights = self.attention(
            encoding_outputs,hidden)
        #   x.shape :(B,1,E)
        x = self.embedding(x)        
        try : 
            # U=E :context_vector.shape (B,1,E), x.shape (B,1,E)
            # combined_x.shape :(B,1,2E)
            combined_x = tf.concat([context_vector,x],axis = -1)
        except Exception as e:
            print('context_vector.shape:{}\n x.shape:{}'.format(context_vector.shape,x.shape))
            print('e :{}'.format(e))
            
        #   output.shape:(batch_size,1,decoding_units)
        #   state.shape:(batch_size,decoding_units)
        output,state = self.gru(combined_x)
        
        #   output.shape : (batch_size,decoding_untis)
        output = tf.reshape(output,(-1,output.shape[2]))
        
        #   output.shape  : [batch_size,vocab_size] 映射到词表
        output = self.fc(output)
        
        return output,state,attention_weights
        
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
    tar = np.random.randint(0,200,
                            config['batch_size']).reshape((-1,1))    
    # inp.shape:(B,M)
    # tar.shape:(B,1) 一个一个token预测
    encoder = Encoder(config)
    hidden = encoder.initialize_hidden_state()
    encoding_outputs,encoding_hidden = encoder(inp,hidden)    
    # 预测第一个词时，后面decoding_hidden由decoder 产生
    decoding_hidden = encoding_hidden 
    decoder = Decoder(config) 
    decoding_output,decoding_hidden,_ = decoder(tar,encoding_outputs,decoding_hidden)      
  

#%% decoder for transformer/bert
from models.common_module.position_embedding import sin_cos_pos_embedding
from models.common_module.multihead_attention import MultiHeadAttention
from models.common_module.create_masks import Create_masks
from models.common_module.encoder import EncoderModel

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

class DecoderLayer(L.Layer):
    '''
    网络结构
    block:
        x                      -> self attention -> add & normalize & dropout-> out1
        out1 ,encoding_outputs -> attention      -> add & normalize & dropout-> out2 
        out2                   -> ffn            -> add & normalize & dropout-> out3
    '''    
    
    def __init__(self,config):
        super(DecoderLayer,self).__init__()
        d_model = config['attention_units']
        num_heads = config['num_heads']
        dff = config['ffn_units']
        rate = config['dropout']                
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)# 给输入做self attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)# encoder decoder之间做attention
        
        self.ffn = feed_forward_network(dff,d_model)
        
        self.layer_norm1 = L.LayerNormalization(epsilon = 1e-6)
        self.layer_norm2 = L.LayerNormalization(epsilon = 1e-6)
        self.layer_norm3 = L.LayerNormalization(epsilon = 1e-6)
        
        self.dropout1 = L.Dropout(rate)
        self.dropout2 = L.Dropout(rate)
        self.dropout3 = L.Dropout(rate)

        
    def call(self,x,encoding_outputs,training,decoder_mask,encoder_decoder_padding_mask):
        # x.shape :(B,M,U) 
        # encoding_outputs.shape:(B,M,U) #d_model 用encoder 的d_model
        # decoder_mask:由look_ahead_mask 和 decoder_padding_mask 合并得到
        # encoder_decoder_padding_mask 用在encoder decoder 之间
        
        # attn1.shape:(batch_size,tartget_seq_len,d_model)
        # attn_weights1.shape (B,H,M_d,M_d) 
        attn1, attn_weights1 = self.mha1(x,x,x,decoder_mask)
        attn1 = self.dropout1(attn1,training=training)
        #   out1.shape(B,M,E)
        out1 = self.layer_norm1(attn1 + x)
        
        # attn2.shape:(batch_size,tartget_seq_len,d_model)
        # attn_weights1.shape (B,H,M_d,M_e) 
        attn2, attn_weights2  = self.mha2(
            out1,encoding_outputs,encoding_outputs,
            encoder_decoder_padding_mask)
        attn2 = self.dropout2(attn2,training = training)
        out2 = self.layer_norm2(attn2 + out1)


        #ffn_output.shape:(batch_size,tartget_seq_len,d_model)        
        ffn_output = self.ffn(out2) #(B,M_d,E)
        ffn_output = self.dropout3(ffn_output,training = training)
        out3 = self.layer_norm3(ffn_output + out2)
        
        return out3,attn_weights1,attn_weights2
    

    
        

class DecoderModel(L.Layer):
    '''
    x -> Emb (word_emb,pos_emb) -> dropout -> decoder_layers-> output
    
    '''
    
    def __init__(self,config):
        super(DecoderModel,self).__init__()
        self.d_model = config['attention_units'] # 与 embed_size 一致才能残差连接
        self.max_length = config['tar_max_len']
        self.num_layers = config['num_decoder_layer']
        rate = config['dropout']               
        self.embedding = L.Embedding(config['tar_vocab_size'],
                                     config['tar_embed_size'])
        self.position_embedding = sin_cos_pos_embedding(config['tar_embed_size'],
                                                        config['tar_max_len'])         
        
        self.dropout = L.Dropout(rate)
        self.decoder_layers = [
            DecoderLayer(config)
            for _ in range(self.num_layers)]
        
    def call(self,x,encoding_outputs,training,decoder_mask,encoder_decoder_padding_mask):
        # x.shape:(B,M)
        output_seq_len = tf.shape(x)[1]
        # # assert output_seq_len <= self.max_length
        tf.debugging.assert_less_equal(
            output_seq_len,self.max_length,
            'output_seq_len should be less or equal to self.max_length')
        #   attention_weights : 由decoder layers 返回得到
        attention_weights = {}
        
        #   x.shape:(batch_size,output_seq_len,d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        pos_emb = tf.cast(self.position_embedding,dtype = x.dtype)
        
        x += pos_emb
        
        x = self.dropout(x,training = training)
        
        for i in range(self.num_layers):
            x,att1,att2 = self.decoder_layers[i](
                x,encoding_outputs,training,
                decoder_mask,encoder_decoder_padding_mask)
            attention_weights['decoder_layer{}_att1'.format(i+1)] = att1
            attention_weights['decoder_layer{}_att2'.format(i+1)] = att2
        #   x.shape:(batch_size,output_seq_len,d_model)
        return x,attention_weights
    
if __name__ == '__main__':

    config = {
        # 关联参数配置
        #   embed_size 与 attention_units ，tar_embed_size 三个相等
        'embed_size':256,# 需要等于 attention_units 残差连接 
        'attention_units':256,# 需要等于 embed_size 残差连接 
        'tar_embed_size':256, # 
        'num_heads':8,  #   num_heads 能被embed_size 整除
        # 无关联参数配置
        'epoch':10,
        'batch_size':16,
        # embedding_layer
        'vocab_size':3196,
        'max_len':40,
        'tar_max_len':20,
        'tar_vocab_size':500,   
        # encoder_decoder_layer
        'num_decoder_layer':2,
        'num_encoder_layer':3,
        'ffn_units':64,
        'activation':'relu',
        'dropout':0.1,
        }   

    cmask = Create_masks()
    encoder = EncoderModel(config)
    inp = np.random.randint(0,1000,
                            config['batch_size']*config['max_len']
                            ).reshape((-1,config['max_len']))
    tar = np.random.randint(0,200,
                            config['batch_size']*config['tar_max_len']
                            ).reshape((-1,config['tar_max_len']))  
    #   get mask
    encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask = cmask.create_masks(inp,tar)
    #   get encoder_output
    encoding_outputs = encoder(inp,1,None)
# =============================================================================
#  测试 decoder_layer 
    decoder_layer = DecoderLayer(config)
    tar_emb = L.Embedding(config['tar_vocab_size'],
                          config['tar_embed_size'])(tar)    
    #  无 mask
    outputs = decoder_layer(tar_emb,encoding_outputs,1,None,None)
    out3,attn_weights1,attn_weights2 = outputs
    #  有 mask
    outputs = decoder_layer(tar_emb,encoding_outputs,1,decoder_mask,encoder_decoder_padding_mask)    
    out3,attn_weights1,attn_weights2 = outputs
# =============================================================================
#  测试 DecoderModel 
    decoder = DecoderModel(config)  
    #  无 mask
    outputs = decoder(tar,encoding_outputs,1,None,None)
    decoding_outputs,attn_weights = outputs
    #  有 mask
    outputs = decoder(tar,encoding_outputs,1,decoder_mask,encoder_decoder_padding_mask)    
    decoding_outputs,attn_weights = outputs
# =============================================================================
    

    