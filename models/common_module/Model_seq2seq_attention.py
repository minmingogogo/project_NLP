# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:01:25 2020

@author: Scarlett
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from models.common_module.attention import Attention
from models.common_module.encoder import Encoder
from models.common_module.decoder import Decoder


class Seq2Seq_Attention(K.Model):
    def __init__(self,config):
        super(Seq2Seq_Attention,self).__init__()   
        '''
        数据要求：
        input/target seq 要有开始结束符:
        <star> I have a cat <end>
        '''        
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.attention = Attention(config)
    def call(self,inp,tar,loss_function):
        #   初始化状态变量
        encoding_hidden = self.encoder.initialize_hidden_state()
        encoding_outputs,encoding_hidden = self.encoder(inp,encoding_hidden)
        decoding_hidden = encoding_hidden
        # decoding_output,decoding_hidden,_ = self.decoder(tar,encoding_outputs,decoding_hidden)      
        #   tar:target
        loss = 0
        batch_loss = 0
        output = []
        for t in range(0,tar.shape[1]-1):
            ''' token逐个预测 ，需要逐个输入计算 loss
            @tf.function 模式 下无法获得tensor.numpy()
            output 建议以tensor 形式输出
            如果output 要用numpy 函数则需要通过tf.py_function 将np 函数封装
            '''
            decoding_input = tf.expand_dims(tar[:,t],1)
            # decoding_output.shape(B,E)
            # decoding_hidden.shape(B,U)
            # attention_weight.shape (B,M,1)
            decoding_output,decoding_hidden,_ = self.decoder(
                decoding_input,encoding_outputs,decoding_hidden)
            # weight = self.decoder.embedding.get_weights()[0]
            tar_ids = tar[:,t+1]
            y_pre = decoding_output 
            # real:(batch_size,1) 
            # y_pre:(batch_size,vocab_size)
            loss += loss_function(tar_ids, y_pre) 
            #   output
            if len(output) == 0:
                output = tar_ids[:,tf.newaxis]  
            else:
                #上下拼接，输出时转置为正常顺序                
                output =tf.concat([output,tar_ids[:,tf.newaxis]],axis = 1)               
        batch_loss = loss/int(tar.shape[0])     
        return loss,batch_loss,output
# a = output[:,tf.newaxis]        
# b = tar_ids[:,tf.newaxis]   
# tmp = tf.concat([a,b],axis = 1)   
    
    def evaluate(self,inp,output_tokenizer):
        ''' 单条输入
        图模式
        '''
        results = ''
        attention_matrix = np.zeros((self.config['tar_max_len'],
                                     self.config['tar_max_len']))
        encoding_hidden = tf.zeros((1,self.config['encoding_units']))  
        encoding_outputs,encoding_hidden = self.encoder(inp,encoding_hidden)
        #   初始化输入
        decoding_input = tf.expand_dims(
            [output_tokenizer.word_index['<start>']],0)
        decoding_hidden = encoding_hidden
        for t in range(self.config['tar_max_len']):
            ''' token逐个预测 ，需要逐个输入计算 loss '''
            decoding_output,decoding_hidden,attention_weights = self.decoder(
                decoding_input,encoding_outputs,decoding_hidden)

        #   attention_weights.shape : [1,input_length,1]
            # (1,16,1)
            attention_weights = tf.reshape(attention_weights,(-1,))
            attention_matrix[t] = attention_weights.numpy()
            #   prediction.shape :(batch_size,vocab_size) :(1,4935)
            predicted_id = tf.argmax(decoding_output[0].numpy()).numpy()
            results += output_tokenizer.index_word[predicted_id] +' '
            if output_tokenizer.index_word[predicted_id] == '<end>':
                return results,attention_matrix
            #   更新decoding_input
            decoding_input = tf.expand_dims([predicted_id],0)
        return results,attention_matrix      

optimizer = K.optimizers.Adam()

def loss_function(real,pred):
    '''
    input:
        real:(batch_size,1)              [15,30,23...]
        pred:(batch_size,vocab_size)    [[0.1,0.2..],[0.002,0.0003,...]]
    mask padding 处理：
        把不应该参与到计算的padding 的损失函数去掉，设为0
        tf.math.logical_not 取反操作后：
            非padding部分都设为1,padding的部分都是0
            tf.math.equal(real,0) 0的部分为1 ，不是0的部分为0，需要取反
    loss_object 选择：
        预测id -分类问题> 交叉熵损失
          如果目标是onehot 编码则是CategoricalCrossentropy
          如果目标是数字编码则用SparseCategoricalCrossentropy        
          需要返回每项的 loss-> mask 过滤 -> new loss
    '''

    loss_object = K.losses.SparseCategoricalCrossentropy(
        from_logits = True,
        reduction = 'none' )

    # loss_object： 
    #     一个batch 每个预测值的loss，
    #     from_logits = True, 输出不经过softmax
    #     reduction = 'none'
    # https://tensorflow.google.cn/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy?hl=en
    

    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    #   mask 类型变换
    mask = tf.cast(mask,dtype = loss_.dtype)
    #   相乘后 padding 部分都不算在loss 中
    loss_ *= mask
    #   乘完mask 后再聚合tf.reduce_mean,这也是loss_object 中reduction ="None"的原因
    return tf.reduce_mean(loss_)




if __name__ == '__main__':
        
    config = {
        # 关联参数配置
        #   embed_size 与 attention_units ，tar_embed_size 三个相等
        'embed_size':256,# 需要等于 attention_units 残差连接 
        'tar_embed_size':256, # 
        #   rnn_layer
        'decoding_units':256,
        'encoding_units':256,# 需要等于 tar_embed_size  才能拼接  
        #   attention_layer        
        'attention_units':256,# 需要等于 embed_size 残差连接 
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
    model = Seq2Seq_Attention(config)

    
    
    
    
    
    inp = np.random.randint(0,1000,
                            config['batch_size']*config['max_len']
                            ).reshape((-1,config['max_len']))
    tar = np.random.randint(0,200,
                            config['batch_size']*config['tar_max_len']
                            ).reshape((-1,config['tar_max_len']))            
            
         


# encoding_hidden = encoder.initialize_hidden_state()
# encoding_outputs,encoding_hidden = encoder(inp,encoding_hidden)
# decoding_hidden = encoding_hidden
# # decoding_output,decoding_hidden,_ = self.decoder(tar,encoding_outputs,decoding_hidden)      
# #   tar:target
# loss = 0
# for t in range(0,tar.shape[1]-1):
#     t = 0
#     ''' token逐个预测 ，需要逐个输入计算 loss'''
#     decoding_input = tf.expand_dims(tar[:,t],1)
#     # decoding_output.shape(B,E)
#     # decoding_hidden.shape(B,U)
#     # attention_weight 没用到 用_ 替代
#     print(decoding_input.shape,encoding_outputs.shape,decoding_hidden.shape)
#     decoding_output,decoding_hidden,_ = decoder(
#         decoding_input,encoding_outputs,decoding_hidden)
#     print(decoding_output.shape,decoding_hidden.shape,)
    
#     loss += loss_function(tar[:,t+1], decoding_output)     
#     weight = decoder.embedding.get_weights()[0]
#     tar_ids = tar[:,t+1]
#     y_true = weight[tar_ids]
#     y_pre = decoding_output 
    
    
#     print(y_true.shape,y_pre.shape,)
#     loss += loss_function(y_true, y_pre)        
#     batch_loss = loss/int(tar.shape[0])
#     predicted_id = tf.argmax(predictions[0].numpy())


#     config = {
#         # 关联参数配置
#         #   embed_size 与 attention_units ，tar_embed_size 三个相等
#         'embed_size':256,# 需要等于 attention_units 残差连接 
#         'tar_embed_size':256, # 
#         #   rnn_layer
#         'decoding_units':256,
#         'encoding_units':256,# 需要等于 tar_embed_size  才能拼接  
#         #   attention_layer        
#         'attention_units':256,# 需要等于 embed_size 残差连接 
#         'num_heads':8,  #   num_heads 能被embed_size 整除
      
#         # 无关联参数配置
#         'epoch':10,
#         'batch_size':16,
#         # embedding_layer
#         'vocab_size':3196,
#         'max_len':40,
#         'tar_max_len':20,
#         'tar_vocab_size':500,   
          
#         # encoder_decoder_layer
#         'num_decoder_layer':2,
#         'num_encoder_layer':3,
#         'ffn_units':64,
#         'activation':'relu',
#         'dropout':0.1,
#         }  
# inp = np.random.randint(0,1000,
#                         config['batch_size']*config['max_len']
#                         ).reshape((-1,config['max_len']))
# tar = np.random.randint(0,200,
#                         config['batch_size']*config['tar_max_len']
#                         ).reshape((-1,config['tar_max_len']))    




