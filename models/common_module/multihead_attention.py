# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:47:33 2020

@author: Scarlett
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
#%% MultiHeadAttention

class MultiHeadAttention(L.Layer):
    ''' 多头注意力
        理论部分和实现有不一样
        理论部分：           实际中
        x -> Wq0 ->q0       q -> Wq0 ->q0
        x -> Wk0 ->k0       k -> Wk0 ->k0
        x -> Wv0 ->v0       k -> Wk0 ->k0  
        实践时,初始值不一定是一样的，因此输入x改成q\k\v
    '''
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        ''' 
        维度说明 ： batch_size : B ,max_len: M ,attention_units : U , num_heads : H                  
                  depth : D  =>  U/H = D 
        input:
            d_model : attention_units(隐藏层维度)
            d_model 必须是num_heads的倍数保证可均匀分割
       -------     
           d_model = 6,num_heads = 3
            |  h1  |   h2  |   h3  | 
            |v1 v2 | v3 v4 | v5 v6 |d_model
                    depth = 2
        '''
        self.num_heads = num_heads
        self.d_model = d_model
        ''' d_model 要均匀分成 num_heads,要整除 '''
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads #   depth 每份多少个向量
    
        #   定义三个全连接层做q,k,v 的转换Wq,Wk,Wv
        self.WQ = L.Dense(self.d_model)        
        self.WK = L.Dense(self.d_model)           
        self.WV = L.Dense(self.d_model)   

        #   拼接再经过全连接层(用于调整输出维度)
        self.dense = L.Dense(self.d_model)
        
    def split_heads(self,x):
        # x.shape (B,M,U)=>(B,M,H,U/H)=>(B,M,H,D)
        # 重排:(B,M,H,D) =[0,2,1,3] =>(B,H,M,D) 
        batch_size = x.shape[0]
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm = [0,2,1,3])
    
    def scaled_dot_product_attention(self,q,k,v,mask):
        '''
            缩放点积注意力
        Parameters
        ----------
        q\k\v : .shape (B,H,M,D) 
        mask : .shape == [...,seq_len_q,seq_len_k]
        compute:  attention_weights = softmax(((q*k_T)/(dk**0.5) + mask))
                  output = attention_weights*V
        Returns
        -------
        output:  (B,H,M,D) 与v 一致
        attention_weights:(为了可视化需要返回)weights of attention
    
        '''
        matmul_qk = tf.matmul(q,k,transpose_b = True) #(B,H,M,D) *(B,H,D,M)
        dk = tf.cast(tf.shape(k)[-1],tf.float32)
        scaled_attention_logits = matmul_qk/tf.math.sqrt(dk) #(B,H,M,M) 
        
        if mask is not None:
            # mask 中凡是要弃除的都是1，保留的都是0,使在softmax后的值趋近于0
            scaled_attention_logits += (mask * -1e9)        
        #   attention_weights : (B,H,M,M) 
        #   output : (B,H,M,D)
        attention_weights = tf.nn.softmax(
            scaled_attention_logits,axis =-1) #(B,H,M,M) 
        output = tf.matmul(attention_weights,v) #(B,H,M,M) * (B,H,M,D)
        return output,attention_weights

    def call(self,q,k,v,mask):
        #   线性变换 
        q = self.WQ(q)      #q.shape:(B,M,U)
        k = self.WK(k)      #k.shape:(B,M,U)
        v = self.WV(v)      #v.shape:(B,M,U)
        
        #   按多头split 一个维度 , q\k\v.shape:(B,H,M,D)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        #   scaled_attention_outputs : (B,H,M,D)
        #   attention_weights : (B,H,M,M) 
        scaled_attention_outputs ,attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        #   多头信息存在第二维和第四维上(B,H,M,D) 
        #   重排调整回原来顺序( B,H,M,D) =[0,2,1,3]=> (B,M,H,D) 
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs,perm = [0,2,1,3])
        #   合并多头信息 : (B,M,H*D) = (B,M,U)
        batch_size = scaled_attention_outputs.shape[0]
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size,-1,self.d_model))
        #   调整输出维度output_dim : O => (B,M,O)，维度一致则不变(B,M,U)
        output = self.dense(concat_attention)
        
        return output,attention_weights    
    

if __name__ == '__main__':

    from common_module.create_masks import Create_masks
    config = {
        'epoch':10,
        'embed_size':256,
        'vocab_size':3196,
        'max_len':16,
        'tar_embed_size':256,
        'tar_vocab_size':1000,   
        'tar_max_len':16,
        # 'mlp_units':64,
        # 'num_classes':96,
        # 'hidden_dim':512,
        # 'activation':'relu',
        'batch_size':16,
        'units':64,
        'attention_units':64,
        'num_heads':8,
        }     
    
    inp = np.random.randint(0,1000,2000).reshape((-1,config['max_len']))
    tar = np.random.randint(0,200,2000).reshape((-1,config['tar_max_len']))
    emb_inp = L.Embedding(config['vocab_size'],config['embed_size'])
    emb_tar = L.Embedding(config['tar_vocab_size'],config['tar_embed_size'])
    inp_emb = emb_inp(inp)
    tar_emb = emb_tar(tar)
    
    
    cmask = Create_masks()
    '''mask 输入是 2D '''
    encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask = cmask.create_masks(inp,tar)
    
    '''mha 输入是 3D '''
    mha = MultiHeadAttention(config['attention_units'],config['num_heads'])
    #   无mask
    output,attention_weights  = mha(inp_emb,inp_emb,inp_emb,None)
    print(output.shape,attention_weights.shape)
    #(125, 16, 64) (125, 8, 16, 16)
    #   有 padding mask 
    output,attention_weights  = mha(inp_emb,inp_emb,inp_emb,encoder_padding_mask)
    print(output.shape,attention_weights.shape)
    #(125, 16, 64) (125, 8, 16, 16)
    #   有 lookahead mask 
    output,attention_weights  = mha(inp_emb,tar_emb,tar_emb,decoder_mask)
    print(output.shape,attention_weights.shape)
    #(125, 16, 64) (125, 8, 16, 16)
        