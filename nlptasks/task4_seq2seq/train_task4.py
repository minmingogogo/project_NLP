# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:22:24 2020

@author: Scarlett

data: spanish_english

# step1 load data & config=============================================================================
# step2 load model=============================================================================
# step3 train & save model=============================================================================
# step4 evaluate & analyse =============================================================================

"""
import os
import re
import json
import tqdm
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from models.common_module.attention import Attention
from models.common_module.encoder import Encoder
from models.common_module.decoder import Decoder
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split

# =============================================================================
'''
cpu性能参数设置
'''
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 1
os.environ['KMP_BLOCKTIME'] = "1"
os.environ['KMP_SETTINGS'] = "1"
os.environ['KMP_AFFINITY'] = "granularity=fine,verbose,compat,1,0"
os.environ['OMP_NUM_THREADS'] = "8"
tf.compat.v1.Session(config=config)
# path =============================================================================

path = 'nlptasks//task4_seq2seq//translate_spanish_english'
datapath = os.path.join(path,'model_data')
if os.path.exists(datapath):
    print('data  exists')
else:
    print('data not exists')
analyse_dir = os.path.join(path,'analyse_dir')
if not os.path.exists(analyse_dir):
    os.makedirs(analyse_dir)    
    
    
input_tokenizer_json_dir = os.path.join(datapath,'input_tokenizer.json')
target_tokenizer_json_dir = os.path.join(datapath,'target_tokenizer.json')
inp_text_dir = os.path.join(datapath,'input_text.txt')
tar_text_dir = os.path.join(datapath,'target_text.txt')
input_tensor_dir = os.path.join(datapath,'input_data.txt')
target_tensor_dir = os.path.join(datapath,'target_data.txt')


# step1 load data & config =============================================================================
def data_load():    
    with open(input_tokenizer_json_dir) as f:
        data = json.load(f)
        input_tokenizer = tokenizer_from_json(data)    
    with open(target_tokenizer_json_dir) as f:
        data = json.load(f)
        output_tokenizer = tokenizer_from_json(data)          
    input_data = np.loadtxt(input_tensor_dir)
    input_data = input_data.astype('int32')#  转回整型  
    target_data = np.loadtxt(target_tensor_dir)
    target_data = target_data.astype('int32')#  转回整型          
    input_text = []
    for line in open(inp_text_dir,"r", encoding='utf-8'): #设置文件对象并读取每一行文件
        input_text.append(line)           
    target_text = []
    for line in open(tar_text_dir,"r", encoding='utf-8'): #设置文件对象并读取每一行文件
        target_text.append(line) 
        
# data_load()        
input_train,input_eval,target_train,target_eval = train_test_split(
    input_data,target_data,test_size = 0.3)
print(len(input_train),len(target_eval))

def max_length(tensor):
    return max(len(t) for t in tensor)
max_length_input = max_length(input_data)
max_length_output = max_length(target_data)
print('max_len:',max_length_input,max_length_output)
vocab_size_inp = len(input_tokenizer.word_counts)
vocab_size_tar = len(output_tokenizer.word_counts)
print('vocab_size:',vocab_size_inp+1,vocab_size_tar+1)
print('需要验证tokenizer最后一位')
print(input_tokenizer.index_word[vocab_size_inp])

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
    # 'num_heads':8,  #   num_heads 能被embed_size 整除  
    # 无关联参数配置
    'epoch':10,
    'batch_size':128,# 增大batch_size 可以提升GPU
    # embedding_layer:tokenizer 从1开始，vocab_size +1
    'vocab_size':17020,
    'tar_vocab_size':9556,   
    'max_len':49,
    'tar_max_len':49,
      
    # encoder_decoder_layer for multihead attention
    # 'num_decoder_layer':2,
    # 'num_encoder_layer':3,
    # 'ffn_units':64,
    # 'activation':'relu',
    # 'dropout':0.1,
    }  

def get_batch(inp_array,tar_array,batch_size):
    '''
    按 batch 分割
    '''
    batchs = math.ceil(len(inp_array)/batch_size)-1
    num = int(batchs*batch_size)
    inp_array = inp_array[:num]
    tar_array = tar_array[:num]
    dataset = tf.data.Dataset.from_tensor_slices((inp_array, tar_array))
    dataset = dataset.batch(batch_size)
    print('batchs : {}'.format(batchs))
    #   校验是否每个dataset batch 都与config 一致
    tag = 0
    ind = 0
    for i in dataset:
        ind+=1
        inp,tar = i
        if inp.shape[0]<config['batch_size']:
            tag+=1
            print('batch_size not equal in ind: ',ind)    
    print('not equal batch_size number : ',tag)
    return dataset,batchs

train_set,batchs = get_batch(input_train,target_train,config['batch_size'])


# step2 load model=============================================================================
from models.common_module.Model_seq2seq_attention import Seq2Seq_Attention,loss_function
from models.common_module.customized_optimizer import CustomizedSchedule
#   model & loss_func & optimizer
model = Seq2Seq_Attention(config)
learning_rate = CustomizedSchedule(config['embed_size'],warmup_steps = 100)
optimizer = K.optimizers.Adam(learning_rate,
                                  beta_1 = 0.9,
                                  beta_2 = 0.98,
                                  epsilon = 1e-9) 
loss_function

checkpoint_dir = os.path.join(path,'checkpoint')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)    
model_name = '/Model_seq2seq_attention'

def build_train_op(config):
    ''' model  & optimizer & ckpt & metrics '''
    #   1) model
    model = Seq2Seq_Attention(config)
    #   2) optimizer
    optimizer = K.optimizers.Adam(learning_rate,
                                  beta_1 = 0.9,
                                  beta_2 = 0.98,
                                  epsilon = 1e-9) 
    #   3) Checkpoint
    ckpt = tf.train.Checkpoint(model = model,optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir + model_name, max_to_keep=2)
    #恢复旧模型
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("creat new model....")
    #   4) metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_batch_loss = tf.keras.metrics.Mean(name='train_batch_loss')            
    return model, ckpt, ckpt_manager, optimizer,train_loss,train_batch_loss


# step3 train & save model=============================================================================

def train(train_set,train_ops):
    model, ckpt, ckpt_manager, optimizer,train_loss,train_batch_loss = train_ops
    @tf.function(experimental_relax_shapes=True)
    def train_step(input_ids,input_labels):
        with tf.GradientTape() as tape:
            loss,batch_loss,_= model(input_ids,input_labels,loss_function)
            # logits,loss = model(batch_data,batch_label,training = training)            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)#   loss 得到固定值后，多次调用train_loss(loss) ，会一直变；下面打印train_loss.result()平均损失而执行的实例操作
        train_batch_loss(batch_loss)#   loss 得到固定值后，多次调用train_loss(loss) ，会一直变；下面打印train_loss.result()平均损失而执行的实例操作

    train_loss.reset_states()
    train_batch_loss.reset_states()
    patience = 0
    last_loss = None  
    delta  =0.001    
    for index,batch in enumerate(train_set):
        # if index>1:
        #     break
        input_ids,input_labels = batch
        # batch_label = tf.cast(batch_label,dtype = batch_data.dtype)
        train_step(input_ids,input_labels)
        #-------early stop----------------
        if not last_loss:
            last_loss = train_loss.result()
        else:
            if abs(last_loss - train_loss.result())>delta:
                patience = 0
                last_loss = train_loss.result()
            else:
                patience +=1
            if patience >=5:
                # print('earlystop Epoch {} Loss : {:.4f}  Accuracy :{:.4f}'.format(epoch,train_loss.result(), train_batch_loss.result()))
                break
        #-------early stop----------------        
        # print('Epoch {} Loss : {:.4f}  batch_loss :{:.4f}'.format(epoch,train_loss.result(), train_batch_loss.result()))
        if index % 100 == 0 and index > 0:
            save_path = ckpt_manager.save()
            # print("Saved checkpoint {}".format(save_path))
    save_path = ckpt_manager.save()
    print("Saved checkpoint {}".format(save_path))            
    return train_loss.result(), train_batch_loss.result()

''' 训练'''  
# tf.compat.v1.enable_eager_execution()              
#开始训练数据
def train_step(analyse_dir):
    train_ops = build_train_op(config)    
    patience = 0
    last_loss = None  
    delta  =0.001    
    
    metrics = {'epoch':[],
               'loss':[],
               'batch_loss':[]}
    for epoch in tqdm.tqdm(range(config['epoch'])):
        # epoch = 0
        # train(train_set,train_ops)
        s = time.time()
        _loss, _batch_loss = train(train_set, train_ops)
        e = time.time()
        cost = (e-s)/60
        metrics['epoch'].append(epoch)
        metrics['loss'].append(_loss)
        metrics['batch_loss'].append(_batch_loss)
        print('Epoch {} Loss : {:.4f} BatchLoss :{:.4f} cost: {:.4f} mins'.format(epoch,_loss, _batch_loss,cost))
        #-------early stop----------------
        if not last_loss:
            last_loss = _loss
        else:
            if abs(last_loss - _loss) > delta:
                patience = 0
                last_loss = _loss
            else:
                patience +=1
            if patience >=5:
                print('earlystop ')
                break    
        #-------early stop----------------
    
    metrics['loss'] = [x.numpy() for x in metrics['loss']]
    metrics['batch_loss'] = [x.numpy() for x in metrics['batch_loss']]
    metrics_df = pd.DataFrame(metrics)
        
    metrics_dir = os.path.join(analyse_dir,'train_metrics_Model_seq2seq_attention.xlsx')
    metrics_df.to_excel(metrics_dir)

        

# WARNING: AutoGraph could not transform <bound method Attention.call of <models.common_module.attention.Attention object at 0x000002B303E4DE08>> and will run it as-is.
# Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
# Cause: 
# WARNING:tensorflow:AutoGraph could not transform <bound method Attention.call of <models.common_module.attention.Attention object at 0x000002B303E4DE08>> and will run it as-is.
# Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
# Cause: 

# step4 evaluate & analyse =============================================================================
# 评估指标 BLEU,jaccad
#       bleu score: n-gram 1-4 分数 + 权重调整分数 + 惩罚后分数
from models.common_module.Model_seq2seq_attention import Seq2Seq_Attention,loss_function
from analyse_func.bleu import bleu_v2_output
model_name = '/Model_seq2seq_attention'

def infer(config,input_eval,target_eval,input_tokenizer,output_tokenizer,analyse_dir):
    from models.common_module.Model_seq2seq_attention import Seq2Seq_Attention,loss_function
    from analyse_func.bleu import bleu_v2_output
    # restore model from ckpt
    model = Seq2Seq_Attention(config)
    #从训练的检查点恢复权重
    ckpt = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir + model_name)
    #添加expect_partial()关闭优化器相关节点warnning打印
    status = ckpt.restore(latest_ckpt).expect_partial()
    
    result_text = []
    result_df = pd.DataFrame()
    score_df =pd.DataFrame()
    for ind in tqdm.tqdm(range(len(input_eval))):
        # if ind>5:
        #     break
        if ind%1000 ==0:
            print(ind)
        try:
            inp = input_eval[ind]
            tar = target_eval[ind]
            inp_ids = [x for x in inp if x !=0]
            tar_ids = [x for x in tar if x !=0]
            inp_seq = [input_tokenizer.index_word[x] for x in inp_ids]
            tar_seq = [output_tokenizer.index_word[x] for x in tar_ids]
            inp_text = ' '.join(inp_seq)
            tar_text = ' '.join(tar_seq)
            inp = inp[tf.newaxis,:]
            out,_ = model.evaluate(inp,output_tokenizer)
            # print('{}\n{}\n{}\n--\n'.format(inp_text,out,tar_text))
            #   score : jaccad/bleu
            '''这部分要提到for 外处理，按批量跑out,加快速度'''
            train_sentences = ' '.join(tar_seq[1:-2])
            predict_sentence = re.sub('. <end> ','',out).strip()
            bleu_v2_score_list = bleu_v2_output(predict_sentence, train_sentences, 4, weights=[0.25, 0.25, 0.25, 0.25])
            # print('{}\n{}\n--\n'.format(train_sentences,predict_sentence))
            a = set(tar_seq[1:-2])
            b = set(re.sub('. <end> ','',out).strip().split(' '))
            inp_len = len(inp_seq)-3
            tar_len = len(a)
            out_len = len(b)
            jaccad = round(len(a & b)/ len(a|b),4)   
            # score_data = [[ind],[inp_len],[tar_len],[out_len],[jaccad]]
            
            score_data = [ind,inp_len,tar_len,out_len,jaccad]
            score_data.extend([list(a&b)])
            score_data.extend(bleu_v2_score_list)
            
            # score_data.extend([[x] for x in  bleu_v2_score_list])
            cols = ['index','inp_len','tar_len','out_len','jaccad','matchwords']
            blue_cols = ['bleu_1g','bleu_2g','bleu_3g','bleu_4g','bleu_with_weight','bleu_with_len_penalty']
            cols.extend(blue_cols)
            len(score_data) == len(cols)
            score_data = np.array(score_data).reshape(-1,1)
            score_data = pd.DataFrame(score_data.T)
            score_data.columns = cols        
        
            score_df = pd.concat([score_df,score_data])
            #   
            result_text.extend([str(ind),inp_text,out,tar_text,'--'])
            data = [[ind],[inp_ids],[tar_ids],[inp_seq],[tar_seq],[out]]
            data = np.array(data).T
            subdf = pd.DataFrame(data,columns = ['index','inp_ids','tar_ids','inp_seq','tar_seq','pre_seq'])
            result_df = pd.concat([result_df,subdf])
        except:
            continue
    result_text_dir = os.path.join(analyse_dir,'evaluate_Model_seq2seq_attention.txt')
    result_df_dir = os.path.join(analyse_dir,'evaluate_textlist_Model_seq2seq_attention.xlsx')
    score_df_dir = os.path.join(analyse_dir,'evaluate_score_Model_seq2seq_attention.xlsx')

    np.savetxt(result_text_dir, result_text,fmt="%s",delimiter=" ", encoding='utf-8')
    result_df.to_excel(result_df_dir)
    score_df.to_excel(score_df_dir)
# =============================================================================

# 多轮训练 =============================================================================
# from nltk.translate.bleu_score import corpus_bleu

for run_ind in range(2,7):
    analyse_dir = os.path.join(path,'analyse'+'_'+str(run_ind)+'run')
    if not os.path.exists(analyse_dir):
        os.makedirs(analyse_dir)     
    train_step(analyse_dir)    
    infer(config,input_eval,target_eval,input_tokenizer,output_tokenizer,analyse_dir)




