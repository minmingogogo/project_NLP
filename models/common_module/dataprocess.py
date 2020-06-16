# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:24:40 2020

@author: Scarlett
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os,sys
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split
import math
import re

'''
文本数据处理：
    1 字符增、删、改
    2 加分隔标记<start>,<end> <pad>等
    3 tokenizer(word2id,id2word)
    4 获得训练结构 input,target
'''
# =============================================================================
# dump / load data vocab
# =============================================================================


from tensorflow.keras.preprocessing.text import tokenizer_from_json

input_tokenizer_json_dir = os.path.join(output_dir,'input_tokenizer.json')
target_tokenizer_json_dir = os.path.join(output_dir,'target_tokenizer.json')
inp_text_dir = os.path.join(output_dir,'input_text.txt')
tar_text_dir = os.path.join(output_dir,'target_text.txt')
input_tensor_dir = os.path.join(output_dir,'input_data.txt')
target_tensor_dir = os.path.join(output_dir,'target_data.txt')

def data_dump():

    input_tokenizer_json = input_tokenizer.to_json()
    with open(input_tokenizer_json_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(input_tokenizer_json, ensure_ascii=False))
        
    output_tokenizer_json = output_tokenizer.to_json()
    with open(target_tokenizer_json_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(output_tokenizer_json, ensure_ascii=False))
    
    # with open(inp_text_dir,"w", encoding='utf-8') as f:
    #     f.writelines(text_inp)
    # with open(tar_text_dir,"w", encoding='utf-8') as f:
    #     f.writelines(text_tar)


    np.savetxt(inp_text_dir, text_inp,fmt="%s",delimiter=" ", encoding='utf-8')
    np.savetxt(tar_text_dir, text_tar,fmt="%s",delimiter=" ", encoding='utf-8')


    np.savetxt(input_tensor_dir, input_tensor,fmt="%d",delimiter=" ")
    np.savetxt(target_tensor_dir, output_tensor,fmt="%d",delimiter=" ")
    input_tensor.to_csv()
    
def data_load():    
    with open(input_tokenizer_json_dir) as f:
        data = json.load(f)
        input_tokenizer = tokenizer_from_json(data)    
    # input_tokenizer.index_word[6]
    with open(target_tokenizer_json_dir) as f:
        data = json.load(f)
        output_tokenizer = tokenizer_from_json(data)          
        

    input_data = np.loadtxt(input_tensor_dir)
    input_data = input_data.astype('int32')#  转回整型  
    target_data = np.loadtxt(target_tensor_dir)
    target_data = target_data.astype('int32')#  转回整型          
    
    #   str_list 用这种读方便
    input_text = []
    for line in open(inp_text_dir,"r", encoding='utf-8'): #设置文件对象并读取每一行文件
        input_text.append(line)           
    target_text = []
    for line in open(tar_text_dir,"r", encoding='utf-8'): #设置文件对象并读取每一行文件
        target_text.append(line)           
               




# =============================================================================
# 3 tokenizer & vocab
# =============================================================================
def tokenizer(split_seq_data,vocab_size =3000,lang_tokenizer=None):
    '''
    功能文本转id
    input : 
        split_seq_data:空格分割的seq_list
        lang_tokenizer: tokenizer 实例规则
        ------
        tf.keras.preprocessing.text.Tokenizer(
            num_words=None, 
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
            lower=True,split=' ', char_level=False,
            oov_token=None, document_count=0, **kwargs
            ) 
    output:
        tensor: 转为id 的seq_list
        lang_tokenizer ,token 与 id
    tokenizer 模式下vocabsize +1
    '''
    if not lang_tokenizer:
        lang_tokenizer = Tokenizer(num_words = vocab_size,oov_token = 'UNK')
        print('tokenizer msg :\n vocab_size {},oov: UNK '.format(vocab_size))
    lang_tokenizer.fit_on_texts(split_seq_data)
    #   转Id
    tensor = lang_tokenizer.texts_to_sequences(split_seq_data)
    #   padding = 'post' 在后面做padding
    return tensor,lang_tokenizer


# =============================================================================
# 2 添加标记
#   统一 max_len: pad or filter
def add_start_end(x,start='<start>',end='<end>'):
    '''
    x: array (:,seq_len),文本，未转id
             (seq_len,)
    -----example:
    x = ['abcde','defgh']
    tmp = add_start_end(x) 
        >Out: ['<start> abcde <end>', '<start> defgh <end>']
    x = [list(a) for a in x]
    tmp = add_start_end(x)    
        >Out: 
        [['<start>', 'a', 'b', 'c', 'd', 'e', '<end>'],
         ['<start>', 'd', 'e', 'f', 'g', 'h', '<end>']]
                             
    '''    
    inp_shape = len(np.array(x).shape)
    if inp_shape == 1:
        ''' 输入一维 未split 空格连接'''
        output = [start+' '+ subseq +' '+ end for subseq in x ]
    else:
        output = [[start] + subseq_list +[end] for subseq_list in x]
    return output
 

def padding(x,max_len,padding = 'post'):
    '''
    x: array/tensor
    '''
    tensor = pad_sequences(x,maxlen = max_len,padding = 'post')    
    return tensor


#   fillter 低于maxlength 的样本
def filter_by_max_length(x_seq,y_seq,max_len):
    '''只对两个输入
    x_seq:array
    y_seq:array
    '''
    return tf.logical_and(tf.size(x_seq)<=max_len,
                          tf.size(y_seq)<=max_len)

# =============================================================================
# 4 train_data:batch,shuffle,split_train_text
#   生成dataset

input_train,input_eval,output_train,output_eval = train_test_split(
    input_tensor,output_tensor,test_size = 0.2) 
   
def get_batch(inp_array,tar_array,batch_size):
    '''
    按 batch 分割
    '''
    num = int(math.ceil(len(tmp_x)/batch_size)*batch_size)
    tmp_x = tmp_x[:num]
    tmp_y = tmp_y[:num]
    dataset = tf.data.Dataset.from_tensor_slices((tmp_x, tmp_y))
    dataset = dataset.batch(batch_size)
    return dataset

def shuffle_all(shuffle_data):
    '''模型数据 shuffle 多任务的要一起shuffle
        a1 = np.array(range(8))
        b1 = np.array(range(10,18))
        b2 = np.array(range(20,28))
        shuffle_data = [a1,b1,b2]  
    '''
    for i in range(len(shuffle_data)):
        if i == 0:
            state = np.random.get_state()
            np.random.shuffle(shuffle_data[i])
        else:
            np.random.set_state(state)
            np.random.shuffle(shuffle_data[i]) 
    print(shuffle_data)
    return shuffle_data



def make_dataset(input_tensor,output_tensor,
                 batch_size,epochs,shuffle):
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor,output_tensor))
    if shuffle:
        dataset = dataset.shuffle(30000)
        #
    dataset = dataset.repeat(epochs).batch(batch_size,drop_remainder = True)    
    return dataset

# batch_size = 64
# epochs = 10
# train_dataset = make_dataset(input_train, output_train, batch_size, epochs, True)
# eval_dataset = make_dataset(input_eval, output_eval, batch_size, 1, False) 

# for x,y in train_dataset.take(1):
#     print('x shape : {} \n y shape : {} \n x : \n{}\n y : \n{}\n'.format(x.shape,y.shape,x,y))

# >x shape : (64, 16)
# >y shape : (64, 11)    





# =============================================================================
































    #   plan1   
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1' 
    #   plan2
# gpus= tf.config.experimental.list_physical_devices('GPU')
gpus= tf.config.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True) # 其实gpus本身就只有一个元素



datadir = os.path.join('D:\\myGit\Algorithm\\tensorflow2.0_example\\chapter_10\\data_spa_en\\data_spa_en')
en_spa_file_path = os.path.join(datadir,"spa.txt")

# =============================================================================
# 1 数据预处理
# =============================================================================
#   1.1 去掉注音
import unicodedata
def unicode_to_ascii(s):
    #   NFD :如果unicode 由多个asii 码组成则拆开，
    #   if unicodedata.category(c) != 'Mn' 过滤注音
    return ''.join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != 'Mn')
    
en_sentence = "I ran"
sp_sentence = "Corrí."

# print(unicode_to_ascii(en_sentence))
# print(unicode_to_ascii(sp_sentence))

#   1.2 去标点
import re
def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
     
    #   标点符号前后加空格
    s = re.sub(r"([?.!,¿])",r" \1 ",s)
    #   空格去重
    s = re.sub(r'[" "]+'," ",s)
    
    #   除标点和字母都是空格
    s = re.sub(r'[^a-zA-Z?.!,¿]'," ",s)
    #   去掉前后空格
    s = s.rstrip().strip()
    
    #   特殊字符操作
    s = '<start> ' + s + ' <end>'
    
    return s

# print(preprocess_sentence(en_sentence))
# print(preprocess_sentence(sp_sentence))


#    导入数据
def parse_data(filename):
    lines = open(filename,'r',encoding="UTF-8").read().strip().split('\n')
    #   讲每一行分为两部分，英文和西班牙语
    sentence_pairs = [line.split('\t') for line in lines]
    preprocess_sentence_pairs = [
        (preprocess_sentence(en),preprocess_sentence(sp)) for en,sp in sentence_pairs]
    return zip(*preprocess_sentence_pairs)


en_dataset,sp_dataset = parse_data(en_spa_file_path)

print(en_dataset[-1])
print(sp_dataset[-1])



'''
#   zip 解包功能例子
a = [(1,2),(3,4),(5,6)]
c,d = zip(*a)

    c
    Out[4]: (1, 3, 5)
    
    d
    Out[5]: (2, 4, 6)

'''

#   词语转id



def tokenizer(lang):
    lang_tokenizer = K.preprocessing.text.Tokenizer(
        #   不限制词表大小
        num_words = None,
        #   黑名单空
        filters='',
        #   分割符号
        split = ' ')
    #   fit_on_text 统计词频生成词表
    lang_tokenizer.fit_on_texts(lang)
    #   转Id
    tensor = lang_tokenizer.texts_to_sequences(lang)
    #   padding = 'post' 在后面做padding
    tensor = keras.preprocessing.sequence.pad_sequences(tensor,
                                                    padding = 'post')
    return tensor,lang_tokenizer

#   训练一个西班牙语到英语的转换# 测试前30000个效果
input_tensor ,input_tokenizer = tokenizer(sp_dataset[:30000])
output_tensor,output_tokenizer = tokenizer(en_dataset[:30000])

def max_length(tensor):
    return max(len(t) for t in tensor)

max_length_input = max_length(input_tensor)
max_length_output = max_length(output_tensor)
print(max_length_input,max_length_output)

from sklearn.model_selection import train_test_split

input_train,input_eval,output_train,output_eval = train_test_split(
    input_tensor,output_tensor,test_size = 0.2)


print(len(input_train),len(output_eval))


#   验证tokenizer
def convert(example,tokenizer):
    for t in example:
        if t != 0:
            print('%d --> %s'%(t,tokenizer.index_word[t]))
            
convert(input_train[0],input_tokenizer)
print(' ')
convert(output_train[0],output_tokenizer)

#   生成dataset
def make_dataset(input_tensor,output_tensor,
                 batch_size,epochs,shuffle):
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor,output_tensor))
    if shuffle:
        dataset = dataset.shuffle(30000)
        #
    dataset = dataset.repeat(epochs).batch(batch_size,drop_remainder = True)    
    return dataset

batch_size = 64
epochs = 10
train_dataset = make_dataset(input_train, output_train, batch_size, epochs, True)
eval_dataset = make_dataset(input_eval, output_eval, batch_size, 1, False) 

for x,y in train_dataset.take(1):
    print('x shape : {} \n y shape : {} \n x : \n{}\n y : \n{}\n'.format(x.shape,y.shape,x,y))

# >x shape : (64, 16)
# >y shape : (64, 11)

