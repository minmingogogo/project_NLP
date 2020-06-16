# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:07:25 2020

@author: Scarlett
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")        
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
     


VOCAB_SIZE = 20000
EMBEDDING_DIM = 300 #    有预训练与预训练一致


corpus_dir = 'D:\\myGit\\Algorithm\\A_my_nlp_project\\pre_train_model\\word2vec\\embeddings'
readfile = os.path.join(corpus_dir ,'sgns.zhihu.word.bz2')
writefile = os.path.join(corpus_dir ,'sgns.zhihu.word')

with open(writefile, 'wb') as new_file, open(readfile, 'rb') as file:
    decompressor = bz2.BZ2Decompressor()
    for data in iter(lambda : file.read(100 * 1024), b''):
        new_file.write(decompressor.decompress(data))
# 使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format(writefile,  binary=False) 
EMBEDDING_DIM = cn_model['哈哈'].shape[0] 

def create_word_ind_Dic(vocab_size):
    #   载入词典
    '''
    word2ind_dic = cn_model.vocab[word].index
    ind2word_dic = cn_model.index2word[i]
    '''        
    word2ind_dic = {}
    ind2word_dic = {}
    #   预训练不能偏移
    # 将词表偏移，留出槽位存储特殊字符
    # word2ind_dic['<PAD>'] = 0     #padding时候填充
    # word2ind_dic['<UNK>'] = 1     #找不到这个字的时候返回unk
    # ind2word_dic[0] = '<PAD>' 
    # ind2word_dic[1] = '<UNK>'                
    # for i in range(2,vocab_size):   
    for i in range(vocab_size):        
        word = cn_model.index2word[i]
        word2ind_dic[word] = i
        ind2word_dic[i] = word       
    return word2ind_dic,ind2word_dic
        
word2ind_dic, ind2word_dic = create_word_ind_Dic(VOCAB_SIZE)      
print(len(word2ind_dic))


# 只使用前VOCAB_SIZE个词
VOCAB_SIZE = 50000
def embedding_M():
    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    # embedding_matrix为一个 [VOCAB_SIZE ，embedding_dim] 的矩阵
    # 维度为 50000 * 300
    for i in range(VOCAB_SIZE):
        embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    return embedding_matrix
embedding_matrix = embedding_M()

# len(cn_model.vocab)
# Out[29]: 259869
dump_dir = 'D:\\myGit\\Algorithm\\A_my_nlp_project\\project_NLP\\pretrain_weights'

filename = 'gensim_50000vocab_300d'
filepath = os.path.join(dump_dir,filename)
if not os.path.exists(filepath):
    os.makedirs(filepath)    
embedding_matrix.dump(os.path.join(filepath,'embedding_matrix.txt'))
import json
word2id_dir = os.path.join(filepath,'word2id.json')
json.dump(word2ind_dic, open(word2id_dir, "w"))
id2word_dir = os.path.join(filepath,'id2word.json')
json.dump(ind2word_dic, open(id2word_dir, "w"))


# load
dump_dir = 'D:\\myGit\\Algorithm\\A_my_nlp_project\\project_NLP\\pretrain_weights'
filename = 'gensim_50000vocab_300d'
filepath = os.path.join(dump_dir,filename)
embed_matrix_dir = os.path.join(filepath,'embedding_matrix.txt')
word2id_dir = os.path.join(filepath,'word2id.json')
id2word_dir = os.path.join(filepath,'id2word.json')

word2id = json.load(open(word2id_dir))    
id2word = json.load(open(id2word_dir))    
embedding_matrix = np.load(embed_matrix_dir)



