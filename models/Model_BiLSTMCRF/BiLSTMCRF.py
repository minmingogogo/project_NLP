import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint
try:
    from CRF import CRF
except:
    from NER_model.BiLSTMCRF.CRF import CRF
import os
# from CRF import CRF
import tensorflow.keras as K
import tensorflow.keras.layers as L
import matplotlib as mpl
import matplotlib.pyplot as plt



class MyBiLSTMCRF:    
    '''该模式不适合checkpoint 保存，需要改成子类API写法 
    仅定义Model 
    在实例化训练过程中定义fit及compile
    '''
    def __init__(self,config):
        self.vocab_size = config.vocab_size
        self.embed = config.embed
        self.maxLen = config.max_length
        self.tagSum = config.tgt_size

    def buildBiLSTMCRF(self,save_weights_dir=None):
        '''仅ner 任务模型'''
        myModel=Sequential()
        myModel.add(tf.keras.layers.Input(shape=(self.maxLen,)))
        myModel.add(tf.keras.layers.Embedding(self.vocab_size, self.embed))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    self.tagSum, return_sequences=True, activation="tanh"), merge_mode='sum'))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    self.tagSum, return_sequences=True, activation="softmax"), merge_mode='sum'))
        crf=CRF(self.tagSum,name='crf_layer')
        '''这里CRF内部自动实例化转移矩阵么？？'''
        myModel.add(crf)
        
        myModel.summary()
        '''自定义学习率优化器'''
        learning_rate = CustomizedSchedule(self.embed)
        optimizer = K.optimizers.Adam(learning_rate,
                                          beta_1 = 0.9,
                                          beta_2 = 0.98,
                                          epsilon = 1e-9)            
                       
        myModel.compile(optimizer = optimizer,
                        loss={'crf_layer': crf.get_loss}
                        )                        
        # myModel.compile(optimizer = 'adam',
        #                 loss={'crf_layer': crf.get_loss},
        #                 metrics = ['accuracy'])
        if save_weights_dir!= None:
            print('加载权重')
            myModel.load_weights(save_weights_dir)
        self.myBiLSTMCRF=myModel


    def fit(self,X,y,epochs=100,save_weights_dir = None):
        # if len(y.shape)==3:
        #     y = np.argmax(y,axis=-1)
        # if self.sequenceLengths is None:#   没用啊这个
        #     self.sequenceLengths=[row.shape[0] for row in y]            
        earlystop_callback = EarlyStopping(patience=5,min_delta=1e-3,monitor = 'loss')
        
        #   数据集过小导致提前停止报错？
        '''
        该句报错，网上说是验证集太小，增大验证集
        ETA: 3s - loss: 24.5373WARNING:
            tensorflow:Early stopping conditioned on metric `val_loss` 
            which is not available. Available metrics are: loss
        '''        
        callbacks = [earlystop_callback]

        history=self.myBiLSTMCRF.fit(X,y,epochs=epochs,
                                     validation_split = 0.2,
                                     callbacks = callbacks)
        try :            
            self.myBiLSTMCRF.save_weights(save_weights_dir)
            # self.myBiLSTMCRF.save_weights('./save_weight/ner_bilstmcrf_weights')
        except Exception as e:
            print('保存失败 :{}'.format(e))
        return history
        

    def build_textcnn(self,embedding_matrix=None,is_pretrain=False,save_weights_dir=None):        
        '''textcnn classify 任务 ''' 
        input_layer=L.Input(shape=(None,),name='feature_input')
        emb_layer =L.Embedding(input_dim=config.vocab_size,
                               output_dim=config.embed,
                               name = 'embedding')(input_layer)        
        
        if is_pretrain:
            ''' 预训练词向量'''
            emb_layer=L.Embedding(input_dim=config.vocab_size,
                                 output_dim=config.embed,
                                 weights = [embedding_matrix], 
                                 trainable = True,name = 'pretrain_emb')(input_layer)   
        con_layer = L.Conv1D(filters=config.num_filters,
                             kernel_size=config.kernel_size)(emb_layer)                        
        con_layer = L.BatchNormalization()(con_layer)
        con_layer = L.Activation('relu')(con_layer)
        con_layer = L.GlobalMaxPool1D()(con_layer)
        
        output_layer = L.Dense(config.hidden_dim,activation = 'relu',name="feature_output")(con_layer)        
        output_layer=L.Dense(config.num_classes,activation='softmax',name = 'out_classify')(output_layer)

        '''模型 实例化 '''
        model=K.models.Model(inputs=[input_layer],
                             outputs=[output_layer])
        model.summary()
        self.emb_layer = emb_layer
        
        '''模型编译 '''
        '''自定义学习率优化器'''
        learning_rate = CustomizedSchedule(self.embed)
        optimizer = K.optimizers.Adam(learning_rate,
                                          beta_1 = 0.9,
                                          beta_2 = 0.98,
                                          epsilon = 1e-9)              
        
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                       metrics=['accuracy'])    

                 
        if save_weights_dir!= None:
            print('加载权重')
            model.load_weights(save_weights_dir)
        self.text_cnn=model
            
    
    def fit_textcnn(self,X,y,epochs=10,save_weights_dir = None):
       
        earlystop_callback = EarlyStopping(patience=5,min_delta=1e-3,monitor = 'loss')        
        callbacks = [earlystop_callback]

        history=self.text_cnn.fit(X,y,epochs=epochs,
                                     validation_split = 0.2,
                                     callbacks = callbacks)
        try :            
            self.text_cnn.save_weights(save_weights_dir)
            # self.myBiLSTMCRF.save_weights('./save_weight/nerClass_bilstmcrf_weights')
        except Exception as e:
            print('保存失败 :{}'.format(e))
        return history
    def save_weight_textcnn(self,save_weights_dir,embedding_weight_index=None,is_save_weight = True):
        if embedding_weight_index!=None:
            ind = embedding_weight_index
            emb_layer = self.ner_model.layers[ind]  #embedding层是模型第一层
            weights = emb_layer.get_weights()[0]
            print(weights.shape)
            outdir = save_weights_dir + '_embedding_weights.pkl'
            f= open(outdir,'wb')
            pickle.dump((weights),f)          
        
        if is_save_weight :
            try :            
                self.text_cnn.save_weights(save_weights_dir)
            except Exception as e:
                self.text_cnn.save_weights('./save_weight/Classify_textcnn_weights')
                print('指定路径保存失败，采用默认路径保存 :{}'.format(e))
            
    
    def build_BiLSTMCRF(self,save_weights_dir=None):
        '''ner + classify 任务 多输出模型'''
        input_layer = L.Input(shape=(self.maxLen,))
        emb_layer = L.Embedding(self.vocab_size, self.embed)(input_layer)
        lstm_l1 = L.Bidirectional(L.LSTM(self.tagSum, return_sequences=True, activation="tanh"),
                                  merge_mode='sum')(emb_layer)
        lstm_l2 = L.Bidirectional(L.LSTM(self.tagSum, return_sequences=True, activation="softmax"),
                                  merge_mode='sum')(lstm_l1)
        crf=CRF(self.tagSum,name='crf_layer')
        output_crf = crf(lstm_l2)
        
        # self.emb_layer = emb_layer
                
        '''模型 实例化 '''
        model=K.models.Model(inputs=[input_layer],
                             outputs=[output_crf])
        model.summary()
        
        '''模型编译 '''
        '''自定义学习率优化器'''
        learning_rate = CustomizedSchedule(self.embed)
        optimizer = K.optimizers.Adam(learning_rate,
                                          beta_1 = 0.9,
                                          beta_2 = 0.98,
                                          epsilon = 1e-9)              
        
        model.compile(optimizer=optimizer,
                      loss={'crf_layer': crf.get_loss},
                      # metrics=['mae','acc']
                      )                     
        if save_weights_dir!= None:
            print('加载权重')
            model.load_weights(save_weights_dir)
        self.ner_model=model
            
    
    def fit_ner(self,X,y,epochs=10,save_weights_dir = None):       
        earlystop_callback = EarlyStopping(patience=5,min_delta=1e-3,monitor = 'loss')        
        callbacks = [earlystop_callback]

        history=self.ner_model.fit(X,y,epochs=epochs,
                                     validation_split = 0.2,
                                     callbacks = callbacks)

        try :            
            self.ner_model.save_weights(save_weights_dir)
            # self.myBiLSTMCRF.save_weights('./save_weight/nerClass_bilstmcrf_weights')
        except Exception as e:
            print('保存失败 :{}'.format(e))
        return history
    
    def save_embedding_ner(self,save_dir,
                        embedding_weight_index = 1):
        ind = embedding_weight_index
        emb_layer = self.ner_model.layers[ind]  #embedding层是模型第一层
        weights = emb_layer.get_weights()[0]
        print(weights.shape)
        f= open(save_dir,'wb')
        pickle.dump((weights),f)     
    
    
    def save_weight_ner(self,save_weights_dir,
                        embedding_weight_index = None,
                        is_save_weight=True):
        if embedding_weight_index!=None:
            ind = embedding_weight_index
            emb_layer = self.ner_model.layers[ind]  #embedding层是模型第一层
            weights = emb_layer.get_weights()[0]
            print(weights.shape)
            outdir = save_weights_dir + '_embedding_weights.pkl'
            f= open(outdir,'wb')
            pickle.jump((weights),f)                    
        if is_save_weight:
            try :            
                self.ner_model.save_weights(save_weights_dir)
            except Exception as e:
                self.ner_model.save_weights('./save_weight/ner_bilstmcrf_weights')
                print('指定路径保存失败，采用默认路径保存 :{}'.format(e))

    def predict(self,X):
        preYArr=self.myBiLSTMCRF.predict(X)
        return preYArr
    def predict_ner(self,X):
        preYArr=self.ner_model.predict(X)
        return preYArr    
    def predict_textcnn(self,X):
        preYArr=self.text_cnn.predict(X)
        return preYArr       

    def evaluate(self,X,y):
        loss, acc=self.myBiLSTMCRF.evaluate(X,y)
        print("Restored model, accuracy:{:5.2f}%".format(100 * acc))
        return acc


class CustomizedSchedule(
    K.optimizers.schedules.LearningRateSchedule):
    '''自定义学习率优化器
    # lrate = (d_model ** -0.5) * min(step_num ** (-0.5),
#                                 step_num * warm_up_steps **(-1.5))
    '''
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomizedSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))
        
        arg3 = tf.math.rsqrt(self.d_model)
        
        return arg3 * tf.math.minimum(arg1, arg2)
    
# learning_rate = CustomizedSchedule(d_model)
# optimizer = keras.optimizers.Adam(learning_rate,
#                                   beta_1 = 0.9,
#                                   beta_2 = 0.98,
#                                   epsilon = 1e-9)        
# temp_learning_rate_schedule = CustomizedSchedule(100)

# plt.plot(
#     temp_learning_rate_schedule(
#         tf.range(40000, dtype=tf.float32)))
# plt.ylabel("Leraning rate")
# plt.xlabel("Train step")        
def tmp_model():
    
    vocab_size = config.vocab_size
    embed = config.embed
    maxLen = config.max_length
    tagSum = config.tgt_size    
    
    input_layer = L.Input(shape=(maxLen,))
    emb_layer = L.Embedding(vocab_size, embed)(input_layer)
    lstm_l1 = L.Bidirectional(L.LSTM(tagSum, return_sequences=True, activation="tanh"),
                              merge_mode='sum')(emb_layer)
    lstm_l2 = L.Bidirectional(L.LSTM(tagSum, return_sequences=True, activation="softmax"),
                              merge_mode='sum')(lstm_l1)
    crf=CRF(tagSum,name='crf_layer')
    output_crf = crf(lstm_l2)    
            
    '''模型 实例化 '''
    model=K.models.Model(inputs=[input_layer],
                         outputs=[output_crf])
    model.summary()
    
    '''模型编译 '''
    '''自定义学习率优化器'''
    learning_rate = CustomizedSchedule(embed)
    optimizer = K.optimizers.Adam(learning_rate,
                                      beta_1 = 0.9,
                                      beta_2 = 0.98,
                                      epsilon = 1e-9)              
    
    model.compile(optimizer=optimizer,
                  loss={'crf_layer': crf.get_loss},
                  # metrics=['mae','acc']
                  )                     
    model.fit(x,y_ner,2)

    ''' embedding 导出'''
    
    emb = model.layers[1]  #embedding层是模型第一层
    weights = emb.get_weights()[0]
    print(weights.shape)
    outdir = './save_weight/ner_embedding_weights.pkl'
    f= open(outdir,'wb')
    pickle.jump((weights),f)
 
    ''' embedding 加载-'''
    f= open(outdir,'rb')
    weights = pickle.load(f)
       
        
    

if __name__=="__main__":
    import pickle
    from dataProcess import Get_corpus
    corpus = Get_corpus()    
    from ner_config import nerConfig
    config = nerConfig()
    # =============================================================================
    # LOAD DATA: 
    # =============================================================================
    datadir = 'D:\\myGit\\Algorithm\\A_my_nlp_project\\my_nlp_classify_example\\address_ner_classify\\data\\train_data'
    log_dir = "D:\\myGit\\Algorithm\\A_my_nlp_project\\my_nlp_classify_example\\address_ner_classify\\NER_model\logs"
    save_weight_path = os.path.join(log_dir,'save_weights')
    if not os.path.exists(save_weight_path):
        os.makedirs(save_weight_path)
    filename = os.path.join(datadir,'train_data_PCZSVKN.pkl')
    f= open(filename,'rb')
    word_list,char_list,bio_list,bieo_list,label_list,strategy,msg = pickle.load(f)
    
    # =============================================================================
    # Train data: 
    # =============================================================================
    x_seq_list = char_list
    y_seq_list = bio_list
    max_seq_len = config.max_length
    #   获取 ner 的输入输出
    word2id,id2word,tag2id,id2tag,x_train,y_train,x,y = corpus.wordSeq2train_data(x_seq_list,y_seq_list,max_seq_len)
    #   获取 classify  的输入输出
    labels,id2category,category2id = corpus.classify_category_data(label_list)
    # =============================================================================
    # CONFIG : 
    # =============================================================================
    '''与配置文件参数校验'''
    vocab_size = len(word2id)
    tgt_size=len(tag2id) #   状态转移矩阵,状态个数，等于label 个数
    max_length = 80  
    num_classes = len(id2category)
    if vocab_size==config.vocab_size:        
        print('vocab_size 一致')
    else:
        print('vocab_size 不一致')
    if tgt_size==config.tgt_size:        
        print('tgt_size 一致')
    else:
        print('tgt_size 不一致')
    if num_classes==config.num_classes:        
        print('num_classes 一致')
    else:
        print('num_classes 不一致')
        
        
    from ner_config import nerConfig
    config = nerConfig()   
    '''参数校验完毕加载模型'''
    # =============================================================================
    # 训练shuffle数据 防止过拟合
    # =============================================================================
    #   shuffle

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
    def jiaoyan_label():
        '''分类 category 校验'''
        errortag =0
        for ind in tqdm.tqdm(range(len(label_list))):
            # if ind>10:
            #     break
            # ind ==0    
            cat_index = [i for i,j in enumerate(labels[ind]) if j==1] 
            arr_point = id2category[cat_index[0]]
            raw_point = label_list[ind]
            if arr_point!=raw_point:
                errortag+=1
        if errortag==0:
            print('校验一致')
        else:
            print('错误个数',errortag)   
    #   shuffle 前校验label是否一致
    jiaoyan_label()              
    shuffle_data = [x_train,y_train,labels,label_list]
    tmp = shuffle_all(shuffle_data)
    #   shuffle 后校验 label是否一致
    jiaoyan_label()
            
    # =============================================================================
    # 筛选训练数据    
    # =============================================================================

    test_num = 4
    train_size = 5000
    star = test_num*1000
    end = star + train_size
    test_size = 2000        
    # x = x_train[5000:10000]
    # y = y_train[5000:10000]  
    
    x = x_train[star:end]
    y_ner = y_train[star:end] 
    y_cla = labels[star:end] 
    y_cla_label = label_list[star:end] 
    

    x_test = x_train[end:end+test_size]
    y_test_ner = y_train[end:end+test_size]
    y_test_cla = labels[end:end+test_size]
    y_test_cla_label = label_list[end:end+test_size]
  
    # =============================================================================
    # 功能1 验证 ：实体提取
    # =============================================================================
    save_weight_path = os.path.join('save_weight')
    if not os.path.exists(save_weight_path):
        os.makedirs(save_weight_path)
    # save_weights_dir = os.path.join(save_weight_path,'ner_bilstmcrf_weights')
    # '''模型实例化-训练-保存'''
    # ner_bilstm_model = MyBiLSTMCRF(config)
    # ner_bilstm_model.buildBiLSTMCRF()
    # history = ner_bilstm_model.fit(x,y_ner,10) 
    '''模型实例化-训练-保存'''
    ner_bilstm_model = MyBiLSTMCRF(config)
    ner_bilstm_model.build_BiLSTMCRF()
    history = ner_bilstm_model.fit_ner(x_test,y_test_ner,50,'./save_weight/ner_bilstmcrf_weights') 

    '''模型加载-训练-保存'''
    ner_bilstm_model = MyBiLSTMCRF(config)
    ner_bilstm_model.build_BiLSTMCRF('./save_weight/ner_bilstmcrf_weights')
    history = ner_bilstm_model.fit_ner(x_test,y_test_ner,2,'./save_weight/ner_bilstmcrf_weights') 
    
    '''模型-embedding-保存'''
    embed_save_dir = os.path.join(save_weight_path,'ner_embedding_weights.pkl')

    ner_bilstm_model.save_embedding_ner(embed_save_dir)
    ''' 训练效果分析 '''
    ner_result(x_test,y_test_ner)
    '''
    50轮训练
    8 turn : cost :0.09 ;
    msg :8 turn : jacc max:1.0 jacc mean: 0.9712345949037305
    5 turn : cost :0.08 ;
    msg :5 turn : jacc max:1.0 jacc mean: 0.9538857384219747
    
    '''
    
    
    
    # =============================================================================
    # 功能2 加载预训练embedding 分类    
    # =============================================================================
    '''模型 -embedding-加载'''
    save_weight_path = os.path.join('save_weight')
    if not os.path.exists(save_weight_path):
        os.makedirs(save_weight_path)        
    embed_save_dir = os.path.join(save_weight_path,'ner_embedding_weights.pkl')
    f= open(embed_save_dir,'rb')
    embedding_matrix = pickle.load(f)
    
    '''模型textcnn -embedding-fine-tune'''
    # save_weights_dir = './save_weight/classify_textcnn_weights'
    cla_model = MyBiLSTMCRF(config)
    cla_model.build_textcnn(embedding_matrix,is_pretrain =1)
    cla_model.fit_textcnn(x,y_cla,20)
    ''' 训练效果分析 '''
    classify_result(x_test,y_test_cla,y_test_cla_label)
    
    
    ''' 模型加载-预测-分析
    预训练两轮加载到分类的效果
    总地址数: 2000
    解析率: 0.9705
    不排除失败准确数: 0.918  排除失败准确率: 0.9335394126738794
    
    加载50轮预训练的效果好像差一些
    总地址数: 2000
    解析率: 0.9515
    不排除失败准确数: 0.884  排除失败准确率: 0.9033105622700999
    总地址数: 2000
    解析率: 0.951
    不排除失败准确数: 0.8755  排除失败准确率: 0.8969505783385909
    
    不用预训练的效果    
    总地址数: 2000
    解析率: 0.973
    不排除失败准确数: 0.914  排除失败准确率: 0.9285714285714286
    '''
    
    '''模型textcnn 不加载预训练'''
    # save_weights_dir = './save_weight/classify_textcnn_weights'
    cla_model = MyBiLSTMCRF(config)
    cla_model.build_textcnn(is_pretrain =0)
    cla_model.fit_textcnn(x,y_cla,20)
    ''' 训练效果分析 
    '''
    classify_result(x_test,y_test_cla,y_test_cla_label)
        
    
    
    # ner_model = MyBiLSTMCRF(config)
    # ner_model.build_BiLSTMCRF('./save_weight/ner_bilstmcrf_weights')    
    # embedding_matrix = ner_model.emb_layer
    # y_test_ner_pre = ner_model.predict(x_test) 
    

    # =============================================================================
    # 分析 实体提取 
    # =============================================================================
    import time
    from sklearn.metrics import accuracy_score,precision_score
    import pandas as pd
    
    def id2word_label(tmp_x,tmp_y):
        '''打印原始数据'''
        for i in range(len(tmp_x)):
            word = [id2word[x] for x in tmp_x[i]]
            print(word)
            label = [id2tag[x] for x in tmp_y[i]]
            print(label)
    #   统计总正确率和分类正确率
            
    def analyse_ner(tmp_x,tmp_y):
        '''分析NER'''
        w_list = []
        pre_list = []
        true_list = []
        jaccad_list = []    
        if len(tmp_y.shape)==1:
            tmp_y = tmp_y[np.newaxis,:] 
        if len(tmp_x.shape)==1:
            tmp_x = tmp_x[np.newaxis,:]#    增加一个维度      
            
        tmp_y_pre = ner_bilstm_model.predict_ner(tmp_x)            
        for i in range(len(tmp_x)):
            word = [id2word[x] for x in tmp_x[i]]
            # print(word)
            w_l = [x for x in word if x !='<PAD>']
            wlen = len(w_l)# 有效长度
            label_pre = [id2tag[x] for x in tmp_y_pre[i]]
            label_true = [id2tag[x] for x in tmp_y[i]]
            l_p = label_pre[:wlen]
            l_t = label_true[:wlen]
            l_p = list(map(lambda x : x if x!='PAD' else 'O',l_p))
            jaccad = np.sum(np.array(l_t) == np.array(l_p))/wlen
            #   加了准确率十轮训练效果一般
            w_list.append(w_l)
            pre_list.append(l_p)
            true_list.append(l_t)
            jaccad_list.append(jaccad)
        return w_list,pre_list,true_list,jaccad_list    

    def ner_result(tmp_x,tmp_y):
        ''' 输入 实际的x,y'''
        s = time.time()
        msg = '_'
        jacc_max = 'unk'
        jacc_mean = 'unk'
        w_list,pre_list,true_list,jaccad_list = analyse_ner(tmp_x,tmp_y)
        try:
            jacc_max = np.max(jaccad_list)
            jacc_mean = np.mean(jaccad_list)
        except:
            pass
        msg = "{} turn : jacc max:{} jacc mean: {}".format(
            test_num,jacc_max,jacc_mean)        
        e = time.time()
        cost = round((e-s)/60,2)
        print('{} turn : cost :{} ;msg :{}'.format(test_num,cost,msg))  

    def classify_result(x_test,y_test_cla,y_test_cla_label):
        logits = cla_model.predict_textcnn(x_test)
        #   最大值的位置
        pred = np.argmax(logits, 1)        
        #   约束：top1>top2 0.4
        sorted_soft_list = np.sort(logits)
        threshold = 0.4
        jiexi_result = [id2category[pred[i]] if sorted_soft_list[i, -1] - sorted_soft_list[i, -2] >= threshold else '解析失败!' for i in range(len(sorted_soft_list))]
        # jiexi_result = [pred[i] if sorted_soft_list[i, -1] - sorted_soft_list[i, -2] >= threshold else '解析失败!' for i in range(len(sorted_soft_list))]
        #   不排除解析失败的准确率
        y_true = np.argmax(y_test_cla,1)
        y_pred = pred
        acc_with_fail = accuracy_score(y_true, y_pred)
    
        #   排除解析失败的准确率
        # 召回:        
        tmp_pre_true = pd.DataFrame(jiexi_result,columns = ['pre'])
        tmp_pre_true['true'] = list(y_test_cla_label)
        tmp = tmp_pre_true[tmp_pre_true['pre'] != '解析失败!']
        acc_with_nofail = accuracy_score(list(tmp['true']), list(tmp['pre']))
        recall_no_fail = len(tmp)/len(x_test)
        
        print('总地址数:', len(x_test))
        print('解析率:', recall_no_fail)
        print('不排除失败准确数:', acc_with_fail, ' 排除失败准确率:' ,acc_with_nofail)
    
















    # =============================================================================
    # 功能2 验证 多任务
    # =============================================================================
    save_weight_path = os.path.join('save_weight_ner_classify')
    if not os.path.exists(save_weight_path):
        os.makedirs(save_weight_path)
    save_weights_dir = './/save_weight_ner_classify//bilstmcrf_textcnn_weights'
    '''模型实例化-训练-保存'''
    # x = x_train[star:end]
    # y_ner = y_train[star:end] 
    # y_cla = labels[star:end] 
    # y_cla_label = label_list[star:end]     
    
    
    ner_cla_model = MyBiLSTMCRF(config)
    ner_cla_model.buildBiLSTMCRF_muloutput()
    
    #   单独crf
    ner_cla_model = MyBiLSTMCRF(config)
    ner_cla_model.buildBiLSTMCRF_textCnn()
    history = ner_cla_model.textcnn_fit(x,[y_ner,y_cla],10) 

    ''' 模型加载-预测-分析'''
    ner_cla_model = MyBiLSTMCRF(config)
    ner_cla_model.buildBiLSTMCRF_muloutput(save_weights_dir) 
    y_test_ner_pre,y_test_cla_pre = ner_cla_model.predict([x_test]) 
    
    
    
    # =============================================================================
    # MODEL Restore : 
    # =============================================================================
    
    # step8 重新创建模型
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
     
    
    model = MyBiLSTMCRF(vocab_size,max_length,tag2id,tgt_size)
    model.buildBiLSTMCRF()
    # step9 恢复权重
    # model.load_weights('./save_weights/my_save_weights')
    model.load_weights('./my_save_weights')
    
    
    # step10 测试模型
    x_test = x_train[8000:8050]
    y_test = y_train[8000:8050]   
    y_true = model.predict(x_test)
    loss, acc = model.evaluate(x_test, y_test)
    print("Restored model, accuracy:{:5.2f}%".format(100 * acc))

    
    
    
    
    
    
    
    
    
    