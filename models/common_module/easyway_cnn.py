import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import matplotlib as mpl
import matplotlib.pyplot as plt
from customized_optimizer import CustomizedSchedule


def textcnn(embedding_matrix=None,
            is_pretrain=False,
            save_weights_dir=None,
            trainable = True):        
    '''textcnn classify 任务 ''' 
# =============================================================================
#   定义模型结构
    input_layer=L.Input(shape=(None,),name='feature_input')
    emb_layer =L.Embedding(input_dim=config['vocab_size'],
                           output_dim=config['embed_size'],
                           name = 'embedding')(input_layer)            
    if is_pretrain:
        ''' 预训练词向量'''
        emb_layer=L.Embedding(input_dim=config['vocab_size'],
                             output_dim=config['embed_size'],
                             weights = [embedding_matrix], 
                             trainable = trainable,
                             name = 'pretrain_emb')(input_layer)   
    con_layer = L.Conv1D(filters=config['num_filters'],
                         kernel_size=config['kernel_size'])(emb_layer)                        
    con_layer = L.BatchNormalization()(con_layer)
    con_layer = L.Activation('relu')(con_layer)
    con_layer = L.GlobalMaxPool1D()(con_layer)    
    output_layer = L.Dense(config['hidden_dim'],activation = 'relu',name="feature_output")(con_layer)        
    output_layer=L.Dense(config['num_classes'],activation='softmax',name = 'out_classify')(output_layer)
# =============================================================================
    '''模型 实例化 '''
    model=K.models.Model(inputs=[input_layer],
                         outputs=[output_layer])
    model.summary()
# =============================================================================   
    '''模型编译 '''
    '''自定义学习率优化器'''
    learning_rate = CustomizedSchedule(config['embed_size'])
    optimizer = K.optimizers.Adam(learning_rate,
                                      beta_1 = 0.9,
                                      beta_2 = 0.98,
                                      epsilon = 1e-9)                  
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                   metrics=['accuracy'])    
# =============================================================================
#   有预训练则加载        
    if save_weights_dir!= None:
        print('加载权重')
        model.load_weights(save_weights_dir)
    return model
    
    
    
def resnet50_finetune():
    resnet50 = K.applications.ResNet50(include_top = False,
                                           pooling = 'avg',
                                           weigths = 'imagenet')                
    resnet50.summary()
    #   设置后五层可训练   
    for layer in resnet50.layers[0:-5]:
        layer.trainable = False        
    resnet50_new = K.models.Sequential([
            resnet50,
            L.Dense(config['num_classes'],activation = 'softmax')])        
    resnet50_new.compile(loss = 'categorical_crossentropy',
                         optimizer = 'sgd',
                         metrics = ['accuracy'])
    resnet50_new.summary() 
    return resnet50_new       
        
def multiKernel_model(config):
    '''
    该卷积为并列关系而非纵深关系：
    0.85
    '''
    model = K.Sequential()
    model.add(L.Embedding(input_dim=config['vocab_size'],
                           output_dim=config['embed_size'],
                           input_length = config['max_len'],
                           name = 'embedding'))        
    
    def conv_Model(filters= 64,kernel_size_list = [2,4]):
        '''
        函数式写法
        model = K.Model(inputs=[],
                        outputs = [])

        '''
        try:
            filters=config['num_filters']
        except:
            pass
        
        try:
            kernel_size_list=config['kernel_size_list']
        except:
            pass        
        input_layer = L.Input(shape = (config['max_len'],config['embed_size']))
        cons = []
        for size  in kernel_size_list:            
            con = L.Conv1D(filters = filters,kernel_size = size,
                           activation = 'relu',
                           kernel_regularizer=K.regularizers.l2(0.0001))(input_layer)
            con = L.BatchNormalization()(con)
            pool_size = int(con.shape[-2])
            pool = L.MaxPool1D(pool_size)(con)
            cons.append(pool)
        output_layer = L.concatenate(cons)
        model = K.Model(inputs = input_layer,
                        outputs = output_layer)
        model.summary()
        return model  
    conv = conv_Model()
    model.add(conv)
    '''卷积后必须Flatten'''
    model.add(L.Flatten())
    ''' selu 在卷积中未必有更好效果'''
    model.add(L.Dense( config['hidden_dim'],activation = 'selu'))
    # x = L.Dropout(0.5)(x)
    model.add(L.Dense(config['num_classes'],activation = 'softmax'))
    model.summary()
    return model
model = multiKernel_model()
    


def multiLayerConv_model(config):
    '''
    该卷积为纵深关系：这个效果会差过并列关系 
    0.83
    注意max pooling 后 shape [-2]如果< kernel size 则不能再卷了
    '''
    model = K.Sequential()
    model.add(L.Embedding(input_dim=config['vocab_size'],
                           output_dim=config['embed_size'],
                           input_length = config['max_len'],
                           name = 'embedding'))         
    for ind in range(config['num_layers']):
        model.add(L.Conv1D(filters = config['num_filters'],
                           kernel_size = config['kernel_size'],
                           activation = 'selu'))
        model.add(L.MaxPool1D(5,padding = 'same'))
    
    '''卷积后必须Flatten'''
    model.add(L.Flatten())
    model.add(L.Dense(config['hidden_dim'],activation = 'selu'))
    # x = L.Dropout(0.5)(x)
    model.add(L.Dense(config['num_classes'],activation = 'softmax'))
    model.summary()
    return model
model = multiLayerConv_model()

        

    
if __name__ == '__main__':
    config = {
        'epoch':10,
        'batch_size':16,
        'num_classes':96,        
        'max_len':80,
        'embed_size':256,
        'vocab_size':3196,
        #   普通层配置
        'mlp_units':64,
        'hidden_dim':512,
        'activation':'relu',
        #   卷积层配置
        'num_layers':3,   # 多层卷积，通道和卷积核应该调整，否则可能后面层参数不够
        'num_filters':512, #卷积输出通道
        'kernel_size':5,
        'kernel_size_list':[2,4] ,#考虑多尺寸卷积核
        'pool_size':5 # 池化尺寸
        }         
    