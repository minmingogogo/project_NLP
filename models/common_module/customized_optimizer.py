# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:50:00 2020

@author: Scarlett

好像有这样一篇论文，没找回来
"""
import tensorflow.keras as K
import tensorflow as tf
class CustomizedSchedule(
    K.optimizers.schedules.LearningRateSchedule):
    '''自定义学习率优化器
    # lrate = (d_model ** -0.5) * min(step_num ** (-0.5),
#                                 step_num * warm_up_steps **(-1.5))
    '''
    def __init__(self, d_model, warmup_steps = 400):
        super(CustomizedSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))        
        arg3 = tf.math.rsqrt(self.d_model)        
        return arg3 * tf.math.minimum(arg1, arg2)    
if __name__ == '__main__':
    
     
   
    
    '''自定义学习率优化器'''
    learning_rate = CustomizedSchedule(embed)
    optimizer = K.optimizers.Adam(learning_rate,
                                      beta_1 = 0.9,
                                      beta_2 = 0.98,
                                      epsilon = 1e-9)     