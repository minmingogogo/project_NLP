### 机器翻译：西班牙语转英语
**Model** : 
> seq2seq-注意  

1. loss 变化 
![epochs](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task4_seq2seq/imgs/epoch_loss.png)

2. 评价指标jaccard

![epochs](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task4_seq2seq/imgs/jaccard1.png)

_由于训练数据普遍较短，bleu 1-4gram 分数较低此处不做图表，所有score数据数据在/analyse_..文件夹中

![epochs](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task4_seq2seq/imgs/jaccard2.png)

> **对比6run训练验证**：      
**低分区域**随着训练的增加比例持续下降；  
**高分区域**第2run时，100%比例最高   

3. input,target,predict 文本长度对比  
> 模型倾向预测短序列  

![epochs](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task4_seq2seq/imgs/seq_len.png)
