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
>  短序列区域在第6轮达到最高，模型逐渐倾向预测短序列

![epochs](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task4_seq2seq/imgs/seq_len.png)

4. 结论：  
decoder 端通过预测出一个词预测，再作为input 输入预测下一个词直到end/max_len时候停止。  
output 更倾向于更可能出现什么词序的组合而非原文写什么内容。  
生成任务没有预训练机制协助下可能要配合实体/时态等任务实现增强,后续更新。  
> 下面是一些例子，顺序是：西班牙原文，模型输出，真实值。  
>  11  
<start> ¿ te gusta tu jefe ? <end>  
do you like your name ? <end>   
<start> do you like your boss ? <end>  
> 15  
<start> quiero conocer a tu hermana mayor . <end>  
i want to know your father . <end>   
<start> i want to meet your older sister . <end>      
>  42  
<start> este es mi plan . <end>  
this is my life . <end>   
<start> this is my plan . <end>  
  
  




