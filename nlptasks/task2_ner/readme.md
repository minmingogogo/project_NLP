##	对比Bert+CRF 与 bi-LSTM+CRF 在实体提取任务表现  

### 比较方案  
 1. 两种模型对比
 2. 两种标记方案对比
 3. 单场景训练多场景验证


#### 数据说明  
> 某省分级地址文本  
> 颗粒度划分：  
> 行政区域：省、市、区、县、镇、村、  
> 点线面范围细分 ：org: 建筑（公司/商铺/等）、road:路（路/街/巷/等）、park:园区（工业园/农场/等）  

#### 1 模型架构  
> **Bert+CRF**  
>> transformer_block    
>>> word_embedding  
>>> position_embedding  
>>> LN  
>>> multi_head_attention_block    

>> dense_layer  
>> crf_layer  


> **bi-LSTM+CRF**  
>> word_embedding  
>> bilstm1
>> bilstm2
>> crf_layer

**optimizer : 采用带warmup 的衰减学习率**
#### 2 标注方案说明
> 分别采用 bio 和 bieo 方式对数据集标注   
  **bio:**  
  "text": "安 徽 省 合 肥 市 蜀 山 区 南 岗 镇 瓦 屋 村 惠 明 新 村 1 3 栋",  
  "true": "b_pro i_pro i_pro b_cit i_cit i_cit b_zon i_zon i_zon b_str i_str i_str b_vil i_vil i_vil b_org i_org i_org i_org i_org i_org i_org"  
  **bieo:**  
  "text": "安 徽 省 合 肥 市 蜀 山 区 南 岗 镇 瓦 屋 村 惠 明 新 村 1 3 栋",  
  "true": "b_pro i_pro e_pro b_cit i_cit e_cit b_zon i_zon e_zon b_str i_str e_str b_vil i_vil e_vil b_org i_org i_org i_org i_org i_org e_org"  
  

#### 3 训练/测试场景说明  
> 上述八种实体文本分别对应以下字符：  
> 行政区划：P C Z S V => 省、市、区、县、镇、村  ,小写 p c z s v 对应行政区划简称，  
_eg: P: 广东省 ，p :广东 ；S ：大沥镇， s :大沥_  
> 地址明细：org road park 统一对应为 K  

训练/测试策略如下:

    strategy ={  
              #   单一干扰  
             'PCZSVK':'无干扰',             
             'PCZSVNK':'中间噪声',      
             'NPCZNSVNK':'多噪声',    
             'CPZSVK':'乱序 PC',         
             'SVKPCZ':'前后乱序',                 
             'CZSVK':'缺失 P',      
             'ZSVK':'缺失 PC ',              
             'pcZSVK':'简称 pc',     
             'pczsK':'简称 pczs',            
             'CZPCZSVK':'重复 CZ',           
             'PCZpczSVK':'重复 pcz',         
             #   多干扰  
             'CPZSVKN':'乱序 PC + 结尾噪声',               
             'NCZSVNKN':'缺失 P + 多噪声',     
             'zsK':'缺失 + 简称',             
             'KSCZNpNczN':'乱序 + 缺失 + 重复 +噪声 ',        
             }  
#### 效果对比  
> 训练策略: 'PCZSVNK'   
> 测试策略：'PCZSVK CPZSVK SVKPCZ CZSVK ZSVK pcZSVK pczsK CZPCZSVK PCZpczSVK CPZSVKN NCZSVNKN zsK KSCZNpNczN'  

**训练速度比较**  
bert+crf 模型最快到达最优  

![train](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/compare_models_trainprocess.png)



**模型健壮性比较**  
仅训练 5000条 'PCZSVNK'策略数据,在13种测试策略上验证效果，比较模型的健壮性。   
> 在 bio 标注模式下，bert+crf 略微优于 bi-lstm+crf 表现   
> 在 bieo 标注模式下，bert+crf 在13种干扰场景下，11种准确率在 0.79以上，其中6种在0.9以上  

![evaluate](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/compare_models.png)








