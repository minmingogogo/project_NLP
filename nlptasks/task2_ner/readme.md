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
### 效果对比  
> 训练策略: 'PCZSVNK'   
> 测试策略：'PCZSVK CPZSVK SVKPCZ CZSVK ZSVK pcZSVK pczsK CZPCZSVK PCZpczSVK CPZSVKN NCZSVNKN zsK KSCZNpNczN'  

**训练情况**  
bert+crf 模型最快到达最优  

![train](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/compare_models_trainprocess.png)

![trainm_etrics](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/macro_micro_metrics.png)

**模型健壮性比较**  
仅训练 5000条 'PCZSVNK'策略数据,在13种测试策略上验证效果，比较模型的健壮性。   
> 在 bio 标注模式下，bert+crf 略微优于 bi-lstm+crf 表现   
> 在 bieo 标注模式下，bert+crf 在13种干扰场景下，11种准确率在 0.79以上，其中6种在0.9以上  

![evaluate](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/compare_models.png)

**训练/预测场景下分类f1 score**  

训练场景：PCZSVNK 2000条非模型数据实体提取效果：  
![train_patt](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/%E5%88%86%E7%B1%BB%E5%87%86%E7%A1%AE%E7%8E%872.png)

测试场景：zsK 1000条  
![test](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/%E5%88%86%E7%B1%BB%E5%87%86%E7%A1%AE%E7%8E%875.png)

     效果对比： 
      1 ----------      
      true :
      {'str': {'南岗': array([2, 3])}, 'org': {'幸福金色年华小区6栋': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13])}, 'zon': {'蜀山': array([0, 1])}}}
      pred :
      {'cit': {'蜀山': array([0, 1])}, 'org': {'幸福金色年华小区6栋': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13])}}}
      2 ----------
      true :
      {'road': {'天柱山大道与团肥路交口': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])}, 'str': {'南岗': array([2, 3])}, 'zon': {'蜀山': array([0, 1])}}}
      pred :
      {'road': {'天柱山大道与团肥路交口': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])}, 'cit': {'蜀山': array([0, 1])}}}
      3 ----------
      true :
      {'road': {'磨子潭路1588号': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12])}, 'str': {'南岗': array([2, 3])}, 'zon': {'蜀山': array([0, 1])}}}
      pred :
      {'road': {'磨子潭路1588号': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12])}, 'cit': {'蜀山': array([0, 1])}}}
      4 ----------
      true :
      {'str': {'南岗': array([2, 3])}, 'org': {'惠民新村九栋': array([4, 5, 6, 7, 8, 9])}, 'zon': {'蜀山': array([0, 1])}}}
      pred :
      {'cit': {'蜀山': array([0, 1])}, 'org': {'惠民新村九栋': array([4, 5, 6, 7, 8, 9])}}}
      5 ----------
      true : 
      {'str': {'南岗': array([2, 3])}, 'org': {'福达汽车模具制造有限公司': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])}, 'zon': {'蜀山': array([0, 1])}}}
      pred :
      {'org': {'蜀山南岗福达汽车模具制造有限公司': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])}}}
      ----------

测试场景：PCZpczSVK 1000条  
![test](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/%E5%88%86%E7%B1%BB%E5%87%86%E7%A1%AE%E7%8E%874.png)

    true :
    {'zon': {'蜀山区': array([6, 7, 8]), '蜀山': array([13, 14])}, 'road': {'望江西路与合作化路交叉口': array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])}, 'cit': {'合肥市': array([3, 4, 5]), '合肥': array([11, 12])}, 'str': {'南七街道': array([15, 16, 17, 18])}, 'vil': {'丁香社区': array([19, 20, 21, 22])}, 'pro': {'安徽省': array([0, 1, 2]), '安徽': array([ 9, 10])}, 'org': {'幸福里小区21栋': array([35, 36, 37, 38, 39, 40, 41, 42])}}
    pred :
    {'zon': {'蜀山区': array([6, 7, 8])}, 'road': {'望江西路与合作化路交叉口': array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])}, 'cit': {'合肥市': array([3, 4, 5])}, 'str': {'安徽合肥蜀山南七街道': array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])}, 'vil': {'丁香社区': array([19, 20, 21, 22])}, 'pro': {'安徽省': array([0, 1, 2])}, 'org': {'幸福里小区21栋': array([35, 36, 37, 38, 39, 40, 41, 42])}}

测试场景：SVKPCZ 1000条  
![test](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task2_ner/imgs/%E5%88%86%E7%B1%BB%E5%87%86%E7%A1%AE%E7%8E%87svkpcz.png)  

    1 ----------
    true :
    {'zon': {'蜀山区': array([27, 28, 29])}, 'road': {'黄山路': array([ 8,  9, 10])}, 'str': {'南七街道': array([0, 1, 2, 3])}, 'cit': {'合肥市': array([24, 25, 26])}, 'vil': {'丁香社区': array([4, 5, 6, 7])}, 'pro': {'安徽省': array([21, 22, 23])}, 'org': {'金大地1912区1栋': array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])}}
    pred :
    {'road': {'七街道丁香社区黄山路': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])}, 'org': {'金大地1912区1栋安徽省合肥市蜀山区': array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
           28, 29])}}
    2 ----------
    true :
    {'zon': {'蜀山区': array([34, 35, 36])}, 'road': {'望江西路与合作化路交叉口': array([ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])}, 'str': {'南七街道': array([0, 1, 2, 3])}, 'cit': {'合肥市': array([31, 32, 33])}, 'vil': {'丁香社区': array([4, 5, 6, 7])}, 'pro': {'安徽省': array([28, 29, 30])}, 'org': {'幸福里小区21栋': array([20, 21, 22, 23, 24, 25, 26, 27])}}
    pred :
    {'road': {'江西路与合作化路交叉口': array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])}, 'str': {'街道': array([2, 3])}, 'vil': {'丁香社区': array([4, 5, 6, 7])}, 'pro': {'南七': array([0, 1])}, 'org': {'幸福里小区21栋安徽省合肥市蜀山区': array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])}}


> 从以上训练预测效果对比可以获得以下信息    
1. 带有后缀词训练的模型用于简称场景验证时，行政级别判断不准确：    
   true : 'zon': {'蜀山': array([0, 1]) ; pred :{'cit': {'蜀山': array([0, 1])}    
2. 镇、村级别缺少后缀词难以识别：  （str 代表四级行政区划  镇；vil 代表五级 村/社区  
   true : ''str': {'南岗': array([2, 3])} ; pred :'str': {}    
3. 文本位置先后对实体有明细影响，见 SVKPCZ 效果，四五级在开头时候 ：
   true :  'str': {'南七街道': array([0, 1, 2, 3])} ; pred :'pro': {'南七': array([0, 1])}
   模型将开头实体作为省级，主要受训练数据PCZSVNK  模式影响。
4. 模型学习到了带行政区划前缀的 公司命名方式，见 zsK 效果，这种场景在实际中经常遇到，如：中国南京\*\*\*有限公司 , ：  
   true : 'org': {'福达汽车模具制造有限公司'}; pred:{'org': {'蜀山南岗福达汽车模具制造有限公司'}  

**模型优化**  
通过多种干扰场景测试效果可以在乱序，缺失，重复，噪声，简称，方面进行数据增强或者入模前的数据预处理，从而提升模型健壮性。    
   



