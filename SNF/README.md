#### 使用方法
```python
from SNF import SNF

# 计算样本间欧氏距离
dist1 = SNF.dist2(data1)
dist2 = SNF.dist2(data2)

# 根据每个特征计算相似矩阵
affinity1 = SNF.affinityMatrix(dist1)
affinity2 = SNF.affinityMatrix(dist2)

# 把不同特征的相似矩阵融合成一个相似矩阵
fusion = SNF.SNF([affinity1, affinity2])
```