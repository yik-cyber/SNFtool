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

#### Reference

Similarity Network Fusion:

[Similarity network fusion for aggregating data types on a genomic scale](https://www.nature.com/articles/nmeth.2810)

Silhouette Score:

[Silhouettes: a graphical aid to the interpretation and validation of cluster analysis](https://www.sciencedirect.com/science/article/pii/0377042787901257)

生存分析：

[LIFELINES: Introduction to survival analysis](https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html#introduction-to-survival-analysis)

