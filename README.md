# 阿里云天池医疗大赛·肺结节检测

项目这几天在紧锣密鼓不断优化中，看commits就知道所做的所有事情。**随后补全更完善readme文件**

Tumor.ipynb和Tumor-3D.ipynb是几个月前的练手预处理文件。现在已经废弃了。

## 预处理

> preproces.py

binary->clear-board->label&regions->closing->dilation, with scikit-image

## Segmentation

> seg_model.py, seg_train.py

使用U-Net模型做Segmentation，初步筛选结节区域

## Classification

> DenseNet_model.py/ResNet_model.py/VGG_model.py/Inception_model.py

分别使用VGG/Inception/ResNet/DenseNet做分类Ensemble。其中DenseNet参考了非常多论文的经验，完整实现。

## 其它

> generators.py

- 小于10mm的结节4倍增强；大于10小于30结节3倍增强；剩余不增强；
- 旋转，平移
- Segmentation中加入30%的负样本
- Classification中1:4的正负比例

