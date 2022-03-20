# Classify-leaves with pytorch-lightning 

看沐神的视频萌新想练练手，用resnet50做了个baseline（其他net跑不动了，且residual训练起来较快）

score：

> Private score : 0.97068
>
> Pubilc score : 0.96772



论文地址：https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html



*后续加K折验证或者Bagging之后貌似能提到0.984-0.99左右，个人懒就想再跑了*



- 有一说一pytorch-lightning挺赞，省去了很多要重复编写的代码结构，自带的ckpt和内嵌tensorboard也很爽
- pytorch-lightning支持mixed precision
- pytorch-lightning支持tpu，能在colab上运行
- albumentations的data argument很好用，提供了比原生pytorch中torchvision的transform更多的DA方法
- timm的model真的是多~



