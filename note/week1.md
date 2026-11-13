我跑通了 pytorch_deep_learning_medimg_tutorial 项目。

ResNet18 是当前最好的基线（Train Acc 91.87%）。

ViT 出现异常，loss 卡在 2.185 附近，疑似 patch_size 或训练配置不适配 28×28 小图，需要后续排查。

我们是基于手动下载数据库来完成，将download=true改成flase,并且将数据格式pathmidnist从float64改成unit8.
