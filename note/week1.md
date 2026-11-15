#day1
我跑通了 pytorch_deep_learning_medimg_tutorial 项目。
ResNet18 是当前最好的基线（Train Acc 91.87%）。
ViT 出现异常，loss 卡在 2.185 附近，疑似 patch_size 或训练配置不适配 28×28 小图，需要后续排查。
我们是基于手动下载数据库来完成，将download=true改成flase,并且将数据格式pathmidnist从float64改成unit8.

#day2
1.在原有代码的基础上，一开始只是完成训练部分，我们在main.py中添加了def evaluate(model, test_loader, device, model_name)、 evaluate(model, test_loader, device, model_name)、evaluate(model, test_loader, device, "ViTHandmade") ，完成了一个测试集的预测功能。
2.数据集pathmidnist中，是已经固定划分好训练集89996、测试集10004、验证集7180
3.今天在train的基础上，完成了test部分，后面还要添加val部分，test accuracy 只能用于“最终报告”，不能反复用来挑参数；后续调参应改用 val。
