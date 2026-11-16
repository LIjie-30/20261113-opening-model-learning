#day1
我跑通了 pytorch_deep_learning_medimg_tutorial 项目。
ResNet18 是当前最好的基线（Train Acc 91.87%）。
ViT 出现异常，loss 卡在 2.185 附近，疑似 patch_size 或训练配置不适配 28×28 小图，需要后续排查。
我们是基于手动下载数据库来完成，将download=true改成flase,并且将数据格式pathmidnist从float64改成unit8.

#day2
1.在原有代码的基础上，一开始只是完成训练部分，我们在main.py中添加了def evaluate(model, test_loader, device, model_name)、 evaluate(model, test_loader, device, model_name)、evaluate(model, test_loader, device, "ViTHandmade") ，完成了一个测试集的预测功能。
2.数据集pathmidnist中，是已经固定划分好训练集89996、测试集10004、验证集7180
3.今天在train的基础上，完成了test部分，后面还要添加val部分，test accuracy 只能用于“最终报告”，不能反复用来挑参数；后续调参应改用 val。

### Day 3 - transform 对照实验（最终结论）

## 公平对比：重训 ResNet18
| transform | Train Acc | Test Acc | 落差 |
|---|---:|---:|---:|
| 旧：Lambda 灰度化 + 单通道 Normalize | 91.80% | 77.21% | −14.6 |
| 新：保留 RGB + 三通道 Normalize | 96.18% | 65.45% | −30.7 |

## 关键结论
- 保留颜色后训练集准确率大涨、测试集大跌，是**典型过拟合**。
- 原因推测：病理图像的颜色受扫描仪/染色批次影响，模型把“染色偏色”当成了判别依据，测试集颜色分布不同就崩。
- 旧 transform 的 Lambda 虽然“丢信息”，但意外起到了“去颜色偏色”的作用，迫使模型学形态。
- 方法论：**给模型更多信息 ≠ 模型更好**；要警惕模型学到“捷径特征”（shortcut）。
- 下一步方向：要么回到灰度 + 加入形态增强；要么保留颜色 + 加入颜色抖动/染色归一化，逼模型不依赖颜色。

## 重要提醒
- 到目前为止已多次使用 test 集评估。**test 只能用于最终报告**，后续调参必须先启用 val 集（PathMNIST 自带 10,004 张 val）。
- 明天任务：启用 val_loader，所有后续实验只看 val，不再反复看 test。

### 旧权重验证
- 新建 `evaluate_old_weight.py`，脚本内部自带旧 transform，不依赖 dataset.py 当前状态。
- 加载 `checkpoints/ResNet18_old.pth` + 旧 transform → Test Acc = 77.21%，与原始基线完全一致。
- 结论：**权重必须和训练时的 transform 严格匹配**，否则评估数字无效（如 VGG16 曾掉到 47.88%）。

### 当前两套基线
| 权重 | transform | Train | Test | 问题 |
|---|---|---:|---:|---|
| ResNet18_old.pth | 灰度化 | 91.80% | 77.21% | 丢颜色，泛化尚可 |
| ResNet18.pth | 保留 RGB | 96.18% | 65.45% | 过拟合，学染色偏色 |