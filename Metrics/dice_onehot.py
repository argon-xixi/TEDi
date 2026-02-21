import torch


# 输入为非onehot编码的预测结果和标签
def dice_coeff(pred, target,num_cls=2,epsilon=1e-6): # 此处的numcls算上0标签
    # 使用one-hot编码计算dice值和直接argmax后计算没有区别（两者当针对标签1时输入和输出相同，但smooth取值会略微改变结果）
    
    # target_one_hot = F.one_hot(target.squeeze(1), num_classes=num_cls)  # (batchsize, h, w, numcls)
    # target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (batchsize, numcls, h, w)
    # # pred_probs = F.softmax(pred, dim=1)  # 如果是 logits，先通过 softmax 转为概率
    # pred_binary = (pred_probs > 0.5).float()  # 二值化，阈值可以调整
    mdice=0
    pred = torch.argmax(pred, dim=1)
    # predict = predict.squeeze()
    pred = torch.nn.functional.one_hot(pred.long(), num_cls)
    # 在最后一个维度填充为onehot编码
    pred_binary = torch.transpose(torch.transpose(pred, 1, 3), 2, 3)#将onehot编码挪到dim=1
    intersection = torch.sum(pred_binary * target, dim=(0, 2, 3))  # 按类别计算交集
    pred_sum = torch.sum(pred_binary, dim=(0, 2, 3))  # 按类别计算 pred 总和
    target_sum = torch.sum(target, dim=(0, 2, 3))  # 按类别计算 label 总和

    dice_scores = (2.0 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
    # 输出是每一类的dice值，二分类中通常计算正类（1）
    for i in range(1,num_cls):
        mdice+=dice_scores[i]
    return mdice/(num_cls-1)