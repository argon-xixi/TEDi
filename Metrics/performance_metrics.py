import torch
import torch.nn.functional as F


class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score

class BinaryDiceScore(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=1):
        super(BinaryDiceScore, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        # loss = 1 - dice
        return dice

class MultiDiceScore(torch.nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiDiceScore, self).__init__()
        self.kwargs = kwargs
        self.weights = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        nclass = predict.shape[1]
        predict = torch.argmax(predict, dim=1)
        # predict = predict.squeeze()
        predict = torch.nn.functional.one_hot(predict.long(), nclass)
        predict = torch.transpose(torch.transpose(predict, 1, 3), 2, 3)
        target = torch.squeeze(target, 1)
        target = torch.nn.functional.one_hot(target.long(), nclass)  # [1, 4]->[1, 4, 5]
        target = torch.transpose(torch.transpose(target, 1, 3), 2, 3)
        # target = torch.transpose(target, 1, 2)

        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceScore(**self.kwargs)
        total_loss = 0
        # predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weights is not None:
                    assert self.weights.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1] if self.weights is None else total_loss / (torch.sum(self.weights))


