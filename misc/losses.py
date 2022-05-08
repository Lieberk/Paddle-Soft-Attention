import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LanguageModelCriterion(nn.Layer):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]

        target = F.one_hot(target, input.shape[-1])
        output = -input.multiply(target).sum(-1) * mask
        output = paddle.sum(output) / paddle.sum(mask)

        return output