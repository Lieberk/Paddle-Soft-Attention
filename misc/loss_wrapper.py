import paddle
import misc.losses as loss


class LossWrapper(paddle.nn.Layer):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = loss.LanguageModelCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts):
        out = {}
        loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
        out['loss'] = loss
        return out