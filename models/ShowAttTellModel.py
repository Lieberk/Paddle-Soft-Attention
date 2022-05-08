import paddle
import paddle.nn as nn
from .AttModel import Attention, AttModel
import paddle.nn.functional as F


class LSTMCore(nn.Layer):
    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size + self.rnn_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(paddle.concat([xt, att_res], 1)) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.slice([1], [0], [3 * self.rnn_size])
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.slice([1], [0], [self.rnn_size])
        forget_gate = sigmoid_chunk.slice([1], [self.rnn_size], [self.rnn_size * 2])
        out_gate = sigmoid_chunk.slice([1], [self.rnn_size * 2], [self.rnn_size * 3])

        in_transform = F.tanh(all_input_sums.slice([1], [3 * self.rnn_size], [4 * self.rnn_size]))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class ShowAttTellModel(AttModel):
    def __init__(self, opt):
        super(ShowAttTellModel, self).__init__(opt)
        del self.embed, self.fc_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = lambda x: x
        self.core = LSTMCore(opt)
