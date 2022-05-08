import paddle.nn as nn
import paddle.nn.functional as F


class myvgg(nn.Layer):
    def __init__(self, vgg):
        super(myvgg, self).__init__()
        self.vgg = vgg

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)
        x = self.vgg.features(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().transpose([1, 2, 0])
        
        return fc, att

