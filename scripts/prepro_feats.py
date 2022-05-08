from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import numpy as np
import paddle
import skimage.io
from paddle.vision.transforms import Normalize, CenterCrop, Resize, Compose
normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = Compose([Resize(size=256), CenterCrop(224)])


from misc.vgg_utils import myvgg
from paddle.vision.models import vgg19


def main(params):
    net = vgg19()
    net.load_dict(paddle.load(os.path.join(params['model_root'], 'vgg19-pt' + '.pdparams')))
    my_vgg = myvgg(net)
    my_vgg.eval()

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    dir_fc = params['output_dir'] + '_fc_14x14'
    dir_att = params['output_dir'] + '_att_14x14'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img in enumerate(imgs):
        # load the image
        I = skimage.io.imread(os.path.join(params['images_root'], img.get('filepath', ''), img['filename']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = transform(I).astype('float32') / 255.0
        I = normalize(I.transpose([2, 0, 1]))
        I = paddle.to_tensor(I)
        with paddle.no_grad():
            tmp_fc, tmp_att = my_vgg(I, params['att_size'])
        # write to pkl
        if 'cocoid' in img:
            np.save(os.path.join(dir_fc, str(img['cocoid'])), tmp_fc.numpy())
            np.savez_compressed(os.path.join(dir_att, str(img['cocoid'])), feat=tmp_att.numpy())
        elif 'imgid' in img:
            np.save(os.path.join(dir_fc, str(img['imgid'])), tmp_fc.numpy())
            np.savez_compressed(os.path.join(dir_att, str(img['imgid'])), feat=tmp_att.numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # input json
    parser.add_argument('--input_json', default='D:/dataset/caption_datasets/dataset_flickr8k.json', help='input json file to '
                                                                                                          'process into hdf5')
    parser.add_argument('--output_dir', default='D:/dataset/f8k_data_vgg/f8ktalk', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='D:/dataset/Flicker8k_Dataset',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model_root', default='D:/dataset/imagenet_weights', type=str, help='model root')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
