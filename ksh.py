import paddle
from paddleseg.models import STDC2,PPNewSeg3,PPNewSeg2,PPNewSeg,DSMONet_B
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize
trained_model_path = 'output\dsmonet-b\model.pdparams'
# trained_model_path = 'output/dsmonet/test/423/200000/best_model/model.pdparams'
# trained_model_path = 'output/test/16/200000/best_model/model.pdparams'
# trained_model_path = '../output/ppliteseg/78.97/model.pdparams'
# trained_model_path = '../output/cs+dappm+oce+adam/model.pdparams'
# trained_model_path = 'https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet50_os8_voc12aug_512x512_40k/model.pdparams'
backbone = STDC2()
m = DSMONet_B(num_classes=19,
               backbone=backbone, 
               pretrained=trained_model_path)
utils.utils.load_entire_model(m, trained_model_path)

m.eval()


# import numpy as np
import paddle
paddle.set_device('gpu:0')

from interpretdl.data_processor.readers import read_image, images_transform_pipeline

# img_path = 'frankfurt_000001_011835_leftImg8bit.png'
# img_path = 'frankfurt_000000_000294_leftImg8bit.png'
img_path = 'data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_011835_leftImg8bit.png'
# frankfurt_000001_072295_leftImg8bit
# frankfurt_000000_010351_leftImg8bit
# frankfurt_000001_044787_leftImg8bits
# frankfurt_000001_053102_leftImg8bit
# munster_000005_000019_leftImg8bit
# lindau_000018_000019_leftImg8bit
#lindau_000013_000019_leftImg8bit

uint8_img, float_input_data = images_transform_pipeline(img_path, resize_to=1024, crop_to=None)
input_data = paddle.to_tensor(float_input_data)
r = m(input_data)

import matplotlib.pyplot as plt

c = r[0][0,11].numpy()  # preson
plt.imsave(f'output/ksh/out.png',c)

# for i in range(19):
#     # c = r[0,i].numpy()
#     c = r[0][0,i].numpy()
#     plt.imsave(f'output/result/dsmonet/423/out{i}.png',c)