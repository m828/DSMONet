# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils
from paddleseg.models.layers import tensor_fusion_helper as helper
from paddle.static import InputSpec

@manager.MODELS.add_component
class DSMONet_T(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[1, 2, 3, 4],
                 out_ch=128,
                 pretrained=None):
        super().__init__()

        # backbone
        self.backbone = backbone

        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]
    
        self.cm = DAPPM(backbone_out_chs[-1],out_ch,out_ch)
        self.squeeze_body_edge = SqueezeBodyEdge(128)

        self.seg_heads = SegHead(128, 64, num_classes)

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

        self.edge_fusions = nn.Sequential(
            nn.Conv2D(128 + 128, 128, 1, bias_attr=False),
            layers.SyncBatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 1, bias_attr=False),
            layers.SyncBatchNorm(128),
            nn.ReLU())

        self.se = SELayer(128,128)
 
        self.edge_fusion = nn.Sequential(
            nn.Conv2D(128 + 128 +128, 128, 1, bias_attr=False),
            layers.SyncBatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 1, bias_attr=False),
            layers.SyncBatchNorm(128),
            nn.ReLU())

        self.arm1 = UAFM(1024, 128, True)
        self.arm2 = UAFM(128, 128, False)

        self.bot_fine = ConvBNRelu(64, 128, 3, 2, 1.0)
        self.laplacian = Laplacian(128,128)
        self.conv_up = nn.Sequential(
            nn.Conv2DTranspose(128,256,2,2,bias_attr=False),
            nn.ReLU(),
            nn.Conv2DTranspose(256,128,2,2,bias_attr=False),
            nn.ReLU(),
            nn.Conv2DTranspose(128,128,2,2,bias_attr=False),
            nn.ReLU())

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        high_feats = self.cm(feats_selected[-1])
        high_feat = self.arm1(feats_selected[-1], high_feats)
        head_seg3 = high_feat

        # 语义调整细节
        seg_body, seg_edge = self.squeeze_body_edge(high_feat)

        edge_shallow  = self.bot_fine(feats_selected[0])
        #Laplacian conv优化浅层特征
        edge_laplacian = self.laplacian(edge_shallow)
        edge_deep = self.conv_up(seg_edge)

        high_feat_se = self.se(high_feat)
        feat_se = self.conv_up(high_feat_se)
        #####方便可视化：
        seg_edgess = self.edge_fusions(paddle.concat([edge_deep, edge_laplacian],axis=1))
        seg_edge_ksh = seg_edge
        #######
        seg_edge = self.edge_fusion(paddle.concat([paddle.concat([edge_deep, edge_laplacian], axis=1), feat_se], axis=1))
        head_seg2 = seg_edge

        # 细节指导语义
        high_feats = high_feat + seg_body + high_feat_se
        high_feat = self.conv_up(high_feats)
        out = self.arm2(seg_edge, high_feat)
        head_seg1 = out

        if self.training:
            logit_list = []

            x1 = self.seg_heads(head_seg1)
            logit_list.append(x1)

            x2 = self.seg_heads(head_seg2)
            logit_list.append(x2)

            x3 = self.seg_heads(head_seg3)
            logit_list.append(x3)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.seg_heads(head_seg1)
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

class Laplacian(nn.Layer):
    def __init__(self,in_ch, out_ch):
        super(Laplacian, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv_op = nn.Conv2D(in_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3

    def forward(self, input): 
        laplacian_kernel = self.laplacian_kernel.reshape((1, 1, 3, 3))
        laplacian_kernel = np.repeat(laplacian_kernel, self.in_ch, axis=1)
        laplacian_kernel = np.repeat(laplacian_kernel, self.out_ch, axis=0)
        self.conv_op.weight.data = InputSpec.from_numpy(laplacian_kernel)
        edge_detect = self.conv_op(input)
        return edge_detect + input
        
class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio=16):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = paddle.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out

class UAFM(nn.Layer):
    def __init__(self, x_ch, y_ch, tokens=True, ksize=3):
        super(UAFM, self).__init__()

        self.tokens = tokens

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, 128, kernel_size=3, padding=1, bias_attr=False)

        self.conv_xy_atten1 = nn.Sequential(
            layers.ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

        self.conv_xy_atten2 = nn.Sequential(
            layers.ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))

        self.conv_out = layers.ConvBNReLU(
            y_ch, y_ch, kernel_size=3, padding=1, bias_attr=False)

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        x = self.conv_x(x)
        if self.tokens:
            out = self.fuse(x, y)
        else:
            out = self.fuse_onlys(x, y)
        return out

    def fuse(self, x, y):

        atten1 = helper.avg_max_reduce_hw([x, y], self.training) # n*4c*1*1
        atten1 = F.sigmoid(self.conv_xy_atten1(atten1)) # n*c*1*1

        atten2 = helper.avg_max_reduce_channel([x, y]) # n*4*h*w
        atten2 = F.sigmoid(self.conv_xy_atten2(atten2)) # n*1*h*w

        out1 = x * atten1 + y * (1 - atten1) # n*c*h*w
        # out1 = self.conv_out(out1)

        out2 = x * atten2 + y * (1 - atten2) # n*c*h*w
        # out2 = self.conv_out(out2)

        out = out1 + out2
        out = self.conv_out(out)
        # out = paddle.maximum(out1, out2)
        return out

    def fuse_onlys(self, x, y):

        atten = helper.avg_max_reduce_channel([x, y])  # n*4*h*w
        atten = F.sigmoid(self.conv_xy_atten2(atten))  # n*1*h*w

        out = x * atten + y * (1 - atten)  # n*c*h*w
        out = self.conv_out(out)
        return out

    def fuse_onlyc(self, x, y):

        atten = helper.avg_max_reduce_hw([x, y], self.training)  # n*4c*1*1
        atten = F.sigmoid(self.conv_xy_atten1(atten))  # n*c*1*1

        out = x * atten + y * (1 - atten)  # n*c*h*w
        out = self.conv_out(out)
        return out


class SqueezeBodyEdge(nn.Layer):
    def __init__(self, inplane):
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2D(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            layers.SyncBatchNorm(inplane),
            nn.ReLU(),
            nn.Conv2D(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            layers.SyncBatchNorm(inplane),
            nn.ReLU()
        )
        self.flow_make = nn.Conv2D(inplane * 2, 2, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x):
        size = paddle.shape(x)[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(paddle.concat([x, seg_down], axis=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        input_shape = paddle.shape(input)
        # norm = size[::-1].reshape([1, 1, 1, -1])
        norm = paddle.flip(size, axis=0).reshape([1, 1, 1, -1])
        norm.stop_gradient = True
        h_grid = paddle.linspace(-1.0, 1.0, size[0]).reshape([-1, 1])
        h_grid = h_grid.tile([1, size[1]])
        w_grid = paddle.linspace(-1.0, 1.0, size[1]).reshape([-1, 1])
        w_grid = w_grid.tile([1, size[0]]).transpose([1, 0])
        grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)
        grid.unsqueeze(0).tile([input_shape[0], 1, 1, 1])
        grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

        # output = F.grid_sample(input, grid)
        output = self.paddle_bilinear_grid_sample(input, grid, align_corners=True)
        return output

    def paddle_bilinear_grid_sample(self, im, grid, align_corners=False):
        # this code reference: https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/point_sample.html
        im_shape = paddle.shape(im)
        n, c, h, w = paddle.split(im_shape, num_or_sections=4)
        grid_shape = paddle.shape(grid)
        gn, gh, gw, _ = paddle.split(grid_shape, num_or_sections=4)

        # n, c, h, w = im.shape
        # gn, gh, gw, _ = grid.shape
        # assert n == gn

        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]

        if align_corners:
            x = ((x + 1) / 2) * (w - 1)
            y = ((y + 1) / 2) * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2

        x = paddle.reshape(x, [n, -1])
        y = paddle.reshape(y, [n, -1])

        x0 = paddle.floor(x).astype('float32')
        y0 = paddle.floor(y).astype('float32')
        x1 = x0 + 1
        y1 = y0 + 1

        x1_cast = x1.astype(grid.dtype)
        x0_cast = x0.astype(grid.dtype)
        y1_cast = y1.astype(grid.dtype)
        y0_cast = y0.astype(grid.dtype)
        wa = paddle.unsqueeze(((x1_cast - x) * (y1_cast - y)), 1)
        wb = paddle.unsqueeze(((x1_cast - x) * (y - y0_cast)), 1)
        wc = paddle.unsqueeze(((x - x0_cast) * (y1_cast - y)), 1)
        wd = paddle.unsqueeze(((x - x0_cast) * (y - y0_cast)), 1)

        # Apply default for grid_sample function zero padding
        im_padded = paddle.nn.functional.pad(im,
                                             pad=[1, 1, 1, 1],
                                             mode='constant',
                                             value=0)
        if im_padded.dtype != im.dtype:
            im_padded = paddle.cast(im_padded, im.dtype)
        padded_h = h + 2
        padded_w = w + 2
        # save points positions after padding
        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

        # Clip coordinates to padded image size
        tensor_zero = paddle.full(shape=[1], dtype='float32', fill_value=0.0)
        tensor_padded_w = paddle.full(
            shape=[1], dtype='float32', fill_value=padded_w - 1)
        tensor_padded_h = paddle.full(
            shape=[1], dtype='float32', fill_value=padded_h - 1)
        x0 = paddle.where(x0 < 0, tensor_zero, x0)
        x0 = paddle.where(x0 > padded_w - 1, tensor_padded_w, x0)
        x1 = paddle.where(x1 < 0, tensor_zero, x1)
        x1 = paddle.where(x1 > padded_w - 1, tensor_padded_w, x1)
        y0 = paddle.where(y0 < 0, tensor_zero, y0)
        y0 = paddle.where(y0 > padded_h - 1, tensor_padded_h, y0)
        y1 = paddle.where(y1 < 0, tensor_zero, y1)
        y1 = paddle.where(y1 > padded_h - 1, tensor_padded_h, y1)
        im_padded = paddle.reshape(im_padded, [n, c, -1])

        x0_y0 = paddle.expand(
            paddle.unsqueeze((x0 + y0 * padded_w), 1), [-1, c, -1]).astype('int64')
        x0_y1 = paddle.expand(
            paddle.unsqueeze((x0 + y1 * padded_w), 1), [-1, c, -1]).astype('int64')
        x1_y0 = paddle.expand(
            paddle.unsqueeze((x1 + y0 * padded_w), 1), [-1, c, -1]).astype('int64')
        x1_y1 = paddle.expand(
            paddle.unsqueeze((x1 + y1 * padded_w), 1), [-1, c, -1]).astype('int64')

        # Ia = self.gather(im_padded, 2, x0_y0)
        # Ib = self.gather(im_padded, 2, x0_y1)
        # Ic = self.gather(im_padded, 2, x1_y0)
        # Id = self.gather(im_padded, 2, x1_y1)
        Ia = paddle.take_along_axis(im_padded, x0_y0, 2).astype('float32')
        Ib = paddle.take_along_axis(im_padded, x0_y1, 2).astype('float32')
        Ic = paddle.take_along_axis(im_padded, x1_y0, 2).astype('float32')
        Id = paddle.take_along_axis(im_padded, x1_y1, 2).astype('float32')

        return paddle.reshape((Ia * wa + Ib * wb + Ic * wc + Id * wd),
                              [n, c, gh, gw])
    def gather(self, x, dim, index):
        # index_shape = index.shape
        index_shape = paddle.shape(index)
        x_shape = paddle.shape(x)
        index_flatten = index.flatten()
        if dim < 0:
            dim = len(x.shape) + dim
        nd_index = []
        for k in range(len(x.shape)):
            if k == dim:
                nd_index.append(index_flatten)
            else:
                reshape_shape = [1] * len(x.shape)
                x_shape_k = x_shape[k]
                # x_shape_k = x.shape[k]
                reshape_shape[k] = x_shape[k]
                x_arange = paddle.arange(x_shape_k, dtype=index.dtype)
                x_arange = x_arange.reshape(reshape_shape)
                dim_index = paddle.expand(x_arange, index_shape).flatten()
                nd_index.append(dim_index)
        ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
        paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
        return paddle_out

    def bilinear_grid_sample(self, im, grid, align_corners=False):
        # n, c, h, w = paddle.shape(im)
        # gn, gh, gw, _ = paddle.shape(grid)

        im_shape = paddle.shape(im)
        n, c, h, w = paddle.split(im_shape, num_or_sections=4)
        grid_shape = paddle.shape(grid)
        gn, gh, gw, _ = paddle.split(grid_shape, num_or_sections=4)

        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]

        if align_corners:
            x = ((x + 1) / 2) * (w - 1)
            y = ((y + 1) / 2) * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2

        x = paddle.reshape(x, [n, -1])
        y = paddle.reshape(y, [n, -1])

        x0 = paddle.floor(x).astype('long')
        y0 = paddle.floor(y).astype('long')
        # y0 = paddle.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1

        # wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
        # wb = ((x1 - x) * (y - y0)).unsqueeze(1)
        # wc = ((x - x0) * (y1 - y)).unsqueeze(1)
        # wd = ((x - x0) * (y - y0)).unsqueeze(1)
        wa = paddle.unsqueeze(((x1 - x) * (y1 - y)), 1)
        wb = paddle.unsqueeze(((x1 - x) * (y - y0)), 1)
        wc = paddle.unsqueeze(((x - x0) * (y1 - y)), 1)
        wd = paddle.unsqueeze(((x - x0) * (y - y0)), 1)

        # Apply default for grid_sample function zero padding
        im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
        padded_h = h + 2
        padded_w = w + 2
        # save points positions after padding
        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

        # Clip coordinates to padded image size
        x0 = paddle.where(x0 < 0, paddle.to_tensor(0), x0)
        x0 = paddle.where(x0 > padded_w - 1, paddle.to_tensor(padded_w - 1), x0)
        x1 = paddle.where(x1 < 0, paddle.to_tensor(0), x1)
        x1 = paddle.where(x1 > padded_w - 1, paddle.to_tensor(padded_w - 1), x1)
        y0 = paddle.where(y0 < 0, paddle.to_tensor(0), y0)
        y0 = paddle.where(y0 > padded_h - 1, paddle.to_tensor(padded_h - 1), y0)
        y1 = paddle.where(y1 < 0, paddle.to_tensor(0), y1)
        y1 = paddle.where(y1 > padded_h - 1, paddle.to_tensor(padded_h - 1), y1)

        im_padded = paddle.reshape(im_padded, [n, c, -1])

        # x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        # x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
        # x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        # x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x0_y0 = paddle.expand(
            paddle.unsqueeze((x0 + y0 * padded_w), 1), [-1, c, -1])
        x0_y1 = paddle.expand(
            paddle.unsqueeze((x0 + y1 * padded_w), 1), [-1, c, -1])
        x1_y0 = paddle.expand(
            paddle.unsqueeze((x1 + y0 * padded_w), 1), [-1, c, -1])
        x1_y1 = paddle.expand(
            paddle.unsqueeze((x1 + y1 * padded_w), 1), [-1, c, -1])

        Ia = self.gather(im_padded, 2, x0_y0)
        Ib = self.gather(im_padded, 2, x0_y1)
        Ic = self.gather(im_padded, 2, x1_y0)
        Id = self.gather(im_padded, 2, x1_y1)

        return paddle.reshape((Ia * wa + Ib * wb + Ic * wc + Id * wd),
                              [n, c, gh, gw])

class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=5, stride=2, padding=2),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale2 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=9, stride=4, padding=4),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale3 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=17, stride=8, padding=8),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale0 = nn.Sequential(
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.process1 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process2 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process3 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process4 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.compression = nn.Sequential(
            layers.SyncBatchNorm(branch_planes * 5),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes * 5, outplanes, kernel_size=1, bias_attr=False))
        self.shortcut = nn.Sequential(
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, outplanes, kernel_size=1, bias_attr=False))

    def forward(self, x):
        n, c, h, w = paddle.shape(x)
        x0 = self.scale0(x)
        x1 = self.process1(
            F.interpolate(
                self.scale1(x), size=[h, w], mode='bilinear') + x0)
        x2 = self.process2(
            F.interpolate(
                self.scale2(x), size=[h, w], mode='bilinear') + x1)
        x3 = self.process3(
            F.interpolate(
                self.scale3(x), size=[h, w], mode='bilinear') + x2)
        x4 = self.process4(
            F.interpolate(
                self.scale4(x), size=[h, w], mode='bilinear') + x3)

        out = self.compression(paddle.concat([x0, x1, x2, x3, x4],
                                             1)) + self.shortcut(x)
        return out

class ConvBNRelu(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel=3,
                 stride=1,
                 relative_lr=1.0):
        super(ConvBNRelu, self).__init__()
        param_attr = paddle.ParamAttr(learning_rate=relative_lr)
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            weight_attr=param_attr,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(
            out_planes, weight_attr=param_attr, bias_attr=param_attr)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

