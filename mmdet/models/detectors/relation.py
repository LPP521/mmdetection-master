import pdb
import argparse
import os
from mmcv import Config
import torch.nn as nn
import torch
from mmdet.core.utils import multi_apply
import torch.nn.functional as F
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='mmdet detector')
    parser.add_argument('config', help='config file path')

    # 为保证不报错 添加命令行关键字 以下并无实际作用
    parser.add_argument('--launcher')
    parser.add_argument('--local_rank')

    parser.add_argument('--checkpoint')
    parser.add_argument('--gpus')
    parser.add_argument('--out')
    parser.add_argument('--eval')
    parser.add_argument('--load_pkl')

    args = parser.parse_args()

    return  args

args = parse_args()
mycfg = Config.fromfile(args.config)

class relation_module(nn.Module):
    def __init__(self, dim_in, *dim_out):
        super(relation_module, self).__init__()
        # 维度对应relation network 源码
        # em_dim group 相互独立 互不影响
        # 空间关系编码维度
        self.em_dim = 64  # 64
        # 论文里分组进行卷积（待理解）
        self.group = 16
        # Wg, Wq, Wk 对应论文
        self.Wg = nn.Linear(self.em_dim, self.group)
        # inplace 应设为 False (原因未知)
        self.relu_part = nn.ReLU(inplace=False)
        # 增强后的特征维数
        self.context_rela_dim = self.group * self.em_dim  # 1024
        # box_feat
        self.Wq = nn.Linear(dim_in, self.context_rela_dim)
        # part_feat  part_feat和box_feat 会通过一样的roi pooling 维度一样
        self.Wk = nn.Linear(dim_in, self.context_rela_dim)
        self.conv_1x1 = torch.nn.Conv2d(self.group * dim_in,
            self.group * self.em_dim, kernel_size=1, stride=1, groups=self.group)

        self.feat_dim = dim_in + self.context_rela_dim
        self.fc_cls = nn.Linear(self.feat_dim, dim_out[0])
        self.fc_reg = nn.Linear(self.feat_dim, dim_out[1])

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv_1x1.weight, mean=0.0, std=0.02)
        self.fc_init(self.Wg)
        self.fc_init(self.Wq)
        self.fc_init(self.Wk)

    def fc_init(self, w):
        nn.init.xavier_uniform_(w.weight)
        nn.init.constant_(w.bias, 0)

    # 提取object与part的空间关系
    # 对应论文公式（5）中的 (fgm, fgn)
    def extract_position_matrix(self, boxes, part_boxes):
        # 参考 relation network 源码
        # 获取坐标
        xmin, ymin, xmax, ymax = torch.split(boxes, [1, 1, 1, 1], dim=1)
        xmin_part, ymin_part, xmax_part, ymax_part = torch.split(part_boxes, [1, 1, 1, 1], dim=1)
        # 获取box和part的中心 获取框的宽高
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1
        part_width = xmax_part - xmin_part + 1
        part_height = ymax_part - ymin_part + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        center_x_part = 0.5 * (xmin_part + xmax_part)
        center_y_part = 0.5 * (ymin_part + ymax_part)
        # log(|xm - xn| / wm) log(|ym - yn| / hm)
        center_x_part = torch.transpose(center_x_part, 1, 0)
        center_y_part = torch.transpose(center_y_part, 1, 0)
        delta_x = torch.abs((center_x - center_x_part) / box_width)
        delta_y = torch.abs((center_y - center_y_part) / box_height)
        delta_x[delta_x < 1e-3] = 1e-3
        delta_y[delta_y < 1e-3] = 1e-3
        delta_x = torch.log(delta_x)
        delta_y = torch.log(delta_y)
        # log(wn / wm) log(hn / hm)
        part_width = torch.transpose(part_width, 1, 0)
        part_height = torch.transpose(part_height, 1, 0)
        delta_width = part_width / box_width
        delta_height = part_height / box_height
        delta_width = torch.log(delta_width)
        delta_height = torch.log(delta_height)

        # 增加维度 并cat起来
        cat_list = [delta_x, delta_y, delta_width, delta_height]
        for i in range(len(cat_list)):
            cat_list[i] = cat_list[i].unsqueeze(-1)
        position_matrix = torch.cat((cat_list[0], cat_list[1]), dim=2)
        for i in range(2, len(cat_list)):
            position_matrix = torch.cat((position_matrix, cat_list[i]), dim=2)

        # position_matrix [box_num, part_num, 4]
        return  position_matrix

    # 位置矩阵编码
    # 对应论文公式(5)中的 Eg(fgm, fgn) 升维编码
    def extract_position_embedding(self, position_matrix, em_dim, wave_length=1000):
        em_range = torch.arange(0, em_dim / 8) / (em_dim // 8)  # float 类型
        em_range = em_range.cuda(torch.cuda.current_device())
        # 参考源码 dim_mat=[1., 2.37137365, 5.62341309,
        # # 13.33521461, 31.62277603, 74.98941803, 177.82794189, 421.69650269]
        em_mat = torch.pow(wave_length, em_range)
        # dim_mat [1, 1, 1, 8]
        em_mat = em_mat.view((1, 1, 1, -1))
        position_matrix = 100 * position_matrix
        # [512, 2, 4, 1]
        position_matrix = position_matrix.unsqueeze(-1)
        # div_mat [512, 2, 4, 8]
        div_mat = position_matrix / em_mat
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        # embedding [512, 2, 4, 16]
        embedding = torch.cat((sin_mat, cos_mat), dim=3)
        # embedding [512, 2, 64]
        embedding = embedding.view((embedding.size(0), embedding.size(1), em_dim))

        return embedding

    # 提取part与relation的关系 提取方式可查阅论文
    def extract_part_object_relation(self, box_feats, part_feats, position_embedding):
        # position_embedding [512, 2, 64]
        box_dim = position_embedding.size(0)
        part_dim = position_embedding.size(1)
        feat_dim = position_embedding.size(2)
        # 前两维铺平 [1024, 64]
        position_embedding = position_embedding.view((box_dim * part_dim, feat_dim))
        # [1024, 16] fc_dim # 对应公式（5）中的可学矩阵Wg relu相当于max作用
        position_feat = self.relu_part(self.Wg(position_embedding))
        # [512, 2, 16]
        att_weight = position_feat.view((box_dim, part_dim, self.group))
        # [512, 2, 14] 对应公式（5）计算后的 Wg
        att_weight = torch.transpose(att_weight, 2, 1)

        # query: box 部分的feature
        # [512, 1024, 1, 1] - > [512, 1024]
        # box_feats = box_feats.squeeze(3).squeeze(2)
        # [512, 2014] - > [512, 1024] 对应公式（4）中的Wq（全连接层参数）
        q_data = self.Wq(box_feats)
        # 按group分组
        # [512, 1024] - > [512, 16, 64]
        q_data_batch = q_data.view((-1, self.group, q_data.size(-1) // self.group))
        # [512, 16, 64] - > [16, 512, 64]
        q_data_batch = torch.transpose(q_data_batch, 1, 0)

        # key: part 部分的feature 对应公式（4）中的Wk
        # part_feats = part_feats.squeeze(3).squeeze(2)
        # [2, 1024] - > [2, 1024] 对应公式（4）中的Wk（全连接层参数）
        k_data = self.Wk(part_feats)
        # [2, 1024] - > [2, 16, 64]
        k_data_batch = k_data.view((-1, self.group, k_data.size(-1) // self.group))
        # [2, 16, 64] - > [16, 64, 2]
        k_data_batch = k_data_batch.transpose(1, 0).transpose(1, 2)
        # vaule: [2, 1024]
        v_data = part_feats
        # 对应公式（4）中的计算后的 Wa
        att = torch.matmul(q_data_batch, k_data_batch)
        # 尺度变化
        att_scale = (1.0 / self.group) * att

        # [512, 16, 2] 对应公式（4）计算后的Wa
        att_scale = att_scale.transpose(1, 0)

        att_weight[att_weight < 1e-6] = 1e-6
        # 数学运算小技巧
        weighted_att = torch.log(att_weight) + att_scale
        # 对应公式（3）中计算后的W
        att_softmax = F.softmax(weighted_att, dim=2)
        # pdb.set_trace()
        # [512 * 16, 2]
        att_softmax = att_softmax.view(-1, att_softmax.size(-1))
        # [512 * 16, 2] dot [2, 1024] -> [512*16, 1024]
        output = torch.matmul(att_softmax, v_data)
        # [512*16, 1024] -> [512, 16 * 1024, 1, 1]
        output = output.view((box_dim, -1, 1, 1))
        # 卷积层参数对应 Wv
        # [512, 16 * 1024, 1, 1] -> [512, 16 * 64, 1, 1]
        output = self.conv_1x1(output)

        return output

    # relation network 论文 object part 关系
    def single_get_relation(self, boxes, part_boxes, box_feats, part_feats):
        # device_id = torch.cuda.current_device()
        # boxes = torch.from_numpy(boxes).cuda(device_id) # .detach()
        # part_boxes = torch.from_numpy(part_boxes).cuda(device_id)
        # pdb.set_trace()
        position_matrix = self.extract_position_matrix(boxes, part_boxes)
        position_embedding = self.extract_position_embedding(position_matrix, self.em_dim)
        context_relation = self.extract_part_object_relation(box_feats, part_feats, position_embedding)

        return context_relation

    def get_relation(self, rois, part_rois, box_feats, part_feats):
        device_id = torch.cuda.current_device()
        boxes = rois[:, 1:5]
        part_boxes = part_rois[:, 1:5]
        # 在detectron框架中 经常会有一个batch中采集不够512个框的情况 需要分开处理
        # 在mmdet中也做同样的处理
        box_num1 = int(torch.sum(rois[:, 0] == 0))
        box_num2 = int(rois.size(0)) - box_num1
        part_num1 = int(torch.sum(part_rois[:, 0] == 0))
        # part_num2 = int(part_rois.shape[0]) - part_num1

        temp_boxes = [boxes[:box_num1], boxes[box_num1:]]
        temp_part_boxes = [part_boxes[:part_num1], part_boxes[part_num1:]]
        temp_box_feats = [box_feats[:box_num1], box_feats[box_num1:]]
        temp_part_feats = [part_feats[:part_num1], part_feats[part_num1:]]

        # relation = multi_apply(self.single_get_relation, temp_boxes, temp_part_boxes,
        #                 temp_box_feats, temp_part_feats)

        relation = torch.zeros((box_num1 + box_num2, self.context_rela_dim, 1, 1))
        box_nums = [0, box_num1, box_num1 + box_num2]
        batch_size=2 if box_num2 != 0 else 1
        for i in range(batch_size):
            temp_relation = self.single_get_relation(temp_boxes[i], temp_part_boxes[i],
                                        temp_box_feats[i], temp_part_feats[i])
            relation[box_nums[i]: box_nums[i + 1]] = temp_relation

        relation = relation.cuda(device_id)
        relation = relation.squeeze(3).squeeze(2)

        return  relation