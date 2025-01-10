# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import *
import torchvision.models as models
from torch import Tensor


class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        # print(('x1', x.shape), flush=True)
        x = self.base_model.layer2(x)
        # print(('x2', x.shape), flush=True)
        x = self.base_model.layer3(x)
        # print(('x3', x.shape), flush=True)
        x = self.base_model.layer4(x)
        # print(('x4', x.shape), flush=True)
        return x


class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    # normlized global_query:B, D
    # normlized value: B, D, H, W
    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        # print("global_query:{}".format(global_query.shape))
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        # print("value:{}".format(value.shape))
        new_value = value.permute(0, 2, 3, 1).view(B, W * H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1, 2))
        score = score.view(B, W * H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)

        attn = Variable(torch.zeros(B, H * W).cuda())
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])

        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn


class GCFAgg(nn.Module):
    def __init__(self, in_dim):
        super(GCFAgg, self).__init__()
        D = in_dim
        self.Wq1 = nn.Linear(D, D)
        self.Wq2 = nn.Linear(D, D)
        self.WR = nn.Linear(D, D)

    def forward(self, x):
        q1 = self.Wq1(x)
        q2 = self.Wq2(x)
        s = torch.bmm(q1, q2.permute(0, 2, 1))
        r = self.WR(x)
        out = torch.bmm(s, r)
        return out


class AdvanceModel(nn.Module):
    def __init__(self, emb_size=512, leaky=True):
        super(AdvanceModel, self).__init__()
        ## Visual model
        self.query_resnet = MyResnet()

        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')

        use_instnorm = False

        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.gcf_agg = GCFAgg(128)
        self.crossview_fusionmodule = CrossViewFusionModule()

        self.query_visudim = 512
        self.reference_visudim = 512

        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky,
                                                    instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky,
                                                        instance=use_instnorm)
        ## output head
        self.fcn_out = torch.nn.Sequential(
            ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1))

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        # print("query_imgs:{}".format(query_imgs.shape))
        # print("reference_imgs:{}".format(reference_imgs.shape))
        mat_clickptns = mat_clickptns.unsqueeze(1)
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))
        query_fvisu = self.query_resnet(query_imgs)
        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        reference_fvisu = reference_raw_fvisu[1]

        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)

        B, D, Hquery, Wquery = query_fvisu.shape
        B, D, Hreference, Wreference = reference_fvisu.shape

        # cross-view fusion
        # 输入的特征图分块计算平均值
        query_gvisu = query_fvisu.view(B, D, 4, Hquery // 4, 4, Wquery // 4).permute(0, 1, 2, 4, 3, 5).reshape(B, D, -1)
        last_dim_len = query_fvisu.shape[-1]
        # print("query_gvisu:{}".format(query_gvisu.shape))
        query_gvisu = self.gcf_agg(query_gvisu)
        query_gvisu = torch.mean(query_gvisu, dim=-1)
        # print("gcf_shape:{}".format(query_gvisu.shape))
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        attn_score = attn_score.squeeze(1)
        outbox = self.fcn_out(fused_features)

        return outbox, attn_score


# 输入图片的形状
# 查询图片：（3,256,256）
# 参考图片：（3,1024,1024）
# query_fvisu:torch.Size([3, 512, 8, 16])
# reference_fvisu:torch.Size([3, 512, 64, 64])
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvanceModel()
    query_img = torch.randn(3, 256, 256)
    reference_img = torch.randn(3, 1024, 1024)
