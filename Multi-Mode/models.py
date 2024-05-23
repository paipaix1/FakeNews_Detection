import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from resnet_models import *


class SupConLoss(nn.Module):
    '''
    功能：在训练过程中，它会根据提供的标签或掩码来计算相似度，并使用对比学习的方法来优化模型
    :param features: 输入的特征向量，形状为(batch_size, feature_dim)
    :param labels: 标签，形状为(batch_size,)
    '''

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)


        # 计算两两样本间点乘相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  

        #计算每一行的最大值,并将其从相似度中减去
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)      # 求自然指数

        # 构建mask                                          #对角线为1 其余为0
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        # print(logits_mask)
        positives_mask = mask * logits_mask
        # print(positives_mask)
        # print('*******************')
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class Mynet(nn.Module):
    '''
    功能：实现一个基于BERT和ResNet的图像文本匹配模型
    '''
    def __init__(self,config):
        super(Mynet, self).__init__()
        self.config=config
        resnet_name=self.config.resnet_name#选取resnet种类
        if resnet_name=='resnet18':
            self.resnet=resnet18(self.config.resnet_fc)
        elif resnet_name=='resnet34':
            self.resnet=resnet34(self.config.resnet_fc)
        elif resnet_name=='resnet50':
            self.resnet=resnet50(self.config.resnet_fc)
        elif resnet_name=='resnet101':
            self.resnet=resnet101(self.config.resnet_fc)
        elif resnet_name=='resnet152':
            self.resnet=resnet152(self.config.resnet_fc)

        self.bert= BertModel.from_pretrained(self.config.bert_name)#bert的种类

        # 全连接层设置，输入维度为bert和resnet的特征维度之和，输出维度为类别数（即2）
        self.fc_1 = nn.Linear(self.config.bert_fc+self.config.resnet_fc, self.config.num_classes)
        self.drop=nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inx):
        '''
        :param inx: 是一个列表，其中包含图像特征、文本特征和掩码
        :param img: 图像特征,img.shape = torch.Size([32, 3, 224, 224])
        :param tokens: 文本特征,tokens.shape = torch.Size([32, 128])
        :param mask: 掩码矩阵,mask.shape = torch.Size([32, 128])
        '''
        # inx是一个列表，其中包含图像特征、文本特征和掩码
        # img：图像特征，tokens：文本特征，mask：掩码
        img,tokens,mask=inx

        # 使用ResNet模型对图像特征进行处理,img.shape = torch.Size([32, 256])
        img=self.resnet(img)
        # 使用BERT模型对文本特征进行处理，outputs = (last_hidden_state: torch.Size([32, 128, 256]),pooler_output: torch.Size([32, 256]),...)
        outputs = self.bert(tokens,attention_mask=mask)
        # pooled_output.shape = torch.Size([32, 256])
        pooled_output = outputs[1]
        pooled_output=self.drop(pooled_output)


        # 将BERT和ResNet的特征进行拼接,fea.shape = torch.Size([32, 512])
        fea=torch.cat([img,pooled_output],1)
        logits = self.fc_1(fea)             # 全连接层处理,logits.shape =  torch.Size([32, 2])       
        logits=self.softmax(logits)         # 得到分类类别,logits.shape =  torch.Size([32, 2])

        return img,logits#返回的第一个是需要对比的特征，img就为图像特征，fea就为全特征
