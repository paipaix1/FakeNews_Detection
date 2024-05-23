
import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo


# 3x3卷积的卷积结构
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# 残差网络中的basicblock结构
class BasicBlock(nn.Module):
    '''
    功能：实现残差网络中的basicblock结构
    :param inplanes: 输入通道数
    :param planes: 输出通道数
    :param stride: 步长
    '''
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Conv1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Conv2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    '''
    功能：实现残差网络中的瓶颈层bottleneck结构
    '''
    expansion = 4      # 输出通道数的倍乘
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # conv1   1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # conv2   3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # conv3   1x1
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
    功能：实现残差网络
    :param block: basic block(ResNet50)或者bottleleneck(ResNet101)
    :param layers: 列表，表示残差网络的层数
    '''
    def __init__(self, block, layers, num_classes=768):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 1.conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 2.conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 3.conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 4.conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 5.conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        功能：创建残差网络的层
        步骤：它首先检查stride是否等于1或self.inplanes是否等于planes * block.expansion，如果是，则需要进行下采样。如果需要下采样，则创建一个下采样模块（downsample），该模块包含一个1x1卷积核、批量归一化层和ReLU激活函数。
        :param block: basic block或者bottleleneck
        :param planes: 输出通道数
        :param blocks: 块的数量
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # 创建block实例，并将它保存在layers列表中
        layers = []
        # 添加第一个block块，也许需要进行下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # 修改输入通道数，使其与第一个block的输出通道数相同
        self.inplanes = planes * block.expansion
        
        # 添加剩下的block块，不需要下采样
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展开为一维向量
        x = self.fc(x)

        return x

#残差网络
# resnet18
def resnet18(out_fc):
    model = ResNet(BasicBlock, [2, 2, 2, 2],out_fc)
    return model

# resnet34
def resnet34(out_fc):
    model = ResNet(BasicBlock, [3, 4, 6, 3],out_fc)
    return model

# resnet50
def resnet50(out_fc):
    model = ResNet(Bottleneck, [3, 4, 6, 3],out_fc)
    return model

# resnet101
def resnet101(out_fc):
    model = ResNet(Bottleneck, [3, 4, 23, 3],out_fc)
    return model

# resnet152
def resnet152(out_fc):
    model = ResNet(Bottleneck, [3, 8, 36, 3],out_fc)
    return model

if __name__ == '__main__':
    # 测试输入输出维度
    out_fc = 256
    img = torch.randn(32, 3, 224, 224)
    model = resnet18(out_fc)
    img = model(img)
    print(img.shape)

    outputs = torch.randn(32,128,256)
    pooled_output = torch.randn(32,256)
    print(pooled_output.shape)
    fea=torch.cat([img,pooled_output],1)
    print(fea.shape)
