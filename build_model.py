import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, gpu, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.gpu = gpu
        self.conv1 = conv3x3(inplanes, planes, stride).to(self.gpu)
        self.bn1 = nn.BatchNorm2d(planes).to(self.gpu)
        self.relu = nn.ReLU(inplace=True).to(self.gpu)
        self.conv2 = conv3x3(planes, planes).to(self.gpu)
        self.bn2 = nn.BatchNorm2d(planes).to(self.gpu)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = self.downsample.to(gpu)
        self.stride = stride

    def forward(self, x):
        x = x.to(self.gpu)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SSANet_multi_gpu_model(nn.Module):
    def __init__(self, pretraind_model, sel_gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], upsmaple_rate=1):
        super(SSANet_multi_gpu_model, self).__init__()
        self.inplanes = 64
        self.upsmaple_rate = upsmaple_rate
        self.conv1 = nn.Conv2d(3, 64, 7, 1 , 3, bias=False).to(sel_gpu[0])
        self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64).to(sel_gpu[0])
        self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True).to(sel_gpu[0])

        self.conv2 = self._make_layer(BasicBlock, 64, 3,
                                      gpu=[sel_gpu[0], sel_gpu[0], sel_gpu[0]], stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4,
                                      gpu=[sel_gpu[1], sel_gpu[1], sel_gpu[1], sel_gpu[1]], stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6,
                                      gpu=[sel_gpu[1], sel_gpu[1], sel_gpu[1],
                                           sel_gpu[1], sel_gpu[2], sel_gpu[2]], stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3,
                                      gpu=[sel_gpu[2], sel_gpu[3], sel_gpu[3]], stride=2)
        self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(sel_gpu[0])
        self.sp2 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(sel_gpu[0])
        self.sp3 = nn.Sequential(
            nn.Conv2d(128, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(sel_gpu[1])
        self.sp4 = nn.Sequential(
            nn.Conv2d(256, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(sel_gpu[2])
        self.sp5 = nn.Sequential(
            nn.Conv2d(512, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(sel_gpu[3])

        self.output = self.make_infer(2, 3+5*8).to(torch.device('cuda:0'))

    def _make_layer(self, block, planes, blocks, gpu, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, gpu[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, gpu[i]))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(8, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(True)))
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):
        self.upsmaple_size = (x.size(2) // self.upsmaple_rate, x.size(3) // self.upsmaple_rate)
        self.input_size = (x.size(2), x.size(3))

        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = self.sp1(c1)
        # c1 = F.upsample(c1, size=self.upsmaple_size, mode='bilinear', align_corners=True)

        c2 = self.conv2(c1)
        sp2 = F.upsample(self.sp2(c2), size=self.input_size, mode='bilinear', align_corners=True).to(torch.device('cuda:0'))
        c2 = F.upsample(c2, size=self.upsmaple_size, mode='bilinear', align_corners=True)

        c3 = self.conv3(c2)
        sp3 = F.upsample(self.sp3(c3), size=self.input_size, mode='bilinear', align_corners=True).to(torch.device('cuda:0'))
        c3 = F.upsample(c3, size=self.upsmaple_size, mode='bilinear', align_corners=True)

        c4 = self.conv4(c3)
        sp4 = F.upsample(self.sp4(c4), size=self.input_size, mode='bilinear', align_corners=True).to(torch.device('cuda:0'))
        c4 = F.upsample(c4, size=self.upsmaple_size, mode='bilinear', align_corners=True)

        c5 = self.conv5(c4)
        sp5 = F.upsample(self.sp5(c5), size=self.input_size, mode='bilinear', align_corners=True).to(torch.device('cuda:0'))

        cat = torch.cat([x, sp1, sp2, sp3, sp4, sp5], 1)
        out = self.output(cat)

        return F.sigmoid(out)

class DRIU(nn.Module):
    def __init__(self, pretrained):
        super(DRIU, self).__init__()
        self.conv1 = nn.Sequential(*list(pretrained.features.children())[:4]).to(torch.device('cuda:0'))
        self.conv2 = nn.Sequential(*list(pretrained.features.children())[4:9]).to(torch.device('cuda:1'))
        self.conv3 = nn.Sequential(*list(pretrained.features.children())[9:16]).to(torch.device('cuda:2'))
        self.conv4 = nn.Sequential(*list(pretrained.features.children())[16:23]).to(torch.device('cuda:2'))

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp2 = nn.Sequential(
            nn.Conv2d(128, 16, 3, 1, 1),
            nn.ReLU(True)
        ).to(torch.device('cuda:1'))
        self.sp3 = nn.Sequential(
            nn.Conv2d(256, 16, 3, 1, 1),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))
        self.sp4 = nn.Sequential(
            nn.Conv2d(512, 16, 3, 1, 1),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))

        self.output = nn.Sequential(
            nn.Conv2d(16*4, 1, 1)
        ).to(torch.device('cuda:2'))

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1.to(torch.device('cuda:1')))
        c3  =self.conv3(c2.to(torch.device('cuda:2')))
        c4 = self.conv4(c3.to(torch.device('cuda:2')))

        sp1 = F.upsample(self.sp1(c1), size=(x.size(2), x.size(3)), mode='bilinear').to(torch.device('cuda:1'))
        sp2 = F.upsample(self.sp2(c2), size=(x.size(2), x.size(3)), mode='bilinear').to(torch.device('cuda:1'))
        sp3 = F.upsample(self.sp3(c3), size=(x.size(2), x.size(3)), mode='bilinear').to(torch.device('cuda:1'))
        sp4 = F.upsample(self.sp4(c4), size=(x.size(2), x.size(3)), mode='bilinear').to(torch.device('cuda:1'))

        cat = torch.cat([sp1, sp2, sp3, sp4], 1)
        out = self.output(cat.to(torch.device('cuda:2')))

        return F.sigmoid(out)

def get_model(net_name, sel_gpu, upsmaple_rate=1):    
    if net_name == 'SSA3_ResNet34':
        model = SSANet_multi_gpu_model(models.resnet34(pretrained=True), sel_gpu=sel_gpu, upsmaple_rate=upsmaple_rate)
    elif net_name == 'SSA3_VGG16':
        #TODO
        model = 0
    elif net_name == 'DRIU':
        model = DRIU(models.vgg16(pretrained=True))
    else:
        return False

    return model