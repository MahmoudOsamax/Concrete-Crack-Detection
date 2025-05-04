import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import get_norm_layer, init_net
import logging

logger = logging.getLogger(__name__)

class DeepCrackNet(nn.Module):
    def __init__(self, in_nc, num_classes, ngf, norm='batch'):
        super(DeepCrackNet, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        
        self.conv1 = nn.Sequential(*self._conv_block(in_nc, ngf, norm_layer, num_block=2))
        self.side_conv1 = nn.Conv2d(ngf, num_classes, kernel_size=1, stride=1, bias=True)

        self.conv2 = nn.Sequential(*self._conv_block(ngf, ngf*2, norm_layer, num_block=2))
        self.side_conv2 = nn.Conv2d(ngf*2, num_classes, kernel_size=1, stride=1, bias=True)

        self.conv3 = nn.Sequential(*self._conv_block(ngf*2, ngf*4, norm_layer, num_block=3))
        self.side_conv3 = nn.Conv2d(ngf*4, num_classes, kernel_size=1, stride=1, bias=True)

        self.conv4 = nn.Sequential(*self._conv_block(ngf*4, ngf*8, norm_layer, num_block=3))
        self.side_conv4 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=True)

        self.conv5 = nn.Sequential(*self._conv_block(ngf*8, ngf*8, norm_layer, num_block=3))
        self.side_conv5 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=True)

        self.fuse_conv = nn.Conv2d(num_classes*5, num_classes, kernel_size=1, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        logger.info("DeepCrackNet initialized")

    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=2, kernel_size=3, 
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias),
                     norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv

    def forward(self, x):
        try:
            h, w = x.size()[2:]
            conv1 = self.conv1(x)
            conv2 = self.conv2(self.maxpool(conv1))
            conv3 = self.conv3(self.maxpool(conv2))
            conv4 = self.conv4(self.maxpool(conv3))
            conv5 = self.conv5(self.maxpool(conv4))

            side_output1 = self.side_conv1(conv1)
            side_output2 = self.side_conv2(conv2)
            side_output3 = self.side_conv3(conv3)
            side_output4 = self.side_conv4(conv4)
            side_output5 = self.side_conv5(conv5)

            side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True)
            side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True)
            side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True)
            side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear', align_corners=True)

            fused = self.fuse_conv(torch.cat([side_output1, 
                                            side_output2, 
                                            side_output3,
                                            side_output4,
                                            side_output5], dim=1))
            return side_output1, side_output2, side_output3, side_output4, side_output5, fused
        except Exception as e:
            logger.error(f"Error in DeepCrackNet forward: {str(e)}")
            raise
def define_deepcrack(in_nc, 
                     num_classes, 
                     ngf, 
                     norm='batch',
                     init_type='xavier', 
                     init_gain=0.02, 
                     gpu_ids=[]):
    try:
        net = DeepCrackNet(in_nc, num_classes, ngf, norm)
        net = init_net(net, init_type, init_gain, gpu_ids)
        logger.info("DeepCrack network defined")
        return net
    except Exception as e:
        logger.error(f"Error in define_deepcrack: {str(e)}")
        raise

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, inputs, targets):
        try:
            BCE_loss = self.criterion(inputs, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            return F_loss.mean() if self.size_average else F_loss.sum()
        except Exception as e:
            logger.error(f"Error in BinaryFocalLoss: {str(e)}")
            raise