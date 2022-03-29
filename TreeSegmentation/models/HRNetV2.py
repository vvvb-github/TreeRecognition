import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .blocks import BN_MOMENTUM, BasicBlock, BottleNeck, block_table
from utils import log_error, log_info, log_success, log_warn


class HRModule(nn.Module):
    def __init__(self, num_branch, block, num_blocks, in_channels, out_channels, fuse_method='SUM', multi_scale_output=True) -> None:
        super(HRModule, self).__init__()
        self._check_branches(num_branch, num_blocks, in_channels, out_channels)
        self._fuse_method = fuse_method
        self._num_branch = num_branch
        self._in_channels = in_channels

        self._relu = nn.ReLU(inplace=True)
        self._branches = self._make_branches(
            num_branch, block, num_blocks, out_channels)
        self._fuse_layers = self._make_fuse_layers(
            num_branch, multi_scale_output)

    def forward(self, x):
        if self._num_branch == 1:
            return [self._branches[0](x[0])]

        for i in range(self._num_branch):
            x[i] = self._branches[i](x[i])

        x_fuse = []
        for i in range(len(self._fuse_layers)):
            y = x[0] if i == 0 else self._fuse_layers[i][0](x[0])
            for j in range(1, self._num_branch):
                if i == j:
                    y = y+x[j]
                elif j > i:
                    width = x[i].shape[-1]
                    height = x[i].shape[-2]
                    y = y + \
                        F.interpolate(self._fuse_layers[i][j](x[j]), size=[
                                      height, width], mode='bilinear', align_corners=True)
                else:
                    y = y+self._fuse_layers[i][j](x[j])
            x_fuse.append(self._relu(y))

        return x_fuse

    def get_in_channels(self):
        return self._in_channels

    def _check_branches(self, num_branch, num_blocks, in_channels, out_channels):
        if num_branch != len(num_blocks):
            log_error('HRModule num_branches{} is not equal to length of num_blocks {}.'.format(
                num_branch, len(num_blocks)))
            raise

        if num_branch != len(in_channels):
            log_error('HRModule num_branches{} is not equal to length of in_channels {}.'.format(
                num_branch, len(in_channels)))
            raise

        if num_branch != len(out_channels):
            log_error('HRModule num_branches{} is not equal to length of out_channels {}.'.format(
                num_branch, len(out_channels)))
            raise

    def _make_branches(self, num_branch, block, num_blocks, out_channels):
        branches = []
        for i in range(num_branch):
            branches.append(self._make_one_branch(
                i, block, num_blocks[i], out_channels[i]))

        return nn.ModuleList(branches)

    def _make_one_branch(self, index, block, num_block, out_channel):
        downsample = None
        if self._in_channels[index] != out_channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels[index], out_channel*block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel*block.expansion,
                               momentum=BN_MOMENTUM)
            )

        layers = []
        layers.append(
            block(self._in_channels[index], out_channel, 1, downsample))
        self._in_channels[index] = out_channel*block.expansion
        for i in range(1, num_block):
            layers.append(block(self._in_channels[index], out_channel))

        return nn.Sequential(*layers)

    def _make_fuse_layers(self, num_branch, multi_scale_output):
        if num_branch == 1:
            return None

        fuse_layers = []
        for i in range(num_branch if multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branch):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(
                            self._in_channels[j], self._in_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(
                            self._in_channels[i], momentum=BN_MOMENTUM)
                    ))
                elif j < i:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i-j-1:
                            out_channel = self._in_channels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(
                                    self._in_channels[j], out_channel, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(
                                    out_channel, momentum=BN_MOMENTUM)
                            ))
                        else:
                            out_channel = self._in_channels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(
                                    self._in_channels[j], out_channel, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(
                                    out_channel, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
                else:
                    fuse_layer.append(None)
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)


class HRNetV2(nn.Module):
    def __init__(self, config) -> None:
        super(HRNetV2, self).__init__()

        # stem net
        stem_channel = config['stem_channel']
        self._conv1 = nn.Conv2d(
            3, stem_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn1 = nn.BatchNorm2d(stem_channel, momentum=BN_MOMENTUM)
        self._conv2 = nn.Conv2d(
            stem_channel, stem_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(stem_channel, momentum=BN_MOMENTUM)
        self._relu = nn.ReLU(inplace=True)

        # make each stage
        stages = config['stages']
        self._num_branchs = []

        stage1 = stages[0]
        self._num_branchs.append(stage1['num_branch'])
        channels = stage1['channels']
        block = block_table[stage1['block']]
        num_blocks = stage1['num_blocks']
        self._layer1 = self._make_layer(
            block, stem_channel, channels[0], num_blocks[0])
        stage1_out_channel = channels[0]*block.expansion

        stage2 = stages[1]
        self._num_branchs.append(stage2['num_branch'])
        channels = stage2['channels']
        block = block_table[stage2['block']]
        channels = [channels[i]*block.expansion for i in range(len(channels))]
        self._transition1 = self._make_transition_layer(
            [stage1_out_channel], channels)
        self._stage2, channels_pre = self._make_stage(stage2, channels)

        stage3 = stages[2]
        self._num_branchs.append(stage3['num_branch'])
        channels = stage3['channels']
        block = block_table[stage3['block']]
        channels = [channels[i]*block.expansion for i in range(len(channels))]
        self._transition2 = self._make_transition_layer(channels_pre, channels)
        self._stage3, channels_pre = self._make_stage(stage3, channels)

        stage4 = stages[3]
        self._num_branchs.append(stage4['num_branch'])
        channels = stage4['channels']
        block = block_table[stage4['block']]
        channels = [channels[i]*block.expansion for i in range(len(channels))]
        self._transition3 = self._make_transition_layer(channels_pre, channels)
        self._stage4, channels_pre = self._make_stage(
            stage4, channels, multi_scale_output=True)

        last_channel = np.int(np.sum(channels_pre))

        # full connect
        self._last_layer = nn.Sequential(
            nn.Conv2d(last_channel, last_channel,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_channel, config['out_channel'], kernel_size=config['final_conv_size'],
                      stride=1, padding=1 if config['final_conv_size'] == 3 else 0)
        )

    def forward(self, x):
        x = self._conv1(x)
        x = self._bn1(x)
        x = self._relu(x)
        x = self._conv2(x)
        x = self._bn2(x)
        x = self._relu(x)
        x = self._layer1(x)

        x_list = []
        for i in range(self._num_branchs[1]):
            if self._transition1[i] is not None:
                x_list.append(self._transition1[i](x))
            else:
                x_list.append(x)
        y_list = self._stage2(x_list)

        x_list = []
        for i in range(self._num_branchs[2]):
            if self._transition2[i] is not None:
                if i < self._num_branchs[1]:
                    x_list.append(self._transition2[i](y_list[i]))
                else:
                    x_list.append(self._transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self._stage3(x_list)

        x_list = []
        for i in range(self._num_branchs[3]):
            if self._transition3[i] is not None:
                if i < self._num_branchs[2]:
                    x_list.append(self._transition3[i](y_list[i]))
                else:
                    x_list.append(self._transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self._stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=True)
        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self._last_layer(x)

        return x

    def _make_layer(self, block, in_channel, out_channel, num_block):
        downsample = None
        if in_channel != out_channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel*block.expansion,
                               momentum=BN_MOMENTUM)
            )
        layers = []
        layers.append(block(in_channel, out_channel, 1, downsample))
        in_channel = out_channel*block.expansion
        for i in range(1, num_block):
            layers.append(block(in_channel, out_channel))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, channels_pre, channels_cur):
        num_branch_pre = len(channels_pre)
        num_branch_cur = len(channels_cur)

        transition_layers = []
        for i in range(num_branch_cur):
            if i < num_branch_pre:
                if channels_cur[i] != channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(
                            channels_pre[i], channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(channels_cur[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branch_pre):
                    in_channel = channels_pre[-1]
                    out_channel = channels_cur[i] if j == i - \
                        num_branch_pre else in_channel
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_channel, out_channel,
                                  3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, stage_config, in_channels, multi_scale_output=True):
        num_module = stage_config['num_module']
        num_branch = stage_config['num_branch']
        num_blocks = stage_config['num_blocks']
        channels = stage_config['channels']
        block = block_table[stage_config['block']]
        fuse_method = stage_config['fuse_method']

        modules = []
        for i in range(num_module):
            if not multi_scale_output and i == num_module-1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HRModule(num_branch, block, num_blocks, in_channels,
                         channels, fuse_method, reset_multi_scale_output)
            )
            in_channels = modules[-1].get_in_channels()

        return nn.Sequential(*modules), in_channels
