import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile

from fpga_nips import search_opt_hw_rs as search_opt_hw
# from fpga_nips import search_opt_hw_diff as search_opt_hw

class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, layer_id, stride=1):
        super(MixedOp, self).__init__()
        self.layer_id = layer_id
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, layer_id, stride)

    def forward(self, x):
        return self._op(x)

    def forward_energy(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        energy, size_out = self._op.forward_energy(size)
        return energy, size_out

    def forward_latency(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        latency, size_out = self._op.forward_latency(size)
        return latency, size_out

    def forward_flops(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        flops, size_out = self._op.forward_flops(size)

        return flops, size_out

    def layer_info(self, size):
        block_info, size = self._op.layer_info(size)

        return block_info, size


class FBNet_Infer(nn.Module):
    def __init__(self, alpha, config):
        super(FBNet_Infer, self).__init__()

        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)

        self.num_classes = config.num_classes

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel

        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=2, padding=1, bias=False)

        self.cells = nn.ModuleList()

        layer_id = 1

        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], op_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id])
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], op_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id])
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], op_idx_list[layer_id-1], layer_id, stride=1)
                
                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)

        self._criterion = nn.CrossEntropyLoss()


    def forward(self, input):

        out = self.stem(input)

        for i, cell in enumerate(self.cells):
            out = cell(out)

        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))

        return out
    
    def forward_flops(self, size):

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def forward_edp(self, size):
        energy = self.forward_energy(size)
        latency = self.forward_latency(size)

        return energy * latency


    def forward_energy(self, size):

        energy_total = []

        energy, size = self.stem.forward_energy(size)
        energy_total.append(energy)

        for i, cell in enumerate(self.cells):
            energy, size = cell.forward_energy(size)
            energy_total.append(energy)

        energy, size = self.header.forward_energy(size)
        energy_total.append(energy)

        return sum(energy_total)


    def forward_latency(self, size):

        latency_total = []

        latency, size = self.stem.forward_energy(size)
        latency_total.append(latency)

        for i, cell in enumerate(self.cells):
            latency, size = cell.forward_energy(size)
            latency_total.append(latency)

        latency, size = self.header.forward_energy(size)
        latency_total.append(latency)

        return sum(latency_total)


    def _loss(self, input, target):
        logit = self(input)
        loss = self._criterion(logit, target)

        return loss


    def layer_info(self, size):
        arch_info = []

        block_info, size = self.stem.layer_info(size)
        arch_info.extend(block_info)

        for i, cell in enumerate(self.cells):
            block_info, size = cell.layer_info(size)
            arch_info.extend(block_info)

        block_info, size = self.header.layer_info(size)
        arch_info.extend(block_info)

        return arch_info


    def eval_edp(self, size, epoch=500):
        arch_info = self.layer_info(size)

        opt_hw_list, edp_list = search_opt_hw([arch_info], epoch=epoch)

        self.opt_hw = opt_hw_list[0]
        self.edp = edp_list[0]

        return self.edp