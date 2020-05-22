from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
import sys
import os.path as osp
from easydict import EasyDict as edict
from hw_diff_final.sw_interface import get_hw_efficiency

from fpga_nips import eval_hardware_efficiency


__all__ = ['ConvBlock', 'Skip','ConvNorm', 'OPS']

flops_lookup_table = {}
flops_file_name = "flops_lookup_table.npy"
if osp.isfile(flops_file_name):
    flops_lookup_table = np.load(flops_file_name, allow_pickle=True).item()

energy_lookup_table = {}
energy_file_name = "energy_lookup_table.npy"
if osp.isfile(energy_file_name):
    energy_lookup_table = np.load(energy_file_name, allow_pickle=True).item()

latency_lookup_table = {}
latency_file_name = "latency_lookup_table.npy"
if osp.isfile(latency_file_name):
    latency_lookup_table = np.load(latency_file_name, allow_pickle=True).item()

Conv2d = nn.Conv2d
BatchNorm2d = nn.BatchNorm2d


## Test ##
# def eval_hardware_efficiency(a, b):
#     return 1

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert (
            C % g == 0
        ), "Incompatible group size {} for input channel {}".format(g, C)
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )
        

class ConvBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Conv2d(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias)
        self.bn2 = BatchNorm2d(C_in*expansion)

        self.conv3 = Conv2d(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn3 = BatchNorm2d(C_out)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        # x = self.relu(x)

        return x

    @staticmethod
    def _flops(h, w, C_in, C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer_id = 1
        layer = ConvBlock(C_in, C_out, layer_id, expansion, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    
    # @staticmethod
    # def _energy(layer_id, h, w, C_in, C_out, expansion, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, name=None):
    #     # layer = Conv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
    #     # energy = compute_energy(layer, (1, C_in, h, w))

    #     h_out = h // stride
    #     w_out = w // stride

    #     conv_info_list = [
    #                  [[1, {'ch_out':[C_in*expansion,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
    #                  'row_out':[w,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]],

    #                  [[stride, {'ch_out':[C_in*expansion,0],'ch_in':[C_in*expansion,0],'batch':[1,0],'col_out':[h_out,0],
    #                  'row_out':[w_out,0],'row_kernel':[kernel_size,0],'col_kernel':[kernel_size,0]}]],

    #                  [[1, {'ch_out':[C_out,0],'ch_in':[C_in*expansion,0],'batch':[1,0],'col_out':[h_out,0],
    #                  'row_out':[w_out,0],'row_kernel':[1, 0],'col_kernel':[1 ,0]}]]
    #                 ]

    #     energy = 0

    #     for i, conv_info in enumerate(conv_info_list):

    #         if i == 1:
    #             layer_group = C_in*expansion
    #         else:
    #             layer_group = groups

    #         energy_part = get_hw_efficiency(conv_info, name, group=layer_group)

    #         energy += energy_part

    #     return energy


    def forward_energy(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)

        if name in energy_lookup_table:
            energy = energy_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
            # energy = ConvBlock._energy(self.layer_id, h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, name=name)
            # energy_lookup_table[name] = energy
            # np.save(energy_file_name, energy_lookup_table)

        return energy, (c_out, h_out, w_out)


    def forward_latency(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)

        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()

        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        if self.groups == 2:
            group_type = 2
            group = 2
        else:
            group_type = 0
            group = 1

        block_info = [
                     [1, {'ch_out':[self.C_in*self.expansion/group,0],'ch_in':[self.C_in/group,0],'batch':[1,0],'col_out':[h_in,0],
                     'row_out':[w_in,0],'row_kernel':[1, 0],'col_kernel':[1,0]}, group_type, group],

                     [self.stride, {'ch_out':[self.C_in*self.expansion,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[self.kernel_size,0],'col_kernel':[self.kernel_size,0]}, 1, 1],

                     [1, {'ch_out':[self.C_out/group,0],'ch_in':[self.C_in*self.expansion/group,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[1, 0],'col_kernel':[1 ,0]}, group_type, group]
                     ]

        return block_info, (self.C_out, h_out, w_out)


    def update_penalty(self, size, opt_hw_list):
        c_in, h_in, w_in = size

        block_info, size = self.layer_info(size)

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, self.C_in, self.C_out, self.expansion, 
                                                                                self.kernel_size, self.stride, self.groups)
        if name not in energy_lookup_table:
            energy_list = []
            latency_list = []

            for opt_hw in opt_hw_list:
                energy, latency = eval_hardware_efficiency(block_info, opt_hw)
                energy_list.append(energy)
                latency_list.append(latency)

            block_energy = sum(energy_list)/len(energy_list)
            block_latency = sum(latency_list)/len(latency_list)

            print(name, 'energy:', block_energy, 'latency:', block_latency)

            energy_lookup_table[name] = block_energy
            latency_lookup_table[name] = block_latency

            np.save(energy_file_name, energy_lookup_table)
            np.save(latency_file_name, latency_lookup_table)

        return size



class Skip(nn.Module):
    def __init__(self, C_in, C_out, layer_id, stride=1):
        super(Skip, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d'%C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

        self.layer_id = layer_id

        self.kernel_size = 1
        self.padding = 0

        if stride == 2 or C_in != C_out:
            self.conv = Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = Skip(C_in, C_out, stride)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    # @staticmethod
    # def _energy(layer_id, h, w, C_in, C_out, stride=1, name=None):
    #     # layer = Conv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
    #     # energy = compute_energy(layer, (1, C_in, h, w))

    #     conv_info = [[stride, {'ch_out':[C_out,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],'row_out':[w,0],'row_kernel':[1,0],'col_kernel':[1,0]}]]
    #     energy = get_hw_efficiency(conv_info, name)

    #     return energy

    def forward_energy(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in energy_lookup_table:
            energy = energy_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
            # if self.stride == 2 or self.C_in != self.C_out:
            #     energy = Skip._energy(self.layer_id, h_out, w_out, c_in, c_out, self.stride, name=name)
            # else:
            #     energy = 0
            # energy_lookup_table[name] = energy
            # np.save(energy_file_name, energy_lookup_table)

        return energy, (c_out, h_out, w_out)


    def forward_latency(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
            # if self.stride == 2 or self.C_in != self.C_out:
            #     energy = Skip._energy(self.layer_id, h_out, w_out, c_in, c_out, self.stride, name=name)
            # else:
            #     energy = 0
            # energy_lookup_table[name] = energy
            # np.save(energy_file_name, energy_lookup_table)

        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Skip._flops(h_in, w_in, c_in, c_out, self.stride)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def forward(self, x):
        if hasattr(self, 'conv'):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x

        return out


    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        if self.stride == 2 or self.C_in != self.C_out:
            conv_info = [
                        [self.stride, {'ch_out':[self.C_out,0],'ch_in':[self.C_in,0],'batch':[1,0],'col_out':[h_out,0],'row_out':[w_out,0],
                        'row_kernel':[1,0],'col_kernel':[1,0]}, 0, 1]
                        ]
        else:
            conv_info = []

        return conv_info, (self.C_out, h_out, w_out)


    def update_penalty(self, size, opt_hw_list):
        c_in, h_in, w_in = size

        block_info, size = self.layer_info(size)

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, self.C_in, self.C_out, self.stride)

        if name not in energy_lookup_table:
            if self.stride == 2 or self.C_in != self.C_out:
                energy_list = []
                latency_list = []

                for opt_hw in opt_hw_list:
                    energy, latency = eval_hardware_efficiency(block_info, opt_hw)
                    energy_list.append(energy)
                    latency_list.append(latency)

                block_energy = sum(energy_list)/len(energy_list)
                block_latency = sum(latency_list)/len(latency_list)
            else:
                block_energy = 0
                block_latency = 0

            print(name, 'energy:', block_energy, 'latency:', block_latency)

            energy_lookup_table[name] = block_energy
            latency_lookup_table[name] = block_latency

            np.save(energy_file_name, energy_lookup_table)
            np.save(latency_file_name, latency_lookup_table)

        return size



class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        self.conv = Conv2d(C_in, C_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups, bias=bias)
        self.bn = BatchNorm2d(C_out)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))

        return x

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    
    # @staticmethod
    # def _energy(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, name=None):
    #     # layer = Conv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
    #     # energy = compute_energy(layer, (1, C_in, h, w))

    #     h_out = h // stride
    #     w_out = w // stride

    #     conv_info = [[1, {'ch_out':[C_out,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
    #                     'row_out':[w,0],'row_kernel':[kernel_size, 0],'col_kernel':[kernel_size,0]}]]

    #     energy = get_hw_efficiency(conv_info, name)
    #     return energy


    def forward_energy(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)

        if name in energy_lookup_table:
            energy = energy_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
            # energy = ConvNorm._energy(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, name=name)
            # energy_lookup_table[name] = energy
            # np.save(energy_file_name, energy_lookup_table)

        return energy, (c_out, h_out, w_out)


    def forward_latency(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)

        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
            # energy = ConvNorm._energy(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, name=name)
            # energy_lookup_table[name] = energy
            # np.save(energy_file_name, energy_lookup_table)

        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        conv_info = [
                    [self.stride, {'ch_out':[self.C_out,0],'ch_in':[self.C_in,0],'batch':[1,0],'col_out':[h_out,0],
                    'row_out':[w_out,0],'row_kernel':[self.kernel_size, 0],'col_kernel':[self.kernel_size,0]}, 0, 1]
                    ]
            
        return conv_info, (self.C_out, h_out, w_out)


    def update_penalty(self, size, opt_hw_list):
        c_in, h_in, w_in = size

        block_info, size = self.layer_info(size)

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, self.C_in, self.C_out, self.kernel_size, self.stride, self.groups)

        if name not in energy_lookup_table:
            energy_list = []
            latency_list = []

            for opt_hw in opt_hw_list:
                energy, latency = eval_hardware_efficiency(block_info, opt_hw)
                energy_list.append(energy)
                latency_list.append(latency)

            block_energy = sum(energy_list)/len(energy_list)
            block_latency = sum(latency_list)/len(latency_list)

            print(name, 'energy:', block_energy, 'latency:', block_latency)

            energy_lookup_table[name] = block_energy
            latency_lookup_table[name] = block_latency

            np.save(energy_file_name, energy_lookup_table)
            np.save(latency_file_name, latency_lookup_table)

        return size





OPS = {
    'k3_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
    'k3_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
    'k3_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
    'k3_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
    'k5_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
    'k5_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
    'k5_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
    'k5_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
    'skip' : lambda C_in, C_out, layer_id, stride: Skip(C_in, C_out, layer_id, stride)
}

