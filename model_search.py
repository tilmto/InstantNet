import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
import operations
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile

from fpga_nips import search_opt_hw_rs as search_opt_hw
# from fpga_nips import search_opt_hw_diff as search_opt_hw

## Test ##
# def search_opt_hw(sampled_arch_list):
#     return np.ones(len(sampled_arch_list))


# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits, dim=-1), temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, layer_id, stride=1):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.layer_id = layer_id

        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, layer_id, stride)
            self._ops.append(op)


    def forward(self, x, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            result = result + op(x) * w 
            # print(type(op), result.shape)
        return result


    def forward_energy(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            energy, size_out = op.forward_energy(size)
            result = result + energy * w
        return result, size_out


    def forward_latency(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w
        return result, size_out


    def forward_flops(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            flops, size_out = op.forward_flops(size)
            result = result + flops * w

        return result, size_out


    def derive_layer(self, size, alpha):
        choice = alpha.argmax()

        block_info, size = self._ops[choice].layer_info(size)

        return block_info, size


    def sample_layer(self, size, alpha):
        prob = alpha.cpu().detach().numpy()
        prob /= prob.sum()

        choice = np.random.choice(range(len(self._ops)), p=prob)

        block_info, size = self._ops[choice].layer_info(size)

        return block_info, size


    def update_penalty(self, size, opt_hw_list):
        for op in self._ops:
            size_out = op.update_penalty(size, opt_hw_list)

        return size_out


class FBNet(nn.Module):
    def __init__(self, config):
        super(FBNet, self).__init__()

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
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id])
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id])
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], layer_id, stride=1)
                
                layer_id += 1
                self.cells.append(op)


        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)

        self._arch_params = self._build_arch_parameters()
        self._reset_arch_parameters()

        self._criterion = nn.CrossEntropyLoss().cuda()

        self.sample_func = config.sample_func


    def forward(self, input, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)
    
        out = self.stem(input)

        for i, cell in enumerate(self.cells):
            out = cell(out, alpha[i])

        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))

        return out
        ###################################


    def forward_flops(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size, alpha[i])
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def forward_edp(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)

        size_orig = size

        energy_total = []

        energy, size = self.stem.forward_energy(size)
        energy_total.append(energy)

        for i, cell in enumerate(self.cells):
            energy, size = cell.forward_energy(size, alpha[i])
            energy_total.append(energy)

        energy, size = self.header.forward_energy(size)
        energy_total.append(energy)

        size = size_orig

        latency_total = []

        latency, size = self.stem.forward_latency(size)
        latency_total.append(latency)

        for i, cell in enumerate(self.cells):
            latency, size = cell.forward_latency(size, alpha[i])
            latency_total.append(latency)

        latency, size = self.header.forward_latency(size)
        latency_total.append(latency)

        edp = sum(energy_total) * sum(latency_total)

        return edp


    def forward_energy(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)

        energy_total = []

        energy, size = self.stem.forward_energy(size)
        energy_total.append(energy)

        for i, cell in enumerate(self.cells):
            energy, size = cell.forward_energy(size, alpha[i])
            energy_total.append(energy)

        energy, size = self.header.forward_energy(size)
        energy_total.append(energy)

        return sum(energy_total)


    def forward_latency(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)

        latency_total = []

        latency, size = self.stem.forward_latency(size)
        latency_total.append(latency)

        for i, cell in enumerate(self.cells):
            latency, size = cell.forward_latency(size, alpha[i])
            latency_total.append(latency)

        latency, size = self.header.forward_latency(size)
        latency_total.append(latency)

        return sum(latency_total)


    def _loss(self, input, target, temp=1):

        logit = self(input, temp)
        loss = self._criterion(logit, target)

        return loss


    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)))

        return {"alpha": self.alpha}


    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)

        getattr(self, "alpha").data = Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)


    def clip(self):
        for line in getattr(self, "alpha"):
            max_index = line.argmax()
            line.data.clamp_(0, 1)
            if line.sum() == 0.0:
                line.data[max_index] = 1.0
            line.data.div_(line.sum())


    def sample_arch(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)

        sampled_arch_info = []

        block_info, size = self.stem.layer_info(size)
        sampled_arch_info.extend(block_info)

        for i, cell in enumerate(self.cells):
            block_info, size = cell.sample_layer(size, alpha[i])
            sampled_arch_info.extend(block_info)

        block_info, size = self.header.layer_info(size)
        sampled_arch_info.extend(block_info)

        return sampled_arch_info


    def update_hw(self, size, num_sample, temp=1):
        sampled_arch_list = []

        for i in range(num_sample):
            sampled_arch_info = self.sample_arch(size, temp=temp)
            sampled_arch_list.append(sampled_arch_info)

        self.opt_hw_list, _ = search_opt_hw(sampled_arch_list)

        return self.opt_hw_list


    def update_penalty(self, size, opt_hw_list):
        operations.energy_lookup_table = {}

        size = self.stem.update_penalty(size, opt_hw_list)

        for i, cell in enumerate(self.cells):
            size = cell.update_penalty(size, opt_hw_list)

        size = self.header.update_penalty(size, opt_hw_list)


    def derive_arch(self, size):
        alpha = F.softmax(getattr(self, "alpha"), dim=-1)

        sampled_arch_info = []

        block_info, size = self.stem.layer_info(size)
        sampled_arch_info.extend(block_info)

        for i, cell in enumerate(self.cells):
            block_info, size = cell.derive_layer(size, alpha[i])
            sampled_arch_info.extend(block_info)

        block_info, size = self.header.layer_info(size)
        sampled_arch_info.extend(block_info)

        return sampled_arch_info


    def eval_edp(self, size):
        derived_arch = self.derive_arch(size)

        opt_hw_list, edp_list = search_opt_hw([derived_arch], epoch=500)

        self.opt_hw = opt_hw_list[0]
        self.edp = edp_list[0]

        return self.edp



if __name__ == '__main__':
    model = FBNet(num_classes=10)
    print(model)