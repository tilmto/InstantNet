import torch
import numpy as np
import sys
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchcontrib
import numpy as np
from pdb import set_trace as bp
from thop import profile
from operations import *
from genotypes import PRIMITIVES


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        # self.network_momentum = args.momentum
        # self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args

        self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
        
        self.flops_weight = args.flops_weight

        self.edp_weight = args.edp_weight

        print("architect initialized!")


    def step(self, input_train, target_train, input_valid, target_valid, temp=1):
        self.optimizer.zero_grad()
        if self._args.efficiency_metric == 'flops':
            loss, loss_flops = self._backward_step_flops(input_valid, target_valid, temp)
            loss += loss_flops
        elif self._args.efficiency_metric == 'energy':
            loss, loss_energy = self._backward_step_energy(input_valid, target_valid, temp)
            loss += loss_energy
        elif self._args.efficiency_metric == 'latency':
            loss, loss_latency = self._backward_step_latency(input_valid, target_valid, temp)
            loss += loss_latency
        elif self._args.efficiency_metric == 'edp':
            loss, loss_edp = self._backward_step_edp(input_valid, target_valid, temp)
            loss += loss_edp
        else:
            print('Wrong efficiency metric.')
            sys.exit()


        loss.backward()

        self.optimizer.step()

        return loss


    def _backward_step_edp(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        # flops = self.model.module.forward_flops((16, 224, 224))
        edp = self.model.module.forward_edp((3, 224, 224), temp)
            
        self.edp_supernet = edp
        loss_edp = self.edp_weight * edp

        # print(flops, loss_energy, loss)
        return loss, loss_edp


    def _backward_step_energy(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        # flops = self.model.module.forward_flops((16, 224, 224))
        energy = self.model.module.forward_energy((3, 224, 224), temp)
            
        self.energy_supernet = energy
        loss_energy = self.energy_weight * energy

        # print(flops, loss_energy, loss)
        return loss, loss_energy


    def _backward_step_latency(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        # flops = self.model.module.forward_flops((16, 224, 224))
        latency = self.model.module.forward_latency((3, 224, 224), temp)
            
        self.latency_supernet = latency
        loss_latency = self.latency_weight * latency

        # print(flops, loss_energy, loss)
        return loss, loss_latency



    def _backward_step_flops(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        # flops = self.model.module.forward_flops((16, 224, 224))
        flops = self.model.module.forward_flops((3, 224, 224), temp)
            
        self.flops_supernet = flops
        loss_flops = self.flops_weight * flops

        # print(flops, loss_flops, loss)
        return loss, loss_flops


