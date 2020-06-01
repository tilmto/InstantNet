from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

import torchvision

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tensorboardX import SummaryWriter

from datasets import prepare_train_data, prepare_test_data

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_train import config

from model_search import FBNet as Network
from model_infer import FBNet_Infer

from thop import profile
from thop.count_hooks import count_convNd

import genotypes

import operations
from quantize import QConv2d

from calibrate_bn import bn_update

import argparse

parser = argparse.ArgumentParser(description='Search on ImageNet-100')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to ImageNet-100')
parser.add_argument('-b', '--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers per gpu')
parser.add_argument('--world_size', type=int, default=None,
                    help='number of nodes')
parser.add_argument('--rank', type=int, default=None,
                    help='node rank')
parser.add_argument('--dist_url', type=str, default=None,
                    help='url used to set up distributed training')
args = parser.parse_args()

best_acc = 0
best_epoch = 0

operations.DWS_CHWISE_QUANT = config.dws_chwise_quant

custom_ops = {QConv2d: count_convNd}


def main():
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.world_size is not None:
        config.world_size = args.world_size
    if args.world_size is not None:
        config.rank = args.rank
    if args.dist_url is not None:
        config.dist_url = args.dist_url

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    config.ngpus_per_node = ngpus_per_node
    config.num_workers = config.num_workers * ngpus_per_node

    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    global best_acc
    global best_epoch

    config.gpu = gpu

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))


    if config.distributed:
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        print("Rank: {}".format(config.rank))


    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
        config.save = 'ckpt/{}'.format(config.save)
        logger = SummaryWriter(config.save)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
    else:
        logger = None


    state = torch.load(os.path.join(config.load_path, 'arch.pt'))
    alpha = state['alpha']

    # alpha = torch.zeros(sum(config.num_layer_list), len(genotypes.PRIMITIVES))
    # alpha[:,0] = 10

    model = FBNet_Infer(alpha, config=config)

    print('config.gpu:', config.gpu)
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        print('No distributed.')
        sys.exit()

    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
        flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),), custom_ops=custom_ops)
        logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)

    # if type(config.pretrain) == str:
    #     state_dict = torch.load(config.pretrain)

    #     for key in state_dict.copy():
    #         if 'bn.0' in key:
    #             new_key_list = []

    #             for i in range(1, len(config.num_bits_list)):
    #                 new_key = []
    #                 new_key.extend(key.split('.')[:-2])
    #                 new_key.append(str(i))
    #                 new_key.append(key.split('.')[-1])
    #                 new_key = '.'.join(new_key)

    #                 state_dict[new_key] = state_dict[key]

    #     model.load_state_dict(state_dict, strict=False)

    if type(config.pretrain) == str:
        state_dict = torch.load(config.pretrain)
        model.load_state_dict(state_dict)


    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ##############################
    total_iteration = config.nepochs * config.niters_per_epoch
    
    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        print("Wrong Learning Rate Schedule Type.")
        sys.exit()


    # data loader ############################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


    train_dataset = prepare_train_data(dataset=config.dataset,
                                      datadir=config.dataset_path+'/train')
    test_dataset = prepare_test_data(dataset=config.dataset,
                                    datadir=config.dataset_path+'/val')

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, num_workers=config.num_workers, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=config.num_workers)

    if config.eval_only:
        logging.info('Eval - Acc under different bits: ' + str(infer(0, model, train_loader, test_loader, logger, config.num_bits_list, update_bn=config.update_bn, show_distrib=config.show_distrib)))
        sys.exit(0)

    # tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in range(config.nepochs):

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            # tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
            logging.info("[Epoch %d/%d] lr=%f" % (epoch + 1, config.nepochs, optimizer.param_groups[0]['lr']))

        if config.distributed:
            train_sampler.set_epoch(epoch)

        if config.num_bits_list_schedule:
            num_bits_list = update_num_bits_list(config.num_bits_list, config.num_bits_list_schedule, config.schedule_freq, epoch)
        else:
            num_bits_list = config.num_bits_list

        train(train_loader, model, optimizer, lr_policy, logger, epoch, num_bits_list=num_bits_list, bit_schedule=config.bit_schedule, 
             loss_scale=config.loss_scale, distill_weight=config.distill_weight, cascad=config.cascad, update_bn_freq=config.update_bn_freq, config=config)

        torch.cuda.empty_cache()
        lr_policy.step()

        if epoch < 250:
            eval_epoch = 10
        else:
            eval_epoch = config.eval_epoch

        # validation
        if (epoch+1) % eval_epoch == 0:
            # tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            with torch.no_grad():
                acc_bits_part = infer(epoch, model, train_loader, test_loader, logger, config.num_bits_list, update_bn=False, show_distrib=False)
                acc_bits = []
                for acc in acc_bits_part:
                    acc_new = reduce_tensor(acc, config.world_size)
                    acc_bits.append(acc_new)

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                for i, num_bits in enumerate(config.num_bits_list):
                    logger.add_scalar('acc/val_bits_%d' % num_bits, acc_bits[i], epoch)
                    
                logging.info("Epoch: " + str(epoch) +" Acc under different bits: " + str(acc_bits))
                
                logger.add_scalar('flops/val', flops, epoch)
                logging.info("Epoch %d: flops %.3f"%(epoch, flops))

                save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
        save(model, os.path.join(config.save, 'weights.pt'))

    acc_bits_part = infer(0, model, train_loader, test_loader, logger, config.num_bits_list, update_bn=config.update_bn, show_distrib=False)
    acc_bits = []
    for acc in acc_bits_part:
        acc_new = reduce_tensor(acc, config.world_size)
        acc_bits.append(acc_new)

    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0): 
        logging.info('Final Eval - Acc under different bits: ' + str(acc_bits))



def train(train_loader, model, optimizer, lr_policy, logger, epoch, num_bits_list, bit_schedule, loss_scale, distill_weight, cascad, update_bn_freq, config):
    model.train()

    # bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    # pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)

    if len(num_bits_list) == 1:
        bit_schedule = 'high2low'

    for step, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss_value = [-1 for _ in num_bits_list]

        if bit_schedule == 'avg_loss':
            if distill_weight > 0:
                if cascad:
                    teacher_list = []
                    for num_bits in num_bits_list[::-1]:
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target)

                        if len(teacher_list) > 0:
                            for logit_teacher in teacher_list:
                                loss += distill_weight * nn.MSELoss()(logit, logit_teacher)
                                teacher_list.append(logit.detach())

                        loss = loss * loss_scale[num_bits_list.index(num_bits)]

                        loss.backward()

                        loss_value[num_bits_list.index(num_bits)] = loss.item()

                        del logit
                        del loss

                else:
                    logit = model(input, num_bits_list[-1])
                    loss = model.module._criterion(logit, target)
                    loss = loss * loss_scale[-1]
                    loss.backward()
                    loss_value[-1] = loss.item()

                    logit_teacher = logit.detach()

                    del logit
                    del loss

                    for num_bits in num_bits_list[:-1]:
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target) + distill_weight * nn.MSELoss()(logit, logit_teacher)

                        loss = loss * loss_scale[num_bits_list.index(num_bits)]

                        loss.backward()

                        loss_value[num_bits_list.index(num_bits)] = loss.item()

                        del logit
                        del loss

            else:
                for num_bits in num_bits_list:
                    logit = model(input, num_bits)
                    loss = model.module._criterion(logit, target)

                    loss = loss * loss_scale[num_bits_list.index(num_bits)]

                    loss.backward()

                    loss_value[num_bits_list.index(num_bits)] = loss.item()

                    del logit
                    del loss

            # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        elif bit_schedule == 'max_loss':
            if distill_weight > 0:
                if cascad:
                    loss_list = []
                    teacher_list = []

                    for i, num_bits in enumerate(num_bits_list):
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target)

                        loss_list.append(loss.item())
                        teacher_list.append(logit.detach())

                        del logit
                        del loss

                    num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                    logit = model(input, num_bits_max)
                    loss = model.module._criterion(logit, target)

                    for logit_teacher in teacher_list[num_bits_list.index(num_bits_max)+1:]:
                        loss += distill_weight * nn.MSELoss()(logit, logit_teacher)

                    loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                    loss.backward()

                else:
                    loss_list = []

                    for i, num_bits in enumerate(num_bits_list[:-1]):
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target)

                        loss_list.append(loss.item())

                        del logit
                        del loss

                    logit = model(input, num_bits_list[-1])
                    loss = model.module._criterion(logit, target)
                    loss_list.append(loss.item())

                    logit_teacher = logit.detach()

                    del logit
                    del loss

                    num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                    logit = model(input, num_bits_max)

                    if num_bits_max == num_bits_list[-1]:
                        loss = model.module._criterion(logit, target)
                    else:
                        loss = model.module._criterion(logit, target) + distill_weight * nn.MSELoss()(logit, logit_teacher)

                    loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                    loss.backward()

            else:
                loss_list = []

                for i, num_bits in enumerate(num_bits_list):
                    logit = model(input, num_bits)
                    loss = model.module._criterion(logit, target)

                    loss_list.append(loss.item())

                    del logit
                    del loss

                num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                logit = model(input, num_bits_max)
                loss = model.module._criterion(logit, target)

                loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                loss.backward()
            
            # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # loss_value[num_bits_list.index(num_bits_max)] = loss.item()
            loss_value = loss_list

        else:
            print('Wrong Bit Schedule.')
            sys.exit()

        if step % 10 == 0:
            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                for i, num_bits in enumerate(num_bits_list):
                    if loss_value[i] != -1:
                        logger.add_scalar('loss/num_bits_%d' % num_bits, loss_value[i], epoch*len(train_loader)+step)
                        logging.info("[Epoch %d/%d][Step %d/%d] Num_Bits=%d Loss=%.3f" % (epoch + 1, config.nepochs, step + 1, len(train_loader), num_bits, loss_value[i]))

        # pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader)))

    torch.cuda.empty_cache()



def infer(epoch, model, train_loader, test_loader, logger, num_bits_list, update_bn=False, show_distrib=False):
    model.eval()
    acc_bits = []

    if show_distrib:
        self_entropy_list = [[] for _ in range(len(num_bits_list))]

    for num_bits in num_bits_list:
        prec1_list = []

        if update_bn:
            bn_update(train_loader, model, num_bits=num_bits)
            save(model, os.path.join(config.save, 'weights_ftbn.pt'))
            model.eval()

        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input, volatile=True).cuda()
            target_var = Variable(target, volatile=True).cuda()

            output = model(input_var, num_bits)

            if show_distrib:
                output_softmax = torch.nn.functional.softmax(output, dim=-1)
                self_entropy = -torch.mean(torch.sum(output_softmax * torch.log(output_softmax), dim=-1)).item()
                self_entropy_list[num_bits_list.index(num_bits)].append(self_entropy)

            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list)/len(prec1_list)
        acc_bits.append(acc)

    if show_distrib:
        for i in range(len(self_entropy_list)):
            self_entropy_list[i]  = sum(self_entropy_list[i])/len(self_entropy_list[i])
        print('Self-Entropy under different bits: ' + str(self_entropy_list))

    return acc_bits


def update_num_bits_list(num_bits_list_orig, num_bits_list_schedule, schedule_freq, epoch):
    assert num_bits_list_schedule in ['low2high', 'high2low']

    if num_bits_list_schedule == 'low2high':
        num_bits_list = num_bits_list_orig[:int(epoch // schedule_freq + 1)]

    elif num_bits_list_schedule == 'high2low':
        num_bits_list = num_bits_list_orig[-int(epoch // schedule_freq + 1):]

    return num_bits_list


def reduce_tensor(rt, n):
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
