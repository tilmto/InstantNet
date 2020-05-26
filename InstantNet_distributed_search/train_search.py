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

from datasets import prepare_train_data, prepare_test_data

import time

from tensorboardX import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_search import config

from architect import Architect
from model_search import FBNet as Network
from model_infer import FBNet_Infer

from lr import LambdaLR
from perturb import Random_alpha

import operations
operations.DWS_CHWISE_QUANT = config.dws_chwise_quant

import argparse

parser = argparse.ArgumentParser(description='Search on ImageNet-100')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to ImageNet-100')
parser.add_argument('-b', '--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers')
parser.add_argument('--flops_weight', type=float, default=None,
                    help='weight of FLOPs loss')
parser.add_argument('--gpu', nargs='+', type=int, default=None,
                    help='specify gpus')
parser.add_argument('--world_size', type=int, default=None,
                    help='number of nodes')
parser.add_argument('--rank', type=int, default=None,
                    help='node rank')
parser.add_argument('--dist_url', type=str, default=None,
                    help='url used to set up distributed training')
parser.add_argument('--seed', type=int, default=12345,
                    help='random seed')
args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)


def main(pretrain=True):
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.flops_weight is not None:
        config.flops_weight = args.flops_weight
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

    pretrain = config.pretrain
    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False

    model = Network(config=config)

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


    if type(pretrain) == str:
        state = torch.load(pretrain + "/weights.pt")
        model.load_state_dict(state)

    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    
    parameters = []
    parameters += list(model.module.stem.parameters())
    parameters += list(model.module.cells.parameters())
    parameters += list(model.module.header.parameters())
    parameters += list(model.module.fc.parameters())
    
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            parameters,
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


    # data loader ###########################
    train_data = prepare_train_data(dataset=config.dataset,
                                      datadir=config.dataset_path+'/train')
    test_data = prepare_test_data(dataset=config.dataset,
                                    datadir=config.dataset_path+'/val')

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    train_data_model = torch.utils.data.Subset(train_data, indices[:split])
    train_data_arch = torch.utils.data.Subset(train_data, indices[split:num_train])

    if config.distributed:
        train_sampler_model = torch.utils.data.distributed.DistributedSampler(train_data_model)
        train_sampler_arch = torch.utils.data.distributed.DistributedSampler(train_data_arch)
    else:
        train_sampler_model = None
        train_sampler_arch = None

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, 
        sampler=train_sampler_model, shuffle=(train_sampler_model is None),
        pin_memory=False, num_workers=config.num_workers)

    train_loader_arch = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler_arch, shuffle=(train_sampler_arch is None),
        pin_memory=False, num_workers=config.num_workers)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=config.num_workers)


    # tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in range(config.nepochs):
        lr_policy.step()

        if config.perturb_alpha:
            epsilon_alpha = 0.03 + (config.epsilon_alpha - 0.03) * epoch / config.nepochs
            logging.info('Epoch %d epsilon_alpha %e', epoch, epsilon_alpha)
        else:
            epsilon_alpha = 0

        temp = config.temp_init * config.temp_decay ** epoch
        update_arch = epoch >= config.pretrain_epoch and not config.pretrain

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            logging.info("Temperature: " + str(temp))
            logging.info("[Epoch %d/%d] lr=%f" % (epoch + 1, config.nepochs, optimizer.param_groups[0]['lr']))
            logging.info("update arch: " + str(update_arch))


        train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, num_bits_list=config.num_bits_list, bit_schedule=config.bit_schedule, 
            bit_schedule_arch=config.bit_schedule_arch, loss_scale=config.loss_scale, update_arch=update_arch, epsilon_alpha=epsilon_alpha, criteria=config.criteria, temp=temp, 
            distill_weight=config.distill_weight, cascad_weight=config.cascad_weight, config=config)

        torch.cuda.empty_cache()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            # tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))

            with torch.no_grad():
                acc_bits_part, metric = infer(epoch, model, test_loader, logger, config.num_bits_list, finalize=True, temp=temp)

            acc_bits = []
            for acc in acc_bits_part:
                acc_new = reduce_tensor(acc, config.world_size)
                acc_bits.append(acc_new)

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                for i, num_bits in enumerate(config.num_bits_list):
                    logger.add_scalar('acc/val_bits_%d' % num_bits, acc_bits[i], epoch)

                logging.info("Epoch: " + str(epoch) +" Acc under different bits: " + str(acc_bits))

                state = {}
                
                if config.efficiency_metric == 'flops':
                    logger.add_scalar('flops/val', metric, epoch)
                    logging.info("Epoch: %d Flops: %.3f"%(epoch, metric))
                    state["flops"] = metric
                else:
                    logger.add_scalar('energy/val', metric, epoch)
                    logging.info("Epoch: %d Energy: %.3f"%(epoch, metric))
                    state["energy"] = metric

                state['alpha'] = getattr(model.module, 'alpha')
                state["acc"] = acc_bits

                torch.save(state, os.path.join(config.save, "arch_%d.pt"%(epoch)))
                save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

            if config.efficiency_metric == 'flops':
                if config.flops_weight > 0:
                    if metric < config.flops_min:
                        architect.flops_weight /= 2
                    elif metric > config.flops_max:
                        architect.flops_weight *= 2
                    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                        logger.add_scalar("arch/flops_weight", architect.flops_weight, epoch+1)
                        logging.info("arch_flops_weight = " + str(architect.flops_weight))
            else:
                if config.energy_weight > 0:
                    if metric < config.energy_min:
                        architect.energy_weight /= 2
                    elif metric > config.energy_max:
                        architect.energy_weight *= 2
                    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                        logger.add_scalar("arch/energy_weight", architect.energy_weight, epoch+1)
                        logging.info("arch_energy_weight = " + str(architect.energy_weight))

    if update_arch:
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            torch.save(state, os.path.join(config.save, "arch.pt"))


def train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, num_bits_list, bit_schedule, bit_schedule_arch, 
          loss_scale, update_arch=True, epsilon_alpha=0, criteria=None, temp=1, distill_weight=False, cascad_weight=False, config=None):
    model.train()

    # bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    # pbar = tqdm(range(len(train_loader_model)), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    if len(num_bits_list) == 1:
        bit_schedule = 'high2low'

    for step in range(len(train_loader_model)):
        input, target = dataloader_model.next()

        # end = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # time_data = time.time() - end
        # end = time.time()

        if update_arch:
            # pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_arch)))

            try:
                input_search, target_search = dataloader_arch.next()
            except:
                dataloader_arch = iter(train_loader_arch)
                input_search, target_search = dataloader_arch.next()

            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            loss_arch = architect.step(input, target, input_search, target_search, num_bits_list, bit_schedule_arch, loss_scale, temp=temp)

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                if (step+1) % 10 == 0:
                    for i, num_bits in enumerate(num_bits_list):
                        if loss_arch[i] != -1:
                            logger.add_scalar('loss_arch/num_bits_%d' % num_bits, loss_arch[i], epoch*len(train_loader_model)+step)

                    if config.efficiency_metric == 'flops':
                        logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch*len(train_loader_model)+step)
                    else:
                        logger.add_scalar('arch/energy_supernet', architect.energy_supernet, epoch*len(train_loader_model)+step)


        optimizer.zero_grad()

        loss_value = [-1 for _ in num_bits_list]

        if criteria is not None:
            if criteria == 'min':
                num_bits = min(num_bits_list)
            elif criteria == 'max':
                num_bits = max(num_bits_list)
            else:
                num_bits = np.random.choice(num_bits_list)
            
            logit = model(input, num_bits, temp=temp)
            loss = model.module._criterion(logit, target)

            loss = loss * loss_scale[num_bits_list.index(num_bits)]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            loss_value[num_bits_list.index(num_bits)] = loss.item()

        else:
            if bit_schedule == 'avg_loss':
                if distill_weight > 0:
                    if cascad_weight:
                        teacher_list = []
                        for num_bits in num_bits_list[::-1]:
                            logit = model(input, num_bits, temp=temp)
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
                        logit = model(input, num_bits_list[-1], temp=temp)
                        loss = model.module._criterion(logit, target)
                        loss = loss * loss_scale[-1]
                        loss.backward()
                        loss_value[-1] = loss.item()

                        logit_teacher = logit.detach()

                        del logit
                        del loss

                        for num_bits in num_bits_list[:-1]:
                            logit = model(input, num_bits, temp=temp)
                            loss = model.module._criterion(logit, target) + distill_weight * nn.MSELoss()(logit, logit_teacher)

                            loss = loss * loss_scale[num_bits_list.index(num_bits)]

                            loss.backward()

                            loss_value[num_bits_list.index(num_bits)] = loss.item()

                            del logit
                            del loss

                else:
                    for num_bits in num_bits_list:
                        logit = model(input, num_bits, temp=temp)
                        loss = model.module._criterion(logit, target)

                        loss = loss * loss_scale[num_bits_list.index(num_bits)]

                        loss.backward()

                        loss_value[num_bits_list.index(num_bits)] = loss.item()

                        del logit
                        del loss

                # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            else:
                print('Wrong Bit Schedule.')
                sys.exit()


        if step % 10 == 0:
            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                for i, num_bits in enumerate(num_bits_list):
                    if loss_value[i] != -1:
                        logger.add_scalar('loss/num_bits_%d' % num_bits, loss_value[i], epoch*len(train_loader_model)+step)
                        # logging.info("[Epoch %d/%d][Step %d/%d] Num_Bits=%d Loss=%.3f" % (epoch + 1, config.nepochs, step + 1, len(train_loader_model), num_bits, loss_value[i]))


    torch.cuda.empty_cache()
    if update_arch: del loss_arch


def infer(epoch, model, test_loader, logger, num_bits_list, finalize=False, temp=1):
    model.eval()
    prec1_list = []

    acc_bits = []

    for num_bits in num_bits_list:
        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input, volatile=True).cuda()
            target_var = Variable(target, volatile=True).cuda()

            output = model(input_var, num_bits, temp=temp)
            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list)/len(prec1_list)
        acc_bits.append(acc)

    if finalize:
        model_infer = FBNet_Infer(getattr(model.module, 'alpha'), config=config)
        if config.efficiency_metric == 'flops':
            flops = model_infer.forward_flops((3, 224, 224))
            del model_infer
            return acc_bits, flops
        else:
            energy = model_infer.forward_energy((3, 224, 224))
            del model_infer
            return acc_bits, energy

    else:
        return acc_bits


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
    main(pretrain=config.pretrain) 
