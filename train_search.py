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
args = parser.parse_args()


def main(pretrain=True):
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.flops_weight is not None:
        config.flops_weight = args.flops_weight

    config.save = 'ckpt/{}'.format(config.save)
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Model #######################################
    model = Network(config=config)

    if args.gpu is not None:
        model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    if type(pretrain) == str:
        partial = torch.load(pretrain + "/weights.pt")
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
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
        logging.info("Wrong Optimizer Type.")
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
        logging.info("Wrong Learning Rate Schedule Type.")
        sys.exit()


    # # data loader ###########################
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(224, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010)),
    # ])

    # if config.dataset == 'cifar10':
    #     train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
    #     test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
    # elif config.dataset == 'cifar100':
    #     train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
    #     test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
    # else:
    #     print('Wrong dataset.')
    #     sys.exit()


    train_data = prepare_train_data(dataset=config.dataset,
                                      datadir=config.dataset_path+'/train')
    test_data = prepare_test_data(dataset=config.dataset,
                                    datadir=config.dataset_path+'/val')

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=config.num_workers)

    train_loader_arch = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False, num_workers=config.num_workers)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=config.num_workers)


    tbar = tqdm(range(config.nepochs), ncols=80)


    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))

        lr_policy.step()

        if config.perturb_alpha:
            epsilon_alpha = 0.03 + (config.epsilon_alpha - 0.03) * epoch / config.nepochs
            logging.info('Epoch %d epsilon_alpha %e', epoch, epsilon_alpha)
        else:
            epsilon_alpha = 0

        temp = config.temp_init * config.temp_decay ** epoch
        update_arch = epoch >= config.pretrain_epoch and not config.pretrain

        logging.info("Temperature: " + str(temp))
        logging.info("Update Arch: " + str(update_arch))


        if not config.hw_aware_nas:
            if config.efficiency_metric == 'edp' and epoch % config.update_hw_freq == 0:
                opt_hw_list = model.module.update_hw(size=(3, 224, 224), num_sample=config.num_sample, temp=temp)
                model.module.update_penalty(size=(3, 224, 224), opt_hw_list=opt_hw_list)

        else:
            opt_hw_list = model.module.update_hw(size=(3, 224, 224), num_sample=1, temp=temp)
            model.module.update_penalty(size=(3, 224, 224), opt_hw_list=opt_hw_list)

        # if epoch < 10:
        #     update_arch = False
        # else:
        #     update_arch = True

        train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, 
            update_arch=update_arch, epsilon_alpha=epsilon_alpha, temp=temp)
        torch.cuda.empty_cache()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))

            save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

            with torch.no_grad():
                if pretrain == True:
                    acc = infer(epoch, model, test_loader, logger, temp=temp)
                    logger.add_scalar('acc/val', acc, epoch)
                    logging.info("Epoch %d: acc %.3f"%(epoch, acc))

                else:
                    acc, metric = infer(epoch, model, test_loader, logger, temp=temp, finalize=True)

                    logger.add_scalar('acc/val', acc, epoch)
                    logging.info("Epoch %d: acc %.3f"%(epoch, acc))

                    state = {}
                    
                    if config.efficiency_metric == 'flops':
                        logger.add_scalar('flops/val', metric, epoch)
                        logging.info("Epoch %d: flops %.3f"%(epoch, metric))
                        state["flops"] = metric
                    else:
                        logger.add_scalar('edp/val', metric, epoch)
                        logging.info("Epoch %d: edp %.3f"%(epoch, metric))
                        state["edp"] = metric

                    state['alpha'] = getattr(model.module, 'alpha')
                    state["acc"] = acc

                    torch.save(state, os.path.join(config.save, "arch_%d.pt"%(epoch)))

                    if config.efficiency_metric == 'flops':
                        if config.flops_weight > 0 and update_arch:
                            if metric < config.flops_min:
                                architect.flops_weight /= 2
                            elif metric > config.flops_max:
                                architect.flops_weight *= 2
                            logger.add_scalar("arch/flops_weight", architect.flops_weight, epoch+1)
                            logging.info("arch_flops_weight = " + str(architect.flops_weight))
                    else:
                        if config.edp_weight > 0 and update_arch:
                            if metric < config.edp_min:
                                architect.edp_weight /= 2
                            elif metric > config.edp_max:
                                architect.edp_weight *= 2
                            logger.add_scalar("arch/edp_weight", architect.edp_weight, epoch+1)
                            logging.info("arch_edp_weight = " + str(architect.edp_weight))
                            
        if config.early_stop_by_skip and update_arch:
            groups = config.num_layer_list[1:-1]
            num_block = groups[0]

            current_arch = getattr(model.module, 'alpha').data[1:-1].argmax(-1)

            early_stop = False

            for group_id in range(len(groups)):
                num_skip = 0
                for block_id in range(num_block):
                    if current_arch[group_id * num_block + block_id] == 8:
                        num_skip += 1
                if num_skip >= 2:
                    early_stop = True

            if early_stop:
                print('Early Stop at epoch %d.' % epoch)
                break

    if update_arch:
        torch.save(state, os.path.join(config.save, "arch.pt"))


    if not config.hw_aware_nas:
        edp_final = model.module.eval_edp(size=(3, 224, 224))

        # model.module.update_penalty(size=(3, 224, 224), opt_hw_list=opt_hw_final)

        # model_infer = FBNet_Infer(getattr(model.module, 'alpha'), config=config)
        # edp = model_infer.forward_edp((3, 224, 224))

        logging.info("EDP of Final Arch:" + str(edp_final))

        opt_hw_final = {'opt_hw': model.module.opt_hw, 'edp': model.module.edp}
        torch.save(opt_hw_final, os.path.join(config.save, "opt_hw.pt"))

    else:
        model_infer = FBNet_Infer(getattr(model.module, 'alpha'), config=config)
        edp = model_infer.forward_edp((3, 224, 224))

        opt_hw_final = {'opt_hw': opt_hw_list[0], 'edp': edp}
        torch.save(opt_hw_final, os.path.join(config.save, "opt_hw.pt"))


def train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True, epsilon_alpha=0, temp=1):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(train_loader_model)), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        input, target = dataloader_model.next()

        # end = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # time_data = time.time() - end
        # end = time.time()

        if update_arch:
            pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_arch)))

            try:
                input_search, target_search = dataloader_arch.next()
            except:
                dataloader_arch = iter(train_loader_arch)
                input_search, target_search = dataloader_arch.next()

            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            loss_arch = architect.step(input, target, input_search, target_search, temp=temp)
            if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
                if config.efficiency_metric == 'flops':
                    logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch*len(pbar)+step)
                else:
                    logger.add_scalar('arch/edp_supernet', architect.edp_supernet, epoch*len(pbar)+step)

        # print(model.module.alpha[1])
        # print(model.module.ratio[1])

        if epsilon_alpha and update_arch:
            Random_alpha(model, epsilon_alpha)

        logit = model(input, temp)
        loss = model.module._criterion(logit, target)

        # time_fw = time.time() - end
        # end = time.time()

        optimizer.zero_grad()
        logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # time_bw = time.time() - end
        # end = time.time()

        # print("[Step %d/%d]" % (step + 1, len(train_loader_model)), 'Loss:', loss, 'Time Data:', time_data, 'Time Forward:', time_fw, 'Time Backward:', time_bw)

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()
    del loss
    if update_arch: del loss_arch


def infer(epoch, model, test_loader, logger, temp=1, finalize=False):
    model.eval()
    prec1_list = []

    for i, (input, target) in enumerate(test_loader):
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        output = model(input_var)
        prec1, = accuracy(output.data, target_var, topk=(1,))
        prec1_list.append(prec1)

    acc = sum(prec1_list)/len(prec1_list)

    if finalize:
        model_infer = FBNet_Infer(getattr(model.module, 'alpha'), config=config)
        if config.efficiency_metric == 'flops':
            flops = model_infer.forward_flops((3, 224, 224))
            return acc, flops
        else:
            edp = model_infer.eval_edp((3, 224, 224), epoch=100)
            return acc, edp

    else:
        return acc


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
