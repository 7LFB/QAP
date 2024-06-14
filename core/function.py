# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import scipy


import torch
import torch.nn as nn
from torch.nn import functional as F


from utils.tools import AverageMeter
from utils.tools import *
from models.losses import *

from pdb import set_trace as st

def train(model, trainloader, optimizer, writer_dict, epoch, args):
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    iters_per_epoch = len(trainloader)

    if args.loss_type=='focal':
        lossFunc=FocalLoss()
    elif args.loss_type=='ce':
        lossFunc=F.cross_entropy

    for i_iter, batch in enumerate(trainloader):

        if args.xprompt:
            images, prompts, labels = batch
            images = images.float().cuda()
            prompts = prompts.float().cuda()
            labels = labels.long().cuda()
            logits = model(images,prompts)
        else:
            images, labels = batch
            images = images.float().cuda()
            labels = labels.long().cuda()
            logits = model(images)
        
        losses = lossFunc(logits,labels)
        loss = losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())

        lr = optimizer.param_groups[0]["lr"]

        est_time=batch_time.average()*((args.num_epochs-epoch)*iters_per_epoch+iters_per_epoch-i_iter)/3600
        
        msg = f'Epoch: [{epoch}/{args.num_epochs}] Iter:[{i_iter}/{iters_per_epoch}], Time: {batch_time.average():.2f}, ERT: {est_time:.2f}, lr: {lr:.6f}, Loss: {ave_loss.average():.6f}' 
        print('\r',toGreen(msg),end='')
        

    # writer.add_scalar('train/loss', ave_loss.average(), global_steps)
    # writer.add_scalar('train/lr', lr, global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
        

def validate(model, valloader, writer_dict, args=None):
    model.eval()
    ave_loss = AverageMeter()

    if args.loss_type=='focal':
        lossFunc=FocalLoss()
    elif args.loss_type=='ce':
        lossFunc=F.cross_entropy

    with torch.no_grad():
        gts,probs,predicts=[],[],[]
        for _, batch in enumerate(valloader):
            if args.xprompt:
                images, prompts, labels = batch
                images = images.float().cuda()
                prompts = prompts.float().cuda()
                labels = labels.long().cuda()
                logits = model(images,prompts)
            else:
                images, labels = batch
                images = images.float().cuda()
                labels = labels.long().cuda()
                logits = model(images)

            preds = torch.argmax(logits,dim=-1)
            gts.append(labels.cpu().numpy())
            probs.append(scipy.special.softmax(logits.cpu().numpy(),axis=-1))
            predicts.append(preds.cpu().numpy())

            losses = lossFunc(logits,labels)
            loss = losses.mean()
            ave_loss.update(loss.item())

        # single label:
        gts=np.hstack(gts) 
        probs=np.concatenate(probs,axis=0)
        predicts=np.hstack(predicts)

        # multi labels
        # gts=np.vstack(gts) 
        # predicts=np.vstack(predicts)
        accuracy, specificity, recall, f1, auc = mymetrics(gts,probs,predicts)

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid/loss', ave_loss.average(), global_steps)
    # writer.add_scalar('valid/accuracy', accuracy, global_steps)
    # writer.add_scalar('valid/specificity', specificity, global_steps)
    # writer.add_scalar('valid/recall', recall, global_steps)
    # writer.add_scalar('valid/f1', f1, global_steps)
    # writer.add_scalar('valid/auc', auc, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy, specificity, recall, f1, auc


def test(model, testloader, writer_dict=None, args=None):
    model.eval()
    with torch.no_grad():
        gts,probs,predicts=[],[],[]
        for _, batch in enumerate(testloader):
            if args.xprompt:
                images, prompts, labels = batch
                images = images.float().cuda()
                prompts = prompts.float().cuda()
                labels = labels.long().cuda()
                logits = model(images,prompts)
            else:
                images, labels = batch
                images = images.float().cuda()
                labels = labels.long().cuda()
                logits = model(images)

            preds = torch.argmax(logits,dim=-1)

            gts.append(labels.cpu().numpy())
            probs.append(scipy.special.softmax(logits.cpu().numpy(),axis=-1))
            predicts.append(preds.cpu().numpy())

            # losses = F.cross_entropy(logits,labels)
            # loss = losses.mean()

        # single label:
        gts=np.hstack(gts) 
        probs=np.concatenate(probs,axis=0)
        predicts=np.hstack(predicts)

        # print(gts)
        # print(predicts)

        accuracy, specificity, recall, f1, auc = mymetrics(gts,probs,predicts)
        _, ovr_specificity, ovr_recall, ovr_f1, ovr_auc = mymetrics_without_avg(gts,probs,predicts)
        con_mat = confusion_matrix(gts, predicts)
        ovr_mat = np.array([ovr_specificity,ovr_recall,ovr_f1,ovr_auc])
        ovr_mat = np.round(ovr_mat*100,2)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['test_global_steps']
            # writer.add_scalar('test/accuracy', accuracy, global_steps)
            # writer.add_scalar('test/specificity', specificity, global_steps)
            # writer.add_scalar('test/recall', recall, global_steps)
            # writer.add_scalar('test/f1', f1, global_steps)
            # writer.add_scalar('test/auc', auc, global_steps)

            writer_dict['test_global_steps'] = global_steps + 1


        return gts,predicts,probs,accuracy, specificity, recall, f1, auc, con_mat, ovr_mat
