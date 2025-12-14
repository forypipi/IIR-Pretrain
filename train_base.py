import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
import os.path as osp
from shutil import copyfile

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import amp

from tensorboardX import SummaryWriter

from model import PSPNet

from dataset import coco, pascal
from util import transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, cosine_warmup_lr, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='PSPNet') # 
    parser.add_argument('--split', type=int, default=0) # 
    parser.add_argument('--snapshot_path', type=str) # 
    parser.add_argument('--result_path', type=str) # 
    parser.add_argument('--viz', action='store_true', default=True)
    parser.add_argument('--config', type=str, default='config/pascal/resnet50_base.yaml', help='config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')    
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):

    model = eval(args.arch).OneModel(args, mode='seg')

    if args.distributed:
        # Initialize Process Group
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.cuda()

    # Resume
    # get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():       
                logger.info("=> no checkpoint found at '{}'".format(resume_path))


    # Get model para.
    total_number, learnable_number = get_model_para_number(model)

    if main_process():
        print('Number of Parameters: %d' % (total_number))
        print('Number of Learnable Parameters: %d' % (learnable_number))

    # model = torch.compile(model)
    return model

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def main():

    gpu_count = torch.cuda.device_count()

    # 打印每块GPU的型号
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        current_device = os.environ['CUDA_VISIBLE_DEVICES']
        print(f"current GPU {i}: {props.name} {current_device}")


    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    assert args.classes > 1
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    
    if main_process():
        logger.info("=> creating model ...")
    model = get_model(args)
    if main_process():
        logger.info(model)
    if main_process() and args.viz:
        writer = SummaryWriter(args.result_path)

    print('result_path:', args.result_path)
    print('snapshot_path:', args.snapshot_path)
    if main_process():
        print('Warmup: {}% steps'.format(args.warmup))

# ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    data_dict = {
        'coco': coco,
        'pascal': pascal,
    }
    # Train
    train_transform = transform.Compose([
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    
    train_data = data_dict[args.data_set].BaseData(split=args.split, mode='trn', data_root=args.data_root, data_list=args.train_list, \
                                use_split_coco=args.use_split_coco, \
                                transform=train_transform, batch_size=args.batch_size)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, \
                                                pin_memory=True, sampler=train_sampler, drop_last=True, \
                                                shuffle=False if args.distributed else True)
                                                
    # Val
    if args.evaluate:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = data_dict[args.data_set].BaseData(split=args.split, mode='val', data_root=args.data_root, data_list=args.val_list, \
                                    data_set=args.data_set, use_split_coco=args.use_split_coco, \
                                    transform=val_transform, main_process=main_process(), batch_size=args.batch_size_val)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, \
                                                 shuffle=False, num_workers=args.workers, \
                                                    pin_memory=True, sampler=None)

# ----------------------  TRAINVAL  ----------------------
    global best_miou, best_epoch, keep_epoch, val_num, best_acc
    best_miou = 0.
    best_acc = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)    

        # ----------------------  TRAIN  ----------------------

        if epoch == int(args.epochs*(1-args.classification_rate)):

            param = torch.load(args.snapshot_path + '/best.pth', map_location=lambda storage, loc: storage.cuda())['state_dict']
            model.load_state_dict(param, strict=False)
            model.mode = 'cls'
            model.freeze_modules()
            print("<"*10+"Train FC layer"+"<"*10)

        if epoch < int(args.epochs*(1-args.classification_rate)):
            model.mode = 'seg'
            model.freeze_modules()
            optimizer = model.get_optim(args, LR=args.base_lr)      # LR is not set
            train(train_loader, model, optimizer, epoch, args, mode='seg')
        else:
            model.mode = 'cls'
            optimizer = model.get_optim(args, LR=args.base_lr)      # LR is not set
            train(train_loader, model, optimizer, epoch, args, mode='cls')

        # save model for <resuming>
        if (epoch % args.save_freq == 0) and (epoch > 0) and main_process():
            filename = args.snapshot_path + '/epoch_{}.pth'.format(epoch)
            logger.info('Saving checkpoint to: ' + filename)
            if osp.exists(filename):
                os.remove(filename)            
            torch.save({'epoch': epoch, 'state_dict': model.get_state_dict_without_teacher(), 'optimizer': optimizer.state_dict()}, filename)

        # -----------------------  VAL  -----------------------
        if args.evaluate and epoch%1==0:
            mIoU, acc, val_loss, cls_loss = validate(val_loader, model)            
            val_num += 1
            if main_process() and args.viz:
                writer.add_scalar('mIoU_val', mIoU, epoch_log)
                writer.add_scalar('acc_val', acc, epoch_log)
                writer.add_scalar('segloss_val', val_loss, epoch_log)
                writer.add_scalar('clsloss_val', cls_loss, epoch_log)

        # save model for <testing>
            if epoch < args.epochs*(1-args.classification_rate) and mIoU > best_miou:
                best_miou, best_epoch = mIoU, epoch
                keep_epoch = 0
                if main_process():
                    filename = f'{args.snapshot_path}/train_epoch_{epoch}_{best_miou:.4f}_{best_acc:.4f}.pth'
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.get_state_dict_without_teacher(), 'optimizer': optimizer.state_dict()}, filename)
                    copyfile(filename, args.snapshot_path + '/best.pth')

            elif epoch >= args.epochs*(1-args.classification_rate) and acc > best_acc:
                best_acc, best_epoch = acc, epoch
                keep_epoch = 0
                if main_process():
                    filename = f'{args.snapshot_path}/train_epoch_{epoch}_{best_miou:.4f}_{best_acc:.4f}.pth'
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.get_state_dict_without_teacher(), 'optimizer': optimizer.state_dict()}, filename)
                    copyfile(filename, args.snapshot_path + '/best.pth')


    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))            
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<< <<<<<<<<<<<<<<')
        print(args.arch + f'\t Group:{args.split} \t Best_mIoU:{best_miou:.4f} \t Best_acc:{best_acc:.4f} \t Best_step:{best_epoch}')
        print('>'*80)
        print ('%s' % datetime.datetime.now())


def train(train_loader, model, optimizer, epoch, args, mode=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    segloss_meter = AverageMeter()
    clsloss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()

    max_iter = args.epochs*len(train_loader)
    max_seg_iter = (1-args.classification_rate)*args.epochs*len(train_loader)
    end = time.time()
    val_time = 0.

    scaler = amp.GradScaler()

    for i, (image, target, binary_y) in enumerate(train_loader):

        data_time.update(time.time() - end - val_time)
        current_iter = epoch * len(train_loader) + i + 1
        
        # set LR
        # poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader)//2)
        if mode == 'seg':
            cosine_warmup_lr(optimizer, args.base_lr, current_iter, max_seg_iter, index_split=args.index_split, warmup_iter=args.warmup*max_seg_iter)
        elif mode == 'cls':
            cosine_warmup_lr(optimizer, args.base_lr,
                             current_iter-max_seg_iter, 
                             max_iter-max_seg_iter,
                             index_split=args.index_split, 
                             warmup_iter=args.warmup*(max_iter-max_seg_iter))


        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning_Rate", current_lr, current_iter)
        
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        binary_y = binary_y.cuda(non_blocking=True)

        with amp.autocast('cuda', dtype=torch.bfloat16):
            output, seg_loss, cls_loss, cls_acc = model(x=image, y=target, binary_y=binary_y)

        if mode == 'seg':
            loss = seg_loss
        elif mode == 'cls':
            loss = cls_loss

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n = image.size(0) # batch_size

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target), acc_meter.update(cls_acc.mean())
        
        segloss_meter.update(seg_loss.item(), n)
        clsloss_meter.update(cls_loss.item(), n)
        acc_meter.update(cls_acc.mean().item(), n)

        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '                      
                        'SegLoss {segloss_meter.avg:.4f} '
                        'ClsLoss {clsloss_meter.avg:.4f} '
                        'Accuracy {acc_meter.avg:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        segloss_meter=segloss_meter,
                                                        clsloss_meter=clsloss_meter,
                                                        acc_meter=acc_meter))
            if args.viz:
                writer.add_scalar('segloss_train', segloss_meter.avg, current_iter)
                writer.add_scalar('clsloss_train', clsloss_meter.avg, current_iter)
                writer.add_scalar('clsacc_train', acc_meter.avg, current_iter)
                acc_meter.reset()
                clsloss_meter.reset()
                segloss_meter.reset()



    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc {:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, acc_meter.avg))


def validate(val_loader, model: PSPNet.OneModel):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    segloss_meter = AverageMeter()
    clsloss_meter = AverageMeter()
    acc_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    class_intersection_meter = [0]*(args.classes-1)
    class_union_meter = [0]*(args.classes-1)

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    model.teacher.eval()
    model.encoder.eval()
    model.ppm.eval()
    model.cls.eval()

    end = time.time()
    val_start = end

    iter_num = 0

    for i, logits in enumerate(val_loader):
        iter_num += 1
        data_time.update(time.time() - end)

        if args.batch_size_val == 1:
            image, target, binary_y, ori_label = logits
            ori_label = ori_label.cuda(non_blocking=True)
        else:
            image, target, binary_y = logits                
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        binary_y = binary_y.cuda(non_blocking=True)

        start_time = time.time()
        with torch.no_grad():
            output, ClsLoss, cls_acc = model(x=image, y=target, binary_y=binary_y)
        model_time.update(time.time() - start_time)

        output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)

        SegLoss = criterion(output, target)

        output = output.max(1)[1]

        intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target), acc_meter.update(cls_acc.mean())
        for idx in range(1,len(intersection)):
            class_intersection_meter[idx-1] += intersection[idx]
            class_union_meter[idx-1] += union[idx]

        
        segloss_meter.update(SegLoss.item(), image.size(0))
        clsloss_meter.update(ClsLoss.item(), image.size(0))
        acc_meter.update(cls_acc.mean().item(), image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((iter_num % 100 == 0) or (iter_num == len(val_loader))) and main_process():
            logger.info(f'Test: [{iter_num}/{len(val_loader)}] '
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        f'SegLoss {segloss_meter.val:.4f} ({segloss_meter.avg:.4f})'
                        f'ClsLoss {clsloss_meter.val:.4f} ({clsloss_meter.avg:.4f})')
    val_time = time.time()-val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)

    if main_process():
        logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou)) 
        logger.info('meanAcc---Val result: mAcc {:.4f}.'.format(acc_meter.avg)) 
        for i in range(len(class_intersection_meter)):
            logger.info('Class_{} Result: iou_b {:.4f}.'.format(i+1, class_iou_class[i]))   
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, iter_num))

    return class_miou, acc_meter.avg, segloss_meter.avg, clsloss_meter.avg

if __name__ == '__main__':
    print (datetime.datetime.now())
    main()
