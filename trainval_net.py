# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether to perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '10']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '10']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb_train, roidb_train, ratio_list_train, ratio_index_train = combined_roidb(args.imdb_name,cfg.TRAIN.USE_FLIPPED)
  train_size = len(roidb_train)
  print('{:d} roidb_train entries'.format(len(roidb_train)))

  #test set
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda
  imdb_test, roidb_test, ratio_list_test, ratio_index_test = combined_roidb(args.imdbval_name,cfg.TRAIN.USE_FLIPPED,training=False)
  # imdb_test.competition_mode(on=True)
  print('{:d} roidb_train entries'.format(len(roidb_test)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  #train dataset dataloader
  train_dataset = roibatchLoader(roidb_train, ratio_list_train, ratio_index_train, args.batch_size, \
                           imdb_train.num_classes, training=True)

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  #test dataset dataloader
  test_dataset = roibatchLoader(roidb_test, ratio_list_test, ratio_index_test, args.batch_size, \
                           imdb_test.num_classes)

  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=0)

  # initilize the train tensor holder here.
  im_data_train = torch.FloatTensor(1)
  im_info_train = torch.FloatTensor(1)
  num_boxes_train = torch.LongTensor(1)
  gt_boxes_train = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data_train = im_data_train.cuda()
    im_info_train = im_info_train.cuda()
    num_boxes_train = num_boxes_train.cuda()
    gt_boxes_train = gt_boxes_train.cuda()

  # make variable
  im_data_train = Variable(im_data_train)
  im_info_train = Variable(im_info_train)
  num_boxes_train = Variable(num_boxes_train)
  gt_boxes_train = Variable(gt_boxes_train)

  # initilize the test tensor holder here.
  im_data_test = torch.FloatTensor(1)
  im_info_test = torch.FloatTensor(1)
  num_boxes_test = torch.LongTensor(1)
  gt_boxes_test = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data_test = im_data_test.cuda()
    im_info_test = im_info_test.cuda()
    num_boxes_test = num_boxes_test.cuda()
    gt_boxes_test = gt_boxes_test.cuda()

  # make variable
  im_data_test = Variable(im_data_test)
  im_info_test = Variable(im_info_test)
  num_boxes_test = Variable(num_boxes_test)
  gt_boxes_test = Variable(gt_boxes_test)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb_train.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb_train.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb_train.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb_train.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")
  minimal_test_loss = float('Inf')
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    ###### Train
    fasterRCNN.train()
    loss_temp_train = 0
    start_train = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    train_data_iter = iter(train_dataloader)
    for step in range(iters_per_epoch):
      train_data = next(train_data_iter)
      im_data_train.data.resize_(train_data[0].size()).copy_(train_data[0])
      im_info_train.data.resize_(train_data[1].size()).copy_(train_data[1])
      gt_boxes_train.data.resize_(train_data[2].size()).copy_(train_data[2])
      num_boxes_train.data.resize_(train_data[3].size()).copy_(train_data[3])

      fasterRCNN.zero_grad()
      rois_train, cls_prob_train, bbox_pred_train, \
      rpn_loss_cls_train, rpn_loss_box_train, \
      RCNN_loss_cls_train, RCNN_loss_bbox_train, \
      rois_label_train = fasterRCNN(im_data_train, im_info_train, gt_boxes_train, num_boxes_train)

      loss_train = rpn_loss_cls_train.mean() + rpn_loss_box_train.mean() \
           + RCNN_loss_cls_train.mean() + RCNN_loss_bbox_train.mean()
      loss_temp_train += loss_train.item()

      # backward
      optimizer.zero_grad()
      loss_train.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end_train = time.time()
        if step > 0:
          loss_temp_train /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls_train = rpn_loss_cls_train.mean().item()
          loss_rpn_box_train = rpn_loss_box_train.mean().item()
          loss_rcnn_cls_train = RCNN_loss_cls_train.mean().item()
          loss_rcnn_box_train = RCNN_loss_bbox_train.mean().item()
          fg_cnt = torch.sum(rois_label_train.data.ne(0))
          bg_cnt = rois_label_train.data.numel() - fg_cnt
        else:
          loss_rpn_cls_train = rpn_loss_cls_train.item()
          loss_rpn_box_train = rpn_loss_box_train.item()
          loss_rcnn_cls_train = RCNN_loss_cls_train.item()
          loss_rcnn_box_train = RCNN_loss_bbox_train.item()
          fg_cnt = torch.sum(rois_label_train.data.ne(0))
          bg_cnt = rois_label_train.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp_train, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end_train-start_train))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls_train, loss_rpn_box_train, loss_rcnn_cls_train, loss_rcnn_box_train))
        if args.use_tfboard:
          info_train = {
            'loss_train': loss_temp_train,
            'loss_rpn_cls_train': loss_rpn_cls_train,
            'loss_rpn_box_train': loss_rpn_box_train,
            'loss_rcnn_cls_train': loss_rcnn_cls_train,
            'loss_rcnn_box_train': loss_rcnn_box_train
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info_train, (epoch - 1) * iters_per_epoch + step)

        loss_temp_train = 0
        start_train = time.time()

    ##Test
    fasterRCNN.eval()
    loss_temp_test = 0
    test_data_iter = iter(test_dataloader)
    for i in range(len(imdb_test.image_index)):
      test_data = next(test_data_iter)
      im_data_test.data.resize_(test_data[0].size()).copy_(test_data[0])
      im_info_test.data.resize_(test_data[1].size()).copy_(test_data[1])
      gt_boxes_test.data.resize_(test_data[2].size()).copy_(test_data[2])
      num_boxes_test.data.resize_(test_data[3].size()).copy_(test_data[3])

      rois_test, cls_prob_test, bbox_pred_test, \
      rpn_loss_cls_test, rpn_loss_box_test, \
      RCNN_loss_cls_test, RCNN_loss_bbox_test, \
      rois_label_test = fasterRCNN(im_data_test, im_info_test, gt_boxes_test, num_boxes_test)

      loss_test = rpn_loss_cls_test.mean() + rpn_loss_box_test.mean() \
           + RCNN_loss_cls_test.mean() + RCNN_loss_bbox_test.mean()
      loss_temp_test += loss_test.item()

    print("[epoch %2d] loss: %.4f" \
        % (epoch, loss_temp_test/len(imdb_test.image_index)))

    if args.use_tfboard:
        info_test = {
            'loss_test': loss_temp_test/len(imdb_test.image_index)
        }
        logger.add_scalars("logs_s_{}/losses".format(args.session), info_test, epoch)
    # is_minimal = False
    if loss_temp_test < minimal_test_loss:
        is_minimal = True
        minimal_test_loss = loss_temp_test
    save_name = os.path.join(output_dir, 'faster_rcnn_{}.pth'.format(args.session))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name, is_minimal=is_minimal)
    print('save model: {}'.format(save_name))
    loss_temp_test = 0

  if args.use_tfboard:
    logger.close()
