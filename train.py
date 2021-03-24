import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import os, argparse
import time
from model.decoder import Depth_Net
from utils import clip_gradient, adjust_lr
import dataset_loader

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', type=str, default='', help='train dataset root')
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.3, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--pre_train_mode', type=bool, default=False, help='pre_train_in_duts')
opt = parser.parse_args()

model = Depth_Net()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
train_loader = dataset_loader.getTrainingData(opt.train_root,1)
total_step = len(train_loader)

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    for i, sample_batched in enumerate(train_loader, start=1):

        optimizer.zero_grad()
        images, depth, focal = sample_batched['image'], sample_batched['depth'],sample_batched['focal']
        batch, slices, dim, h, w = focal.size()
        focal = focal.view(slices, dim, h, w)
        images = Variable(images)
        depth = Variable(depth)
        focals = Variable(focal)
        images = images.cuda()
        depth = depth.cuda()
        focals = focals.cuda()
        dps = model(images,focals)
        cos = nn.CosineSimilarity(dim=1, eps=0)
        get_gradient = Sobel().cuda()
        depth_grad = get_gradient(depth)
        output_grad = get_gradient(dps)
        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
        loss_depth = (torch.log(torch.abs(dps - depth) + 1).mean())
        loss_dx = (torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 1).mean())
        loss_dy = (torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 1).mean())
        loss_normal = (torch.abs(1 - cos(output_normal, depth_normal)).mean())
        loss1 = (loss_depth + loss_normal + (loss_dx + loss_dy))
        losses.update(loss1.item(), images.size(0))
        loss1.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        batch_time.update(time.time() - end)
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
    save_path = './checkpoint/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path + '%d' % epoch +  '_w.pth' )

print("Let's go!")
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if opt.pre_train_mode == False:
    progress = range(opt.start_epoch+1 , opt.epoch)
    print("training starts")
    for epoch in progress:
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
