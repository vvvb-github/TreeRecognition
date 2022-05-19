import os
import mmcv
from PIL import Image
from time import time
from datasets import TreeSet
from models import FullModel, iou
import json
import argparse
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler as GradScalar
from torch.utils.data import DataLoader
from utils import log_success, log_warn, log_error, log_info, log_highlight, AverageMeter, load_model, save_model

# try:
#     import apex
# except:
#     log_error('Import apex fail!')
# try:
#     import wandb
# except:
#     log_error('Import wandb fail!')


def adjust_learning_rate(optimizer, data_loader, base_lr, epoch, max_epoch, iter):
    cur_iter = epoch*len(data_loader)+iter
    max_iter = max_epoch*len(data_loader)
    lr = base_lr*(1-float(cur_iter)/max_iter)**0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(data_loader, model, optimizer, epoch, cfg, use_gpu, scalar=None):
    log_highlight(
        'Start training epoch {}/{}.'.format(epoch + 1, cfg['epoch']))
    model.train()

    batch_time = AverageMeter()
    loader_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()

    start_time = time()
    batch_start = time()
    for index, batch in enumerate(data_loader):
        loader_time.update(time()-batch_start)

        adjust_learning_rate(optimizer, data_loader,
                             cfg['learning_rate'], epoch, cfg['epoch'], index)

        img = batch['img']
        gt = batch['gt']
        if use_gpu:
            img = img.cuda()
            gt = gt.cuda()

        optimizer.zero_grad()
        if args.amp:
            with autocast():
                pred, loss = model(img, gt)
            if use_gpu:
                loss = torch.mean(loss)
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            pred, loss = model(img, gt)
            if use_gpu:
                loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

        out_iou = iou((pred > 0).long(), gt)
        losses.update(loss.item())
        ious.update(out_iou.item())

        batch_time.update(time()-batch_start)

        # log
        log_info('[Batch {}/{}]: learning rate {:.6f}, batch time {:.3f}s, total time {:.0f}min, ETA {:.0f}min, loss {:.3f}, iou {:.3f}'.format(
            index+1,
            len(data_loader),
            optimizer.param_groups[0]['lr'],
            batch_time.avg,
            (time()-start_time)/60.0,
            batch_time.avg*(len(data_loader)-index)/60.0,
            losses.avg,
            ious.avg
        ))

        batch_start = time()

    log_info('Training epoch {} cost total time {}s.'.format(
        epoch, time()-start_time))


def valid(data_loader, model, epoch, use_gpu):
    log_highlight(
        'Validating model after training for {} epoch.'.format(epoch+1))
    model.eval()

    losses = AverageMeter()
    ious = AverageMeter()

    save_path = os.path.join(
        args.output_path, 'train/valid_epoch{}'.format(epoch+1))
    mmcv.mkdir_or_exist(save_path)

    for index, batch in enumerate(data_loader):
        img = batch['img']
        gt = batch['gt']
        if use_gpu:
            img = img.cuda()
            gt = gt.cuda()

        with torch.no_grad():
            pred, loss = model(img, gt)

        out_iou = iou((pred > 0).long(), gt)
        losses.update(loss.item())
        ious.update(out_iou.item())

        # save valid result
        res = ((pred > 0)*255.0).squeeze().to(torch.uint8).data.cpu().numpy()
        res = Image.fromarray(res)
        meta = batch['meta']
        res.save(os.path.join(
            save_path, meta['origin_data'][0].replace('jpg', 'png')))

        log_info('[Batch {}/{}]: loss {:.3f}, iou {:.3f}'.format(index +
                 1, len(data_loader), losses.avg, ious.avg))

    return ious.avg


def save_checkpoint(epoch, model, checkpoint_path):
    save_path = os.path.join(checkpoint_path, 'newest.pth')
    save_model(model, save_path)

    # save checkpoint each 10 epochs
    if epoch % 10 == 0:
        save_path = os.path.join(
            checkpoint_path, 'checkpoint_epoch{}.pth'.format(epoch))
        save_model(model, save_path)


def main(model, data_cfg, model_cfg, train_cfg, work_num, use_gpu):
    if use_gpu:
        if not torch.cuda.is_available():
            log_error('Cuda is not available! Try to use cpu.')
            use_gpu = False
    log_warn('Using {} for train.\n'.format('GPU' if use_gpu else 'CPU'))

    # load model
    model = FullModel(model_cfg[model])
    if args.checkpoint != '':
        save_path = os.path.join(train_cfg['checkpoint_path'], args.checkpoint)
        load_model(model, save_path)
    log_success('Load model success!\n')

    # load data
    data_path = data_cfg['data_path']
    train_config = data_cfg['train']
    batch_size = train_config['batch_size']
    train_set = TreeSet(train_config['type'], data_path, train_config['short_size'], train_config['img_size'],
                        train_config['scales'], train_config['aspects'], train_config['repeat_times'])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=work_num, drop_last=True, pin_memory=True)
    log_success(
        'Load train dataset success, data length is {}.'.format(len(train_set)))
    valid_config = data_cfg['valid']
    valid_set = TreeSet(valid_config['type'],
                        data_path, valid_config['short_size'])
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False, num_workers=work_num, drop_last=False, pin_memory=True)
    log_success(
        'Load valid dataset success, data length is {}.\n'.format(len(valid_set)))

    # optimizer
    if train_cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg['learning_rate'])

    if args.amp:
        scalar = GradScalar()
    else:
        scalar = None

    # gpu settings
    if use_gpu:
        model = torch.nn.DataParallel(
            model, device_ids=train_cfg['devices']).cuda()

    # start training
    best_iou = 0
    best_epoch = 0
    for epoch in range(train_cfg['epoch']):
        train(train_loader, model, optimizer,
              epoch, train_cfg, use_gpu, scalar)

        # valid and save checkpoint
        valid_iou = valid(valid_loader, model, epoch, use_gpu)
        if valid_iou > best_iou:
            best_iou = valid_iou
            best_epoch = epoch+1
            save_path = os.path.join(train_cfg['checkpoint_path'], 'best.pth')
            save_model(model, save_path)
        log_highlight('Best Epoch: {}, Best IoU: {:.3f}.\n'.format(
            best_epoch, best_iou))
        save_checkpoint(epoch+1, model, train_cfg['checkpoint_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('data_config', help='config file')
    parser.add_argument('model_config', help='config file')
    parser.add_argument('process_config', help='config file')
    parser.add_argument('output_path', help='output pictures path')
    parser.add_argument('model', help='model')
    parser.add_argument('--work_num', type=int, default=1)
    parser.add_argument('--use_gpu', action='store_true', help='if use gpu')
    parser.add_argument('--amp', action='store_true', help='if use amp')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()

    data_cfg = json.load(open(args.data_config, 'r'))
    model_cfg = json.load(open(args.model_config, 'r'))
    train_cfg = json.load(open(args.process_config, 'r'))['train']
    torch.backends.cudnn.benchmark = True
    main(args.model, data_cfg, model_cfg, train_cfg, args.work_num, args.use_gpu)
