import os
import mmcv
from PIL import Image
from datasets import TreeSet
from models import FullModel, iou
import json
import argparse
import torch
from torch.utils.data import DataLoader
from utils import log_info,log_success,log_error,log_warn,log_highlight,load_model,AverageMeter


def valid(model,data_loader,use_gpu):
    log_highlight('Start validating model.')
    model.eval()

    losses = AverageMeter()
    ious = AverageMeter()
    save_path=args.output_path
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
                 1, len(data_loader), loss.item(), out_iou.item()))
    
    log_highlight('Validating model success, iou={:.3f}, loss={:.3f}.'.format(ious.avg,losses.avg))


def main(data_cfg,model_cfg,model_path,work_num,use_gpu):
    if use_gpu:
        if not torch.cuda.is_available():
            log_error('Cuda is not available! Try to use cpu.')
            use_gpu = False
    log_warn('Validating model {} use {}.\n'.format(model_path, 'GPU' if use_gpu else 'CPU'))

    # load model
    model = model_cfg['model']
    model = FullModel(model_cfg[model])
    load_model(model, model_path)
    log_success('Load model {} success!\n'.format(model_path))

    # load data
    data_path = data_cfg['data_path']
    valid_config = data_cfg['valid']
    valid_set = TreeSet(valid_config['type'],
                        data_path, valid_config['short_size'])
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False, num_workers=work_num, drop_last=False, pin_memory=True)
    log_success(
        'Load valid dataset success, data length is {}.\n'.format(len(valid_set)))

    # gpu settings
    if use_gpu:
        model=model.cuda()

    # validate
    valid(model,valid_loader,use_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('data_config', help='config file')
    parser.add_argument('model_config', help='config file')
    parser.add_argument('model_path', help='path of model to valid')
    parser.add_argument('output_path', help='output pictures path')
    parser.add_argument('--work_num', type=int, default=1)
    parser.add_argument('--use_gpu', action='store_true', help='if use gpu')
    args = parser.parse_args()

    data_cfg = json.load(open(args.data_config, 'r'))
    model_cfg = json.load(open(args.model_config, 'r'))
    torch.backends.cudnn.benchmark = True
    main(data_cfg,model_cfg,args.model_path,args.work_num,args.use_gpu)
