import os
import mmcv
from PIL import Image
from datasets import TreeSet
from models import SynthModel
import json
import argparse
import torch
from torch.utils.data import DataLoader
from utils import log_info, log_success, log_error, log_warn, log_highlight, load_model, AverageMeter


def synthesize(model, data_loader, use_gpu):
    log_highlight('Start synthsizing.')
    model.eval()

    save_path = args.output_path
    mmcv.mkdir_or_exist(save_path)

    for index, batch in enumerate(data_loader):
        img = batch['img']
        meta = batch['meta']
        origin_size = meta['origin_size'].squeeze()
        origin_size=(origin_size[0].item(),origin_size[1].item())
        if use_gpu:
            img = img.cuda()

        with torch.no_grad():
            pred = model(img, origin_size)

        # save valid result
        res = ((pred > 0)*255.0).squeeze().to(torch.uint8).data.cpu().numpy()
        res = Image.fromarray(res)
        res.save(os.path.join(
            save_path, meta['origin_data'][0].replace('jpg', 'png')))

        log_info('[Batch {}/{}]: save {} success.'.format(index +
                 1, len(data_loader), meta['origin_data'][0].replace('jpg', 'png')))

    log_highlight(
        'Synthesizing success, total images are {}.'.format(len(data_loader)))


def main(data_cfg, model_cfg, model_path, work_num, use_gpu):
    if use_gpu:
        if not torch.cuda.is_available():
            log_error('Cuda is not available! Try to use cpu.')
            use_gpu = False
    log_warn('Synthesizing use {}.\n'.format('GPU' if use_gpu else 'CPU'))

    # load model
    model = model_cfg['model']
    model = SynthModel(model_cfg[model])
    load_model(model, model_path)
    log_success('Load model {} success!\n'.format(model_path))

    # load data
    data_path = data_cfg['data_path']
    synth_config = data_cfg['synth']
    synth_set = TreeSet(synth_config['type'],
                        data_path, synth_config['short_size'])
    synth_loader = DataLoader(
        synth_set, batch_size=1, shuffle=False, num_workers=work_num, drop_last=False, pin_memory=True)
    log_success(
        'Load valid dataset success, data length is {}.\n'.format(len(synth_set)))

    # gpu settings
    if use_gpu:
        model = model.cuda()

    # synthesize
    synthesize(model, synth_loader, use_gpu)


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
    main(data_cfg, model_cfg, args.model_path, args.work_num, args.use_gpu)
