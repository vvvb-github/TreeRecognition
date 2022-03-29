import torch
import numpy as np
import os
import mmcv
from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image
from utils import image_utils
from utils import log_success, log_warn, log_error, log_info


class TreeSet(data.Dataset):
    def __init__(self, type, data_path, short_size=2560, img_size=None, scales=[0.7, 1.3], aspects=[0.9, 1.1], repeat_times=5) -> None:
        self._type = type
        self._data_path = data_path
        self._short_size = short_size
        self._img_size = img_size if (img_size is None or isinstance(
            img_size, tuple)) else (img_size, img_size)
        self._scales = scales
        self._aspects = aspects
        self._img_paths = list()
        self._gts_paths = list()
        # load all image path
        if type in ['train', 'valid']:
            img_dir = os.path.join(data_path, 'tree')
            gts_dir = os.path.join(data_path, type)
            for gts_name in mmcv.utils.scandir(gts_dir, '.png'):
                img_name = gts_name.replace('.png', '.jpg')
                img_name = os.path.join(img_dir, img_name)
                gts_name = os.path.join(gts_dir, gts_name)
                if os.path.exists(img_name):
                    self._img_paths.append(img_name)
                    self._gts_paths.append(gts_name)
                else:
                    log_error('Image {} not exists!'.format(img_name))
            if type == 'train':
                self._img_paths = self._img_paths*repeat_times
                self._gts_paths = self._gts_paths*repeat_times
        else:
            img_dir = os.path.join(data_path, 'synth')
            for img_name in mmcv.utils.scandir(img_dir, '.jpg'):
                img_name = os.path.join(img_dir, img_name)
                self._img_paths.append(img_name)

        log_success('Load {} set with length {} success!'.format(
            type, len(self._img_paths)))

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, index):
        if self._type == 'train':
            return self._load_train_data(index)
        elif self._type == 'valid':
            return self._load_valid_data(index)
        else:
            return self._load_synth_data(index)

    def _load_train_data(self, index):
        img_path = self._img_paths[index]
        gt_path = self._gts_paths[index]
        img = image_utils.load_img(img_path)
        gt = image_utils.load_img(gt_path)

        # random transform on images and align the size
        img, gt = image_utils.random_scale_with_gt(
            img, gt, self._short_size, self._scales, self._aspects)
        gt = Image.fromarray(gt)
        gt = (np.array(gt.convert('L')) < 255).astype(np.uint8)
        imgs = [img, gt]
        imgs = image_utils.random_horizontal_flip(imgs)
        imgs = image_utils.random_vertical_flip(imgs)
        imgs = image_utils.random_crop_padding(imgs, self._img_size)
        img, gt = imgs[0], imgs[1]

        # make tensor from image
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ColorJitter(
            brightness=32.0 / 255, saturation=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])(img)
        gt = torch.from_numpy(gt).long()

        data = dict(img=img, gt=gt)
        return data

    def _load_valid_data(self, index):
        img_path = self._img_paths[index]
        gt_path = self._gts_paths[index]
        img = image_utils.load_img(img_path)
        gt = image_utils.load_img(gt_path)

        # scale image and record meta information
        meta = dict(origin_data=img_path.split(
            '/')[-1], origin_size=np.array(img.shape[:2]))
        img = image_utils.scale_aligned_short(img, self._short_size)
        gt = image_utils.scale_aligned_short(gt, self._short_size)

        # make tensor from image
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])(img)
        gt = Image.fromarray(gt)
        gt = (np.array(gt.convert('L')) < 255).astype(np.uint8)
        gt = torch.from_numpy(gt).long()

        data = dict(img=img, gt=gt, meta=meta)
        return data

    def _load_synth_data(self, index):
        img_path = self._img_paths[index]
        img = image_utils.load_img(img_path)

        meta = dict(origin_data=img_path.split(
            '/')[-1], origin_size=np.array(img.shape[:2]))
        img = image_utils.scale_aligned_short(img, self._short_size)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])(img)

        data = dict(img=img, meta=meta)
        return data
