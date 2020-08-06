import os
import json
import cv2
import numpy as np
import random

from shapely import wkt
from shapely.geometry import Polygon
import torch
from torch.utils.data import Dataset, DataLoader
from utils import preprocess
import torchvision.transforms as transforms
import multiprocessing

multiprocessing.set_start_method('spawn', True)


class XView2Dataset(Dataset):
    """xView2
    input: Post image
    target: pixel-wise classes
    """
    dmg_type = {'background': 0, 'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4,
                'un-classified': 255}
    diaster_type = {'earthquake': 0, 'fire': 1, 'tsunami': 2, 'volcano': 3, 'wind': 4, 'flooding': 5}

    def __init__(self, root_dir, rgb_bgr='rgb', preprocessing=None, mode='train'):
        assert mode in ('train', 'test')
        self.mode = mode
        self.root = root_dir
        assert rgb_bgr in ('rgb', 'bgr')
        self.rgb = bool(rgb_bgr == 'rgb')
        self.preprocessing = preprocessing
        self.dirs = {'train_imgs': os.path.join(self.root, 'train', 'images'),
                     'train_labs': os.path.join(self.root, 'train', 'labels'),
                     'tier3_imgs': os.path.join(self.root, 'tier3', 'images'),
                     'tier3_labs': os.path.join(self.root, 'tier3', 'labels'),
                     'test_imgs': os.path.join(self.root, 'test', 'images')}
        train_imgs = [s for s in os.listdir(self.dirs['train_imgs'])]
        tier3_imgs = [s for s in os.listdir(self.dirs['tier3_imgs'])]
        train_labs = [s for s in os.listdir(self.dirs['train_labs'])]
        tier3_labs = [s for s in os.listdir(self.dirs['tier3_labs'])]
        test_imgs = [s for s in os.listdir(self.dirs['test_imgs'])]

        self.sample_files = []
        self.neg_sample_files = []
        if self.mode == 'train':
            self.add_samples_train(self.dirs['train_imgs'], self.dirs['train_labs'], train_imgs, train_labs)
            self.add_samples_train(self.dirs['tier3_imgs'], self.dirs['tier3_labs'], tier3_imgs, tier3_labs)
        else:
            for pre in os.listdir(self.dirs['test_imgs']):
                if pre[:9] != 'test_pre_':
                    continue
                img_id = pre[9:][:-4]
                post = 'test_post_' + pre[9:]
                assert post in test_imgs
                files = {'img_id': img_id,
                         'pre_img': os.path.join(self.dirs['test_imgs'], pre),
                         'post_img': os.path.join(self.dirs['test_imgs'], post)}
                self.sample_files.append(files)

        if mode == 'test':
            self.data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def add_samples_train(self, img_dirs, lab_dirs, imgs, labs):
        for pre in os.listdir(img_dirs):
            if pre[-17:] != '_pre_disaster.png':
                continue
            chop = pre[:-4].split('_')
            img_id = '_'.join(chop[:2])
            post = img_id + '_post_disaster.png'
            pre_json = img_id + '_pre_disaster.json'
            post_json = img_id + '_post_disaster.json'
            assert post in imgs
            assert pre_json in labs
            assert post_json in labs
            assert img_id not in self.sample_files
            files = {'img_id': img_id,
                     'pre_img': os.path.join(img_dirs, pre),
                     'post_img': os.path.join(img_dirs, post),
                     'pre_json': os.path.join(lab_dirs, pre_json),
                     'post_json': os.path.join(lab_dirs, post_json)}
            self.sample_files.append(files)

    def get_sample_info(self, idx):
        files = self.sample_files[idx]
        pre_img = cv2.imread(files['pre_img'])
        post_img = cv2.imread(files['post_img'])
        if self.rgb:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
            post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
        pre_json = json.loads(open(files['pre_json']).read())
        post_json = json.loads(open(files['post_json']).read())
        sample = {'pre_img': pre_img, 'post_img': post_img, 'image_id': files['img_id'],
                  'im_width': post_json['metadata']['width'],
                  'im_height': post_json['metadata']['height'],
                  'disaster': post_json['metadata']['disaster_type'],
                  'pre_meta': {m: pre_json['metadata'][m] for m in pre_json['metadata']},
                  'post_meta': {m: post_json['metadata'][m] for m in post_json['metadata']},
                  'pre_builds': dict(), 'post_builds': dict(), 'builds': dict()}
        for b in pre_json['features']['xy']:
            buid = b['properties']['uid']
            sample['pre_builds'][buid] = {p: b['properties'][p] for p in b['properties']}
            poly = Polygon(wkt.loads(b['wkt']))
            sample['pre_builds'][buid]['poly'] = list(poly.exterior.coords)
        for b in post_json['features']['xy']:
            buid = b['properties']['uid']
            sample['post_builds'][buid] = {p: b['properties'][p] for p in b['properties']}
            poly = Polygon(wkt.loads(b['wkt']))
            sample['post_builds'][buid]['poly'] = list(poly.exterior.coords)
            sample['builds'][buid] = {'poly': list(poly.exterior.coords),
                                      'subtype': b['properties']['subtype']}
        # sample['mask_img'] = self.make_mask_img(**sample)
        return sample

    def __getitem__(self, idx):
        files = self.sample_files[idx]
        pre_img = cv2.imread(files['pre_img'])
        post_img = cv2.imread(files['post_img'])
        if self.rgb:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
            post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            sample = self.get_sample_with_mask(files, pre_img, post_img)
            sample['image_id'] = files['img_id']
            if self.preprocessing is not None:
                transformed = preprocess(sample['pre_img'], sample['post_img'], sample['mask_img'],
                                         flip=self.preprocessing['flip'],
                                         scale=self.preprocessing['scale'],
                                         crop=self.preprocessing['crop'])
                sample['pre_img'] = transformed[0]
                sample['post_img'] = transformed[1]
                sample['mask_img'] = transformed[2]
        else:
            pre_img = self.data_transforms(pre_img)
            post_img = self.data_transforms(post_img)
            sample = {'pre_img': pre_img, 'post_img': post_img, 'image_id': files['img_id']}
        return sample

    @staticmethod
    def _get_building_from_json(post_json):
        buildings = dict()
        for b in post_json['features']['xy']:
            buid = b['properties']['uid']
            poly = Polygon(wkt.loads(b['wkt']))
            buildings[buid] = {'poly': list(poly.exterior.coords),
                               'subtype': b['properties']['subtype']}
        return buildings

    def get_sample_with_mask(self, files, pre_img, post_img):
        post_json = json.loads(open(files['post_json']).read())
        sample = {'pre_img': pre_img, 'post_img': post_img, 'image_id': files['img_id'],
                  'disaster': self.diaster_type[post_json['metadata']['disaster_type']]}

        buildings = self._get_building_from_json(post_json)
        sample['mask_img'] = self.make_mask_img(**buildings)
        return sample

    def make_mask_img(self, **kwargs):
        width = 1024
        height = 1024
        builings = kwargs

        mask_img = np.zeros([height, width], dtype=np.uint8)
        for dmg in self.dmg_type:
            polys_dmg = [np.array(builings[p]['poly']).round().astype(np.int32).reshape(-1, 1, 2)
                         for p in builings if builings[p]['subtype'] == dmg]
            cv2.fillPoly(mask_img, polys_dmg, [self.dmg_type[dmg]])

        return mask_img

    def show_sample(self, **kwargs):
        pass

    def __len__(self):
        return len(self.sample_files)


if __name__ == '__main__':
    root_path = "/mnt/Dataset/xView2/v2"
    dataset = XView2Dataset(root_path, rgb_bgr='rgb', preprocessing={'flip': True, 'scale': None, 'crop': (513, 513)})
    dataset_test = XView2Dataset(root_path, rgb_bgr='rgb',
                                 preprocessing={'flip': False, 'scale': (0.8, 2.0), 'crop': (1024, 1024)})

    n_samples = len(dataset)
    n_train = int(n_samples * 0.85)
    n_test = n_samples - n_train
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, n_test])

    dataloader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=4)

    for i in range(n_test):
        sample = testset[i]
        original_idx = testset.indices[i]
        info = dataset.get_sample_info(original_idx)
        info2 = dataset_test.get_sample_info(original_idx)
        sample2 = dataset_test[original_idx]
        print(i, original_idx, sample['disaster'], sample['image_id'], sample['post_img'].shape)
        print(i, original_idx, sample2['disaster'], sample2['image_id'], sample2['post_img'].shape)
        print(i, original_idx, info['disaster'], info['image_id'])
        print(i, original_idx, info2['disaster'], info2['image_id'])

    for i, samples in enumerate(dataloader):
        print(i, samples['disaster'], samples['image_id'], samples['post_img'].shape)
