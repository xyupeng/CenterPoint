import argparse
import json
import os
import sys

# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
# import warnings
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
import torch.distributed as dist
import subprocess
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    return cfg


def build_ds(cfg):
    ds = build_dataset(cfg.data.train)  # len(train_set) == 158081
    print('==> Dataset built.')
    # sample = ds[0]

    '''
    # -----------------------------------------
    # ds._waymo_infos
    # -----------------------------------------
    assert len(ds._waymo_infos) == len(ds)  # 158081 training samples/frames
    x0 = ds._waymo_infos[0]
    x0 = {
        'path': 'data/Waymo/train/lidar/seq_0_frame_0.pkl',
        'anno_path': 'data/Waymo/train/annos/seq_0_frame_0.pkl',
        'token': 'seq_0_frame_0.pkl',
        'timestamp': 1550083467.34637,
        'sweeps': [],
        'gt_boxes': ndarray(shape=(num_boxes, 9), dtype=float32),
        'gt_names': ndarray(shape=(num_boxes, ), dtype=str)
    }
    '''

    '''
    # -----------------------------------------
    # first 2:
    # dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    # dict(type="LoadPointCloudAnnotations", with_bbox=True),
    # -----------------------------------------
    sample = ds[0]
    sample = {
      'lidar' = {
          'type': 'lidar',
          'points': ndarray(num_pts, 5),
          'annotations': {
              'boxes': ndarray(num_boxes, 9)
              'names': ndarray(num_boxes, ), dtype=str (class name)
          }
          'nsweeps': 1
      },
      'metadata': {'image_prefix', 'num_point_features', 'token'},
      'calib': None,
      'cam': {},
      'mode': 'train',
      'type': 'WaymoDataset'
    }
    '''

    '''
    # -----------------------------------------
    # first 3:
    # dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    # dict(type="LoadPointCloudAnnotations", with_bbox=True),
    # dict(type="Preprocess", cfg=train_preprocessor),
    # -----------------------------------------
    sample = ds[0]
    sample = {
      'lidar' = {
          'type': 'lidar',
          'points': ndarray(num_pts, 5),  # num_pts=191407; raw points + sampled points
          'annotations': {
              'gt_boxes': ndarray(shape=(num_boxes, 9), dtype=float32),  # num_boxes=35
              'gt_names': ndarray(shape=(num_boxes, ), dtype=str),  # class name for each gt_box
              'gt_classes': ndarray(shape=(num_boxes, ), dtype=int32),  # class id for each gt_box, start from 1
          }
          'nsweeps': 1
      },
      'metadata': {'image_prefix', 'num_point_features', 'token'},
      'calib': None,
      'cam': {},
      'mode': 'train',
      'type': 'WaymoDataset'
    }
    '''

    # full pipeline
    # keys: ['metadata', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'coordinates',
    # 'gt_boxes_and_cls', 'hm', 'anno_box', 'ind', 'mask', 'cat']
    return ds


def build_dl(ds):
    # from det3d.datasets.loader import build_dataloader
    # dl = build_dataloader(ds, batch_size=3, workers_per_gpu=4, num_gpus=1, dist=False)
    from det3d.torchie.parallel import collate_kitti
    from torch.utils.data import DataLoader
    dl = DataLoader(
        ds,
        batch_size=3,
        sampler=None,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_kitti,
        # pin_memory=True,
        pin_memory=False,
    )
    print('==> Dataloader built.')

    '''
        x = next(iter(dl))
        x keys: ['metadata', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'coordinates', 'gt_boxes_and_cls',
            'hm', 'anno_box', 'ind', 'mask', 'cat']
        'points': list of tensor(shape=(num_pts, 5))
        'voxel': tensor(shape=(tot_voxels, 20, 5))
        'shape': array([[468, 468, 1], [468, 468, 1], [468, 468, 1]])
        'num_points': tensor(shape=(tot_voxels,))  # num_pts in each voxel
        'num_voxels': tensor(shape=(3,))  # number of voxels for each sample
        'coordinates': tensor(shape=(total_voxels, 4))  # [sample_id, x, y, z]; TODO: check xyz format
        'gt_boxes_and_cls': tensor(shape=(3, 500, 10))
        'hm': [tensor(shape=(3, 3, 468, 468))]
        'anno_box': [tensor(shape=(3, 500, 10))]
        'ind': [tensor(shape=(3, 500))]
        'mask': [tensor(shape=(3, 500))]
        'cat': [tensor(shape=(3, 500))]
    '''
    return dl


def main():
    args = parse_args()
    cfg = get_cfg(args)
    ds = build_ds(cfg)

    # dl = build_dl(ds)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    set_seed(42)
    main()
