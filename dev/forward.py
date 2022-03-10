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


def main():
    args = parse_args()
    cfg = get_cfg(args)

    # build dataset & dataloader
    ds = build_dataset(cfg.data.train)
    print('==> Dataset built.')

    from det3d.torchie.parallel import collate_kitti
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        ds,
        batch_size=3,
        sampler=None,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_kitti,
        pin_memory=False,
    )
    print('==> Dataloader built.')

    # build model
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.CLASSES = ds.CLASSES
    model = model.cuda()

    # train
    from det3d.torchie.trainer.trainer import example_to_device, parse_second_losses
    for i, data_batch in enumerate(train_loader):
        example = example_to_device(
            data_batch, torch.cuda.current_device(), non_blocking=False
        )

        losses = model(example, return_loss=True)
        loss, log_vars = parse_second_losses(losses)
        del losses

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=-1
        )
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
