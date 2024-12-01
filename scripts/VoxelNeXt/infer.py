import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

# 필요한 모듈 임포트
from pcdet.utils import common_utils

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.cuda()


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for inference')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for inference')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to use for inference')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')

    parser.add_argument('--save_dir', type=str, default='inference_results', help='directory to save inference results')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)

    return args, cfg

def inference(model, dataloader, args, output_dir, logger):
    model.eval()
    logger.info('*************** Start Inference ***************')

    dataset = dataloader.dataset
    class_names = dataset.class_names

    if args.infer_time:
        logger.info('Inference time measurement is enabled.')

    with torch.no_grad():
        for idx, batch_dict in enumerate(dataloader):
            load_data_to_gpu(batch_dict)

            start_time = time.time() if args.infer_time else None

            pred_dicts, _ = model(batch_dict)

            if args.infer_time:
                elapsed_time = time.time() - start_time
                logger.info(f'Frame {idx} inference time: {elapsed_time * 1000:.2f} ms')

            # 결과 저장
            for i in range(len(pred_dicts)):
                frame_id = batch_dict['frame_id'][i]
                pred_boxes = pred_dicts[i]['pred_boxes'].cpu().numpy()
                pred_scores = pred_dicts[i]['pred_scores'].cpu().numpy()
                pred_labels = pred_dicts[i]['pred_labels'].cpu().numpy()

                # 결과를 텍스트 파일로 저장하거나 원하는 방식으로 처리
                save_path = output_dir / f'{frame_id}.txt'
                with open(save_path, 'w') as f:
                    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                        line = ' '.join(map(str, box.tolist())) + f' {score} {label}\n'
                        f.write(line)

    logger.info('*************** Inference Done ***************')

def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist = True

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_inference_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
    logger.info(f'CUDA_VISIBLE_DEVICES={gpu_list}')

    for key, val in vars(args).items():
        logger.info(f'{key:16} {val}')
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist)
    model.cuda()

    inference(model, test_loader, args, output_dir, logger)

if __name__ == '__main__':
    main()
