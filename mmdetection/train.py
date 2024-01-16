# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from notification import notification

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--kfold',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--kfold_config',
        help='train config folder path for kfold training'
    )
    parser.add_argument(
        '--epoch',
        type = int,
        default = 12,
        help='max epoch'
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()
    
    root='/data/ephemeral/home/data/dataset/'
    
   
    ## kfold 제대로 구현 안됨!! 코드 다시 확인 해야함 
    if args.kfold:
        cfg_list = os.listdir(args.kfold_config)
        cfg_list.sort()
        for i, cfg_path in enumerate(cfg_list):
            cfg = Config.fromfile(os.path.join(args.kfold_config, cfg_path))
            cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs= args.epoch * (i+1), val_interval=1)
            if cfg.train_pipeline[2]['type'] == 'Resize':
                cfg.train_pipeline[2]['scale'] = (512,512)
            cfg.test_pipeline[1]['scale'] = (512,512)
            cfg.default_hooks = dict(
                early_stopping=dict(
                    type="EarlyStoppingHook",
                    monitor="coco/bbox_mAP",
                    patience=10,
                    min_delta=0.005),
                checkpoint=dict(
                    type="CheckpointHook",
                    save_best="coco/bbox_mAP",
                    rule="greater"
                )
            )
            if 'roi_head' in cfg.model:
                cfg.model.roi_head.bbox_head.num_classes = 10
            cfg.launcher = args.launcher
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)
            if args.work_dir is not None:
                # update configs according to CLI args if args.work_dir is not None
                cfg.work_dir = args.work_dir
            elif cfg.get('work_dir', None) is None:
                # use config filename as default work_dir if cfg.work_dir is None
                cfg.work_dir = osp.join('./work_dirs',
                                        osp.splitext(osp.basename(args.config))[0])
            if args.amp is True:
                cfg.optim_wrapper.type = 'AmpOptimWrapper'
                cfg.optim_wrapper.loss_scale = 'dynamic'
            # enable automatically scaling LR
            if args.auto_scale_lr:
                if 'auto_scale_lr' in cfg and \
                        'enable' in cfg.auto_scale_lr and \
                        'base_batch_size' in cfg.auto_scale_lr:
                    cfg.auto_scale_lr.enable = True
                else:
                    raise RuntimeError('Can not find "auto_scale_lr" or '
                                    '"auto_scale_lr.enable" or '
                                    '"auto_scale_lr.base_batch_size" in your'
                                    ' configuration file.')

            # resume is determined in this priority: resume from > auto_resume
            if args.resume == 'auto':
                cfg.resume = True
                cfg.load_from = None
            elif args.resume is not None:
                cfg.resume = True
                cfg.load_from = args.resume
            
            if i > 0:
                file_list = os.listdir(args.work_dir)
                ckp_list = [ckp for ckp in file_list if ckp.endswith(".pth")]
                ckp_list.sort(reverse=True)
                cfg.resume = True
                cfg.load_from = os.path.join(args.work_dir, ckp_list[0])



            # build the runner from config
            if 'runner_type' not in cfg:
                # build the default runner
                runner = Runner.from_cfg(cfg)
            else:
                # build customized runner from the registry
                # if 'runner_type' is set in the cfg
                runner = RUNNERS.build(cfg)

            # start training
            runner.train()      
              
    else:         
        cfg = Config.fromfile(args.config)
        if cfg.train_pipeline[2]['type'] == 'Resize':
            cfg.train_pipeline[2]['scale'] = (512,512)
        cfg.test_pipeline[1]['scale'] = (512,512)
        cfg.visualizer.vis_backends = [dict(type='TensorboardVisBackend')] # tensorboard용
        cfg.default_hooks = dict(
            early_stopping=dict(
                type="EarlyStoppingHook",
                monitor="coco/bbox_mAP",
                patience=10,
                min_delta=0.005),
            checkpoint=dict(
                type="CheckpointHook",
                save_best="coco/bbox_mAP",
                rule="greater"
                )
            )
        if 'roi_head' in cfg.model:
            cfg.model.roi_head.bbox_head.num_classes = 10
        cfg.launcher = args.launcher
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        
        cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=args.epoch, val_interval=1)

        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

        # enable automatic-mixed-precision training
        if args.amp is True:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

        # enable automatically scaling LR
        if args.auto_scale_lr:
            if 'auto_scale_lr' in cfg and \
                    'enable' in cfg.auto_scale_lr and \
                    'base_batch_size' in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError('Can not find "auto_scale_lr" or '
                                '"auto_scale_lr.enable" or '
                                '"auto_scale_lr.base_batch_size" in your'
                                ' configuration file.')

        # resume is determined in this priority: resume from > auto_resume
        if args.resume == 'auto':
            cfg.resume = True
            cfg.load_from = None
        elif args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()


if __name__ == '__main__':
    main()
    notification()
