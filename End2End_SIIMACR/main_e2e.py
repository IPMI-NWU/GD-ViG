import datetime
import argparse
import os
import random
import time
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from End2End_SIIMACR.dataset_e2e import DatasetGaze_e2e
from End2End_SIIMACR.engines_e2e import train_one_epoch_e2e
from End2End_SIIMACR.models.GDViG import GDViG
from utils import misc
from inference_e2e import infer_e2e


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--output_dir', default='output/exp_1/', help='path where to save, empty for no saving')
    parser.add_argument('--train_dir', default='../Data/SIIM-ACR-Gaze/train/')
    parser.add_argument('--csv_path', default='../Data/SIIM-ACR-Gaze/siim_pneumothorax.csv')
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    return parser


def main(args):
    if not args.eval:
        writer = SummaryWriter(log_dir=args.output_dir + '/summary')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)

    model = GDViG(num_classes=2)
    # model = ViG_ViGUNet_GNNGNN(num_classes=2)

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print('Building training dataset...')
    dataset_train = DatasetGaze_e2e(args.train_dir, args.csv_path, 'train', args.size)
    print('Number of training images: {}'.format(len(dataset_train)))

    print('Building validation dataset...')
    dataset_val = DatasetGaze_e2e(args.val_dir, args.csv_path, 'test', args.size)
    print('Number of validation images: {}'.format(len(dataset_val)))

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)

    if args.eval:
        infer_e2e()
        return

    print("Start training")
    start_time = time.time()

    for epoch in range(0, args.epochs):
        print('-' * 40)
        train_one_epoch_e2e(model, dataloader_train, optimizer, device, epoch, args, writer)

        # lr_scheduler
        lr_scheduler.step()

        # save checkpoint
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            misc.save_on_master({
                'model': model_without_ddp.state_dict(),
            }, checkpoint_path)
        print()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classification training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
