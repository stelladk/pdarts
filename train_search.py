import os
import sys
import time
import glob
from datetime import datetime
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from model import NetworkCIFAR
from genotypes import PRIMITIVES
from genotypes import Genotype
import genotypes as gt_module


if "/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/" not in sys.path:
    sys.path.append("/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/")
if "/home/tau/sdouka/codebase/experimental_grow" not in sys.path:
    sys.path.append("/home/tau/sdouka/codebase/experimental_grow")

from tools.datasets import known_datasets, get_num_classes
from tools.augmentations import default_augmentations, get_transforms, npy_datasets
from logger import Logger


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--data', type=str, default='/scratch/sdouka/data', help='dataset root directory')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--dataset', type=str, default=None,
                    help='custom dataset name (e.g. addnist, multnist, cifartile, geoclassing, '
                         'chesseract, gameoflife, gutenberg, language); '
                         'overrides --cifar100 when set')
parser.add_argument('--no-augment', action='store_true', default=False,
                    help='disable data augmentation')
parser.add_argument('--experiment_name', type=str, default='NAS',
                    help='experiment name for the logger')
parser.add_argument('--no-logger', action='store_true', default=False,
                    help='disable experiment logger')
parser.add_argument('--logger_api', type=str, default='wandb',
                    choices=['mlflow', 'wandb'], help='logging backend')
parser.add_argument('--logger_port', type=int, default=27027,
                    help='port for the local logging server')
parser.add_argument('--log_path', type=str, default=None,
                    help='local directory for logger storage (mlflow tracking URI / wandb dir)')
parser.add_argument("--tmpdir", type=str, default="tmp")
# Evaluation phase arguments (mirrors train_cifar.py defaults)
parser.add_argument('--eval_epochs', type=int, default=600, help='eval: num of training epochs')
parser.add_argument('--eval_init_channels', type=int, default=36, help='eval: num of init channels')
parser.add_argument('--eval_layers', type=int, default=20, help='eval: total number of layers')
parser.add_argument('--eval_auxiliary', action='store_true', default=False, help='eval: use auxiliary tower')
parser.add_argument('--eval_auxiliary_weight', type=float, default=0.4, help='eval: weight for auxiliary loss')
parser.add_argument('--eval_drop_path_prob', type=float, default=0.3, help='eval: drop path probability')
parser.add_argument('--eval_learning_rate', type=float, default=0.025, help='eval: init learning rate')
parser.add_argument('--eval_batch_size', type=int, default=128, help='eval: batch size')
parser.add_argument('--init_genotype', type=str, default=None,
                    help='name of a genotype in genotypes.py to warm-start arch parameters '
                         '(e.g. PDARTS, DARTS_V2). Switches remain fully open; only alphas '
                         'are biased toward the given genotype at stage 0.')
parser.add_argument('--resume', type=str, default=None,
                    help='path to a checkpoint.pt (search or eval phase) to resume an interrupted '
                         'run from; continues in that checkpoint\'s experiment directory and log '
                         'file. The checkpoint records which phase it belongs to, so this works '
                         'whether the run was interrupted during search or during the post-search '
                         'evaluation training.')

args = parser.parse_args()

if args.resume is not None:
    # Continue in the same experiment directory so logs and the checkpoint stay together.
    args.save = os.path.dirname(os.path.abspath(args.resume))
    utils.create_exp_dir(args.save)
else:
    # Millisecond precision avoids two runs launched in the same second colliding on
    # the same experiment directory (e.g. an array of cluster jobs starting together).
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    args.save = '{}search-{}-{}'.format(args.save, args.note, timestamp)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

_dataset_name = 'cifar100' if args.cifar100 else (args.dataset or 'cifar10')
CIFAR_CLASSES = get_num_classes(_dataset_name)

def warm_start_alphas(model, genotype, bias=5.0):
    """Bias arch parameters toward the ops in genotype at the start of search.

    For each edge present in the genotype, the corresponding alpha column is
    set to `bias`; all other columns on that edge are zeroed.  Edges not
    mentioned in the genotype are left at their random initialisation so the
    search can still explore freely.

    Args:
        model: the search Network (after DataParallel wrapping).
        genotype: a Genotype namedtuple.
        bias: the value written to the favoured alpha column (default 5.0).
    """
    # Cumulative start index for each of the 4 intermediate nodes.
    # Node i has (2+i) input edges; starts = [0, 2, 5, 9].
    step_starts = [sum(2 + j for j in range(i)) for i in range(4)]

    def _apply(alphas, gene, switches):
        with torch.no_grad():
            for pair_idx, (op_name, input_node) in enumerate(gene):
                step = pair_idx // 2          # 2 ops retained per node
                edge = step_starts[step] + input_node
                enabled_ops = [j for j in range(len(PRIMITIVES)) if switches[edge][j]]
                if not enabled_ops:
                    continue
                op_global_idx = PRIMITIVES.index(op_name)
                if op_global_idx not in enabled_ops:
                    logging.warning(
                        'init_genotype: op "%s" not enabled on edge %d — skipping',
                        op_name, edge)
                    continue
                col = enabled_ops.index(op_global_idx)
                alphas.data[edge].zero_()
                alphas.data[edge][col] = bias

    # switches at stage-0 are all-True, so every op maps to a valid column
    all_true = [[True] * len(PRIMITIVES) for _ in range(14)]
    _apply(model.module.alphas_normal, genotype.normal, all_true)
    _apply(model.module.alphas_reduce, genotype.reduce, all_true)


# ──────────────────────────────────────────── checkpoint / resume ───── #
# Each phase writes to a single fixed filename that gets overwritten every
# epoch, so a long run never accumulates more than one checkpoint per phase.
def _save_search_checkpoint(path, sp, next_epoch, global_epoch, switches_normal, switches_reduce,
                             model, optimizer, optimizer_a, scheduler):
    torch.save({
        'phase': 'search',
        'sp': sp,
        'epoch': next_epoch,
        'global_epoch': global_epoch,
        'switches_normal': switches_normal,
        'switches_reduce': switches_reduce,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'optimizer_a_state': optimizer_a.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }, path)


def _load_search_checkpoint(ckpt, model, optimizer, optimizer_a, scheduler):
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    optimizer_a.load_state_dict(ckpt['optimizer_a_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])


def _save_eval_checkpoint(path, next_epoch, genotype, best_acc, model, optimizer, scheduler):
    torch.save({
        'phase': 'eval',
        'epoch': next_epoch,
        'genotype': genotype,
        'best_acc': best_acc,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }, path)


def _load_eval_checkpoint(ckpt, model, optimizer, scheduler):
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    tracker = Logger(args.experiment_name, port=args.logger_port,
                     api=args.logger_api, enabled=not args.no_logger)
    tracker.setup_tracking(file_path=args.log_path)
    tracker.start_run(group="P-DARTS")
    for key, value in vars(args).items():
        if key in ("no_logger", "api", "exp_name", "port", "log_path", "tmpdir"):
                continue
        tracker.log_parameter(key, str(value))
    #  prepare dataset
    dataset_name = 'cifar100' if args.cifar100 else (args.dataset or 'cifar10')
    augmentations = None if args.no_augment else default_augmentations.get(dataset_name, [])
    base_transforms, aug_transforms = get_transforms(dataset_name, augmentations)
    if dataset_name in npy_datasets:
        train_transform = transforms.Compose(base_transforms + aug_transforms)
    else:
        train_transform = transforms.Compose(aug_transforms + base_transforms)
    valid_transform = transforms.Compose(base_transforms)
    dataset_cls = known_datasets[dataset_name]
    if dataset_name == 'svhn':
        train_data = dataset_cls(root=args.data, split='train',
                                 download=True, transform=train_transform)
        test_data = dataset_cls(root=args.data, split='test',
                                download=True, transform=valid_transform)
    else:
        train_data = dataset_cls(root=args.data, train=True,
                                 download=True, transform=train_transform)
        test_data = dataset_cls(root=args.data, train=False,
                                download=True, transform=valid_transform)

    # infer input channels from the dataset
    input_channels = train_data[0][0].shape[0]
    logging.info("input channels = %d", input_channels)

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    genotype = None
    if genotype is not None:
        run_evaluation(genotype, train_data, test_data, criterion, tracker, args, input_channels)
        tracker.end_run()
        return

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        shuffle=False, pin_memory=True, num_workers=args.workers)

    # infer input channels from the dataset
    input_channels = train_data[0][0].shape[0]
    logging.info("input channels = %d", input_channels)

    # ── resume ────────────────────────────────────────────────────────────
    # A single checkpoint file per phase, overwritten every epoch, so an
    # interrupted run never leaves behind more than one file per phase.
    search_ckpt_path = os.path.join(args.save, 'checkpoint.pt')
    eval_ckpt_path = os.path.join(args.save, 'eval_checkpoint.pt')

    resume_ckpt = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location='cuda')
        logging.info('Resuming from %s (phase=%s, epoch=%d)',
                     args.resume, resume_ckpt['phase'], resume_ckpt['epoch'])

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
    eps_no_archs = [10, 10, 10]
    global_epoch = 0
    genotype = None

    # Resolve init_genotype once before the stage loop
    init_genotype = None
    if args.init_genotype is not None:
        if not hasattr(gt_module, args.init_genotype):
            logging.error('init_genotype "%s" not found in genotypes.py', args.init_genotype)
            sys.exit(1)
        init_genotype = getattr(gt_module, args.init_genotype)
        logging.info('Warm-starting arch parameters from genotype: %s', args.init_genotype)

    run_search = True
    start_sp = 0
    start_epoch = 0
    if resume_ckpt is not None and resume_ckpt['phase'] == 'eval':
        # Search already finished in the interrupted run; skip straight to
        # resuming the evaluation-training phase with its saved genotype.
        run_search = False
        start_sp = len(num_to_keep)
        genotype = resume_ckpt['genotype']
        logging.info('Skipping search phase; resuming eval phase with genotype: %s', genotype)
    elif resume_ckpt is not None:
        start_sp = resume_ckpt['sp']
        start_epoch = resume_ckpt['epoch']
        global_epoch = resume_ckpt['global_epoch']
        switches_normal = resume_ckpt['switches_normal']
        switches_reduce = resume_ckpt['switches_reduce']

    for sp in range(start_sp, len(num_to_keep)):
        model = Network(args.init_channels + int(add_width[sp]), CIFAR_CLASSES, args.layers + int(add_layers[sp]), criterion, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_rate[sp]), C_in=input_channels)
        model = nn.DataParallel(model)
        model = model.cuda()
        if sp == 0 and init_genotype is not None and args.resume is None:
            warm_start_alphas(model, init_genotype)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tracker.log_metric('training/nb of parameters', n_params, step=global_epoch, step_name='search epoch')

        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)
        optimizer = torch.optim.SGD(
                network_params,
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        if resume_ckpt is not None and resume_ckpt['phase'] == 'search' and sp == start_sp:
            _load_search_checkpoint(resume_ckpt, model, optimizer, optimizer_a, scheduler)
            logging.info('Restored model/optimizer/scheduler state for stage %d, epoch %d', sp, start_epoch)
            resume_ckpt = None

        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        epoch_range_start = start_epoch if sp == start_sp else 0
        for epoch in range(epoch_range_start, epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
                model.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.module.update_p()
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=False)
            else:
                model.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model.module.update_p()
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True)
            logging.info('Train_acc %.4f Train_loss %e', train_acc / 100., train_obj)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('Valid_acc %.4f Valid_loss %e', valid_acc / 100., valid_obj)
            test_acc, test_obj = infer(test_queue, model, criterion)
            logging.info('Test_acc %.4f Test_loss %e', test_acc / 100., test_obj)
            tracker.log_metrics({
                'search/train accuracy': train_acc / 100., 'search/train loss': train_obj,
                'search/val accuracy': valid_acc / 100., 'search/val loss': valid_obj,
                'search/test accuracy': test_acc / 100., 'search/test loss': test_obj,
            }, step=global_epoch, step_name='search epoch')
            global_epoch += 1
            _save_search_checkpoint(search_ckpt_path, sp, epoch + 1, global_epoch,
                                     switches_normal, switches_reduce,
                                     model, optimizer, optimizer_a, scheduler)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement.
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce)

        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk = check_sk_number(switches_normal)
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)

    if run_search:
        # Free search-phase memory before evaluation
        del model, optimizer, optimizer_a, scheduler
        del train_queue, valid_queue
        del arch_param, normal_prob, reduce_prob
        del switches_normal, switches_reduce, switches_normal_2, switches_reduce_2
        del network_params, normal_final, reduce_final
        torch.cuda.empty_cache()

    # Train and evaluate the discovered architecture
    if genotype is None:
        logging.error('Genotype was not found; skipping evaluation phase.')
        tracker.end_run()
        return
    eval_resume_ckpt = resume_ckpt if resume_ckpt is not None and resume_ckpt['phase'] == 'eval' else None
    run_evaluation(genotype, train_data, test_data, criterion, tracker, args, input_channels,
                    checkpoint_path=eval_ckpt_path, resume_ckpt=eval_resume_ckpt)
    tracker.end_run()


def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    # Fix: initialize iterator before the loop to avoid NameError on first access
    valid_queue_iter = iter(valid_queue)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above. 
            try:
                input_search, target_search = next(valid_queue_iter)
            except StopIteration:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def train_eval(train_queue, model, criterion, optimizer):
    """Train one epoch of the evaluation model (NetworkCIFAR). Mirrors train_cifar.py."""
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.eval_auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.eval_auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Eval TRAIN Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer_eval(test_queue, model, criterion):
    """Evaluate the evaluation model on the test set. Mirrors train_cifar.py."""
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Eval TEST Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def run_evaluation(genotype, train_data, test_data, criterion, tracker, args, input_channels=3,
                    checkpoint_path=None, resume_ckpt=None):
    """Train and evaluate the discovered architecture (NetworkCIFAR).

    Follows train_cifar.py as closely as possible:
    - Full training set (no split) for training
    - Test set evaluated every epoch
    - Cosine annealing LR, drop path, optional auxiliary loss
    """
    logging.info('=== Starting Evaluation Phase ===')
    logging.info('Genotype: %s', genotype)

    eval_model = NetworkCIFAR(
        args.eval_init_channels, CIFAR_CLASSES, args.eval_layers,
        args.eval_auxiliary, genotype, C_in=input_channels
    )
    eval_model = nn.DataParallel(eval_model)
    eval_model = eval_model.cuda()
    # Initialize drop_path_prob before any forward pass
    eval_model.module.drop_path_prob = 0.0

    n_params_mb = utils.count_parameters_in_MB(eval_model)
    n_params = sum(p.numel() for p in eval_model.parameters() if p.requires_grad)
    logging.info('Eval model param size = %.4fMB (%d params)', n_params_mb, n_params)
    tracker.log_metric('training/nb of parameters', n_params, step=0, step_name='epoch')

    # Full training set — no train/valid split for the evaluation phase
    eval_train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.eval_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.workers)
    eval_test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size,
        shuffle=False, pin_memory=True, num_workers=args.workers)

    optimizer = torch.optim.SGD(
        eval_model.parameters(),
        args.eval_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.eval_epochs))

    start_epoch = 0
    best_acc = 0.0
    if resume_ckpt is not None:
        _load_eval_checkpoint(resume_ckpt, eval_model, optimizer, scheduler)
        start_epoch = resume_ckpt['epoch']
        best_acc = resume_ckpt.get('best_acc', 0.0)
        logging.info('Resumed eval phase from epoch %d', start_epoch)

    for epoch in range(start_epoch, args.eval_epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('Eval Epoch: %d lr: %e', epoch, lr)
        eval_model.module.drop_path_prob = args.eval_drop_path_prob * epoch / args.eval_epochs

        epoch_start = time.time()
        train_acc, train_obj = train_eval(eval_train_queue, eval_model, criterion, optimizer)
        logging.info('Eval Train_acc %.4f Train_loss %e', train_acc / 100., train_obj)

        test_acc, test_obj = infer_eval(eval_test_queue, eval_model, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
        logging.info('Eval Test_acc  %.4f Test_loss  %e', test_acc / 100., test_obj)
        logging.info('Eval Epoch time: %ds', time.time() - epoch_start)

        tracker.log_metrics({
            'training/train accuracy': train_acc / 100.,
            'training/train loss': train_obj,
            'training/test accuracy': test_acc / 100.,
            'training/test loss': test_obj,
        }, step=epoch, step_name='epoch')

        utils.save(eval_model, os.path.join(args.save, 'eval_weights.pt'))
        if checkpoint_path is not None:
            _save_eval_checkpoint(checkpoint_path, epoch + 1, genotype, best_acc,
                                   eval_model, optimizer, scheduler)

    logging.info('Eval best test accuracy: %.4f', best_acc / 100.)

    # Log final trained evaluation model
    final_params = sum(p.numel() for p in eval_model.parameters() if p.requires_grad)
    logging.info('Final eval model param count = %d', final_params)

    tracker.log_pytorch_model(
        model=eval_model,
        name=f"DARTS_{_dataset_name}",
        x=None,
        path=args.tmpdir,
        run_id=False,
    )
    return eval_model


def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index

def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index

def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)

def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out

def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches

def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
