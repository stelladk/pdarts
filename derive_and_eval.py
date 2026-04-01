"""
Derive the genotype from a saved search checkpoint (weights.pt from sp=0)
and evaluate the search model on the test set.

Usage:
    python derive_and_eval.py \
        --weights /scratch/sdouka/tmp/pdarts/checkpoints/search-try-20260331-125718/weights.pt \
        --data    /scratch/sdouka/data
"""

import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

if "/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/" not in sys.path:
    sys.path.append("/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/")
if "/home/tau/sdouka/codebase/experimental_grow" not in sys.path:
    sys.path.append("/home/tau/sdouka/codebase/experimental_grow")

from genotypes import Genotype, PRIMITIVES
from model_search import Network
from logger import Logger
import utils
from tools.datasets import known_datasets, get_transforms, get_num_classes

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--weights',  required=True, help='path to weights.pt')
parser.add_argument('--data',     required=True, help='dataset root')
parser.add_argument('--dataset',  default='multnist')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--gpu',      type=int, default=0)
# sp=0 search model config — must match the run that produced weights.pt
parser.add_argument('--channels', type=int, default=16)
parser.add_argument('--layers',   type=int, default=5)
parser.add_argument('--experiment_name', type=str, default='NAS')
parser.add_argument('--no-logger', action='store_true', default=False)
parser.add_argument('--logger_api', type=str, default='wandb', choices=['mlflow', 'wandb'])
parser.add_argument('--logger_port', type=int, default=27027)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--search_epoch', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

tracker = Logger(args.experiment_name, port=args.logger_port,
                 api=args.logger_api, enabled=not args.no_logger)
tracker.setup_tracking(file_path=args.log_path)
tracker.start_run(group="P-DARTS")

# ── helpers ───────────────────────────────────────────────────────────────────
def derive_genotype(alphas_normal, alphas_reduce):
    """DARTS-style top-2-edges-per-node derivation, excluding 'none'."""
    NONE_IDX = 0  # PRIMITIVES[0] == 'none'

    def _parse(alphas):
        probs = F.softmax(torch.tensor(alphas), dim=-1).numpy()
        gene, n, start = [], 2, 0
        for _ in range(4):          # 4 intermediate nodes
            end = start + n
            candidates = []
            for edge in range(start, end):
                p = probs[edge].copy()
                p[NONE_IDX] = 0.0   # ignore 'none'
                best_op = int(np.argmax(p))
                candidates.append((p[best_op], best_op, edge - start))
            candidates.sort(reverse=True)
            for _, op_idx, input_node in candidates[:2]:
                gene.append((PRIMITIVES[op_idx], input_node))
            start, n = end, n + 1
        return gene

    return Genotype(
        normal=_parse(alphas_normal), normal_concat=range(2, 6),
        reduce=_parse(alphas_reduce), reduce_concat=range(2, 6),
    )

base_transforms, _ = get_transforms(args.dataset, [])
test_transform = transforms.Compose(base_transforms)
dataset_cls    = known_datasets[args.dataset]
test_data      = dataset_cls(root=args.data, train=False, download=True,
                             transform=test_transform)
test_queue     = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size,
    shuffle=False, pin_memory=False, num_workers=2,
)

num_classes   = get_num_classes(args.dataset)
input_channels = test_data[0][0].shape[0]   # inferred from dataset; multnist = 3

criterion      = torch.nn.CrossEntropyLoss()
switches_true  = [[True] * len(PRIMITIVES) for _ in range(14)]
search_model   = Network(
    args.channels, num_classes, args.layers, criterion,
    switches_normal=switches_true,
    switches_reduce=copy.deepcopy(switches_true),
    p=0.0, C_in=input_channels,
)

# ── 1. load search model weights ──────────────────────────────────────────────
state = torch.load(args.weights, map_location='cpu')
# strip DataParallel 'module.' prefix if present
state = {k.replace('module.', ''): v for k, v in state.items()}
search_model.load_state_dict(state)
search_model = search_model.cuda()
search_model.eval()

print(f"Loaded search model  |  channels={args.channels}  layers={args.layers}")
for key, value in vars(args).items():
    tracker.log_parameter(key, str(value))

# ── 2. derive genotype ────────────────────────────────────────────────────────
alphas_n = search_model.alphas_normal.detach().cpu().numpy()
alphas_r = search_model.alphas_reduce.detach().cpu().numpy()

genotype = derive_genotype(alphas_n, alphas_r)
print("\nDerived genotype:")
print(genotype)
tracker.log_parameter('genotype', str(genotype))

# ── 3. evaluate search model on test set ─────────────────────────────────────
criterion_cuda = torch.nn.CrossEntropyLoss().cuda()
top1 = utils.AvgrageMeter()
objs = utils.AvgrageMeter()

with torch.no_grad():
    for step, (inputs, targets) in enumerate(test_queue):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = search_model(inputs)          # search model returns logits directly
        loss   = criterion_cuda(logits, targets)
        prec1, _ = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        if step % 50 == 0:
            print(f"test {step:03d}  loss={objs.avg:.4f}  acc={top1.avg:.2f}%")

print(f"\nSearch-model test accuracy:  {top1.avg:.2f}%  (loss {objs.avg:.4f})")
tracker.log_metrics({"training/test accuracy": top1.avg / 100., "training/test loss": objs.avg}, step=args.search_epoch, step_name="search epoch")
tracker.end_run()
print("\nTo train the eval model from scratch, add this genotype to genotypes.py")
print("and run train_cifar.py with --arch <name>.")
