"""
Derive the genotype from a saved search checkpoint and evaluate the search
model on the test set.

weights.pt is saved at the END of each search stage (overwriting the previous
one), so the switches active in that stage must be used to reconstruct the
model.

Pass --log to a slurm/training log and both the weights.pt path and the stage
are derived automatically:
  - save dir  →  parsed from the "args = Namespace(..., save='...', ...)" line
  - stage     →  number of "Dropping" lines completed (each stage saves then drops)

--weights and --stage can still be passed explicitly to override the log values.

Usage:
    # fully automatic from log:
    python derive_and_eval.py --log slurm-budget-pdarts_multnist-58330_1.out \\
        --search_epoch 46

    # override weights path:
    python derive_and_eval.py --log slurm-budget-pdarts_multnist-58330_1.out \\
        --weights /other/path/weights.pt --search_epoch 46
"""

import sys
import re
import ast
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
parser.add_argument('--log',      required=True, help='path to the slurm/training log file')
parser.add_argument('--data',     help='dataset root', default="/scratch/sdouka/data")
parser.add_argument('--weights',  default=None,
                    help='path to weights.pt; derived from --log if omitted')
parser.add_argument('--stage',    type=int, default=None,
                    help='which search stage produced weights.pt (0, 1, or 2); '
                         'derived from --log if omitted')
parser.add_argument('--dataset',  default='multnist')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--gpu',      type=int, default=0)
# search model config - must match the run that produced weights.pt
parser.add_argument('--channels', type=int, default=16)
parser.add_argument('--layers',   type=int, default=5)
parser.add_argument('--experiment_name', type=str, default='Budget')
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
def parse_log(log_path):
    """Parse save directory, last completed stage, and per-stage model config
    from a training log.

    weights.pt is saved once per stage before the "Dropping" print, so
    N "Dropping" lines means the last weights.pt is from stage N-1.

    Applies the same add_width/add_layers fallback logic as train_search.py:
      len != 3  →  add_width=[0,0,0], add_layers=[0,6,12]

    Returns (save_dir, stage, channels, layers) for the last saved stage.
    """
    save_dir = None
    init_channels, base_layers = 16, 5
    add_width_raw, add_layers_raw = ['0'], ['0']
    dropping_count = 0

    with open(log_path) as f:
        for line in f:
            if save_dir is None:
                m = re.search(r"args = Namespace\((.+)\)", line)
                if m:
                    ns = m.group(1)
                    for key, var in [('init_channels', 'init_channels'),
                                     ('layers',        'base_layers')]:
                        mv = re.search(rf'\b{key}=(\d+)', ns)
                        if mv:
                            locals()[var]   # just check it exists
                            if key == 'init_channels':
                                init_channels = int(mv.group(1))
                            else:
                                base_layers = int(mv.group(1))
                    for key, var in [('add_width', 'add_width_raw'),
                                     ('add_layers', 'add_layers_raw')]:
                        mv = re.search(rf"{key}=(\[[^\]]*\])", ns)
                        if mv:
                            if key == 'add_width':
                                add_width_raw = ast.literal_eval(mv.group(1))
                            else:
                                add_layers_raw = ast.literal_eval(mv.group(1))
                    mv = re.search(r"save='([^']+)'", ns)
                    if mv:
                        save_dir = mv.group(1)
            if '------Dropping' in line:
                dropping_count += 1

    if save_dir is None:
        raise ValueError(f"Could not find save= in log {log_path}")
    if dropping_count == 0:
        raise ValueError(f"No completed stage found in log {log_path}")

    # same fallback as train_search.py
    add_width  = [int(x) for x in add_width_raw]  if len(add_width_raw)  == 3 else [0, 0, 0]
    add_layers = [int(x) for x in add_layers_raw] if len(add_layers_raw) == 3 else [0, 6, 12]

    stage    = dropping_count - 1
    channels = init_channels + add_width[stage]
    layers   = base_layers   + add_layers[stage]
    return save_dir, stage, channels, layers


def parse_switches_from_log(log_path, occurrence):
    """Return (switches_normal, switches_reduce) for the given occurrence index
    (0-based) of the 'switches_normal = ...' / 'switches_reduce = ...' pairs
    logged after each drop step in train_search.py.

    occurrence=0 → after stage-0 dropping  → used in the stage-1 model
    occurrence=1 → after stage-1 dropping  → used in the stage-2 model
    """
    normals, reduces = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'switches_normal = (\[.*\])', line)
            if m:
                normals.append(ast.literal_eval(m.group(1)))
            m = re.search(r'switches_reduce = (\[.*\])', line)
            if m:
                reduces.append(ast.literal_eval(m.group(1)))

    if occurrence > len(normals) - 1:
        raise ValueError(
            f'Log contains {len(normals)} switches_normal occurrence(s) '
            f'but --stage {occurrence + 1} requires occurrence index {occurrence}. '
            f'The run may not have completed that stage.'
        )
    return normals[occurrence], reduces[occurrence]


def derive_genotype(alphas_normal, alphas_reduce, switches_normal, switches_reduce):
    """DARTS-style top-2-edges-per-node derivation, excluding 'none'.

    switches_normal/reduce map each alpha column back to the correct primitive:
    for edge i, the j-th column of alphas corresponds to the j-th True entry
    in switches[i].
    """
    def _parse(alphas, switches):
        probs = F.softmax(torch.tensor(alphas), dim=-1).numpy()
        gene, n, start = [], 2, 0
        for _ in range(4):          # 4 intermediate nodes
            end = start + n
            candidates = []
            for edge in range(start, end):
                active_ops = [j for j in range(len(PRIMITIVES)) if switches[edge][j]]
                p = probs[edge].copy()
                # zero out 'none' (PRIMITIVES[0]) if it is still active
                if 0 in active_ops:
                    p[active_ops.index(0)] = 0.0
                best_col = int(np.argmax(p))
                best_prim_idx = active_ops[best_col]
                candidates.append((p[best_col], best_prim_idx, edge - start))
            candidates.sort(reverse=True)
            for _, prim_idx, input_node in candidates[:2]:
                gene.append((PRIMITIVES[prim_idx], input_node))
            start, n = end, n + 1
        return gene

    return Genotype(
        normal=_parse(alphas_normal, switches_normal), normal_concat=range(2, 6),
        reduce=_parse(alphas_reduce, switches_reduce), reduce_concat=range(2, 6),
    )


# ── resolve weights path, stage, and model config from log ────────────────────
log_save_dir, log_stage, log_channels, log_layers = parse_log(args.log)
stage    = args.stage   if args.stage   is not None else log_stage
weights  = args.weights if args.weights is not None else f'{log_save_dir}/weights.pt'
channels = args.channels if args.channels != 16 else log_channels  # 16 = argparse default
layers   = args.layers   if args.layers   != 5  else log_layers    # 5  = argparse default
print(f'Log: {args.log}')
print(f'  save dir : {log_save_dir}')
print(f'  stage    : {stage}   {"(from log)" if args.stage is None else "(overridden)"}')
print(f'  weights  : {weights}  {"(from log)" if args.weights is None else "(overridden)"}')
print(f'  channels : {channels}  layers: {layers}  {"(from log)" if args.channels == 16 and args.layers == 5 else "(overridden)"}')

# ── resolve switches ──────────────────────────────────────────────────────────
if stage == 0:
    switches_normal = [[True] * len(PRIMITIVES) for _ in range(14)]
    switches_reduce = [[True] * len(PRIMITIVES) for _ in range(14)]
else:
    # occurrence index = stage - 1  (stage 1 uses the 1st logged switches, etc.)
    switches_normal, switches_reduce = parse_switches_from_log(args.log, stage - 1)
    print(f'  switches_normal[0] active ops: '
          f'{[PRIMITIVES[j] for j in range(len(PRIMITIVES)) if switches_normal[0][j]]}')
    print(f'  switches_reduce[0] active ops: '
          f'{[PRIMITIVES[j] for j in range(len(PRIMITIVES)) if switches_reduce[0][j]]}')

# ── build search model ─────────────────────────────────────────────────────────
base_transforms, _ = get_transforms(args.dataset, [])
test_transform = transforms.Compose(base_transforms)
dataset_cls    = known_datasets[args.dataset]
test_data      = dataset_cls(root=args.data, train=False, download=True,
                             transform=test_transform)
test_queue     = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size,
    shuffle=False, pin_memory=False, num_workers=2,
)

num_classes    = get_num_classes(args.dataset)
input_channels = test_data[0][0].shape[0]

criterion   = torch.nn.CrossEntropyLoss()
search_model = Network(
    channels, num_classes, layers, criterion,
    switches_normal=switches_normal,
    switches_reduce=copy.deepcopy(switches_reduce),
    p=0.0, C_in=input_channels,
)

# ── 1. load search model weights ──────────────────────────────────────────────
state = torch.load(weights, map_location='cpu')
# strip DataParallel 'module.' prefix if present
state = {k.replace('module.', ''): v for k, v in state.items()}

# diagnose shape mismatches before they crash load_state_dict
model_state = search_model.state_dict()
mismatches = [
    (k, state[k].shape, model_state[k].shape)
    for k in state if k in model_state and state[k].shape != model_state[k].shape
]
if mismatches:
    print("Shape mismatches between checkpoint and model:")
    for k, ckpt_shape, model_shape in mismatches:
        print(f"  {k}: checkpoint {ckpt_shape}  vs  model {model_shape}")
    raise RuntimeError(
        "Cannot load checkpoint. "
        "Check --channels / --layers match the run that produced weights.pt, "
        "and that the --stage / --log switches are correct."
    )

search_model.load_state_dict(state)
search_model = search_model.cuda()
search_model.eval()

print(f"Loaded search model  |  channels={channels}  layers={layers}  "
      f"stage={args.stage}  ops_per_edge={search_model.switch_on}")
for key, value in vars(args).items():
    tracker.log_parameter(key, str(value))

# ── 2. derive genotype ────────────────────────────────────────────────────────
alphas_n = search_model.alphas_normal.detach().cpu().numpy()
alphas_r = search_model.alphas_reduce.detach().cpu().numpy()

genotype = derive_genotype(alphas_n, alphas_r, switches_normal, switches_reduce)
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
        logits = search_model(inputs)
        loss   = criterion_cuda(logits, targets)
        prec1, _ = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        if step % 50 == 0:
            print(f"test {step:03d}  loss={objs.avg:.4f}  acc={top1.avg:.2f}%")

print(f"\nSearch-model test accuracy:  {top1.avg:.2f}%  (loss {objs.avg:.4f})")
tracker.log_metrics({"training/test accuracy": top1.avg / 100., "training/test loss": objs.avg}, step=args.search_epoch, step_name="search epoch")

n_params = sum(p.numel() for p in search_model.parameters() if p.requires_grad)
tracker.log_metric('training/nb of parameters', n_params, step=args.search_epoch, step_name="search epoch")

tracker.end_run()
print("\nTo train the eval model from scratch, add this genotype to genotypes.py")
print("and run train_cifar.py with --arch <name>.")
