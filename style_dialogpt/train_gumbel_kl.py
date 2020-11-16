'''
Modified based on Huggingface GPT-2 and DialoGPT implementations.
'''

import json
import os
import sys
import argparse
import logging
import time
from collections import OrderedDict

import datetime
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

import numpy as np
from tqdm import tqdm
from os.path import join
from torch.distributed import get_rank, get_world_size
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, "models"))
from models import GPT2LMHeadModel, Discriminator

from utils import load_model, boolean_string, all_reduce_and_rescale_tensors
from data_loader import BucketDataLoader
from losses import compute_kl
from evaluate import evaluate
from utils import gumbel_sequence_sample


from tensorboardX import SummaryWriter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000  # todo: change

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=50)

parser.add_argument("--skip_eval", action='store_true', help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=4, help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="to increase effective batch size "
                                                                               "and reduce synchronization")
parser.add_argument("--do_eval", type=boolean_string, default=True)
parser.add_argument("--eval_batch_size", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_optim_steps", type=int, default=120_000, help="new API specifies num update steps")
parser.add_argument("--logging_step", type=int, default=50)
parser.add_argument("--valid_step", type=int, default=1500, help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=0)

parser.add_argument("--fp16", type=boolean_string, default=True)
parser.add_argument("--lr_schedule", type=str, choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--log_dir", type=str)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

# distributed
parser.add_argument('--local_rank', type=int, default=-1, help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')

parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument("--beam_num", type=int, default=None)
parser.add_argument("--return_num", type=int, default=50)

parser.add_argument('--sty_lm_model_name_or_path', type=str)
parser.add_argument('--sty_dic_model_fi', type=str)
parser.add_argument('--kl_scale', type=float, default=0.005)
parser.add_argument('--dis_scale', type=float, default=0.1)
parser.add_argument('--ce_scale', type=float, default=1)
parser.add_argument('--gumbel_temp', type=float, default=0.1, help='temperature of gumbel softmax.')

parser.add_argument('--opt_head', action='store_true', help='optimize lm head parameters.')
args = parser.parse_args()

if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            # PHILLY ENV special cases
            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY',
                              os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY',
                              os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)

assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
    'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size
                         // args.gradient_accumulation_steps)
logger.info('train batch size = {}, new train batch size (after gradient accumulation) = {}'.format(
    args.train_batch_size * args.gradient_accumulation_steps, args.train_batch_size))

if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}"
                .format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(args.output_dir, 'GPT2_lr_{}_kl_{}_dis_{}.{}'.format(args.learning_rate,
                                                                       args.kl_scale, args.dis_scale,
                                                                       timestamp))
args.output_dir = output_dir

log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(join(output_dir, 'train.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -  %(message)s'))
    logging.getLogger().addHandler(file_handler)

logger.info('Input Argument Information')
args_dict = {k: str(v) for k, v in vars(args).items()}
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))
with open(join(output_dir, 'args.json'), 'w', encoding='utf8')as p:
    json.dump(args_dict, p, indent=2)

#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
num = enc.add_special_tokens({'pad_token': '@'})
assert num == 0

config = GPT2Config.from_json_file(join(args.model_name_or_path, 'config.json'))

train_dataloader = BucketDataLoader(
    args.train_input_file, args.train_batch_size, args.max_seq_length,
    enc, False, True, rank=args.local_rank, world_size=get_world_size() if args.local_rank != -1 else -1
)
logger.info(f"{len(train_dataloader)} batches for one epoch.")

#########################################################################
# Prepare Model and Optimizer
##########################################################################
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint, args, verbose=True)

if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(params, float(torch.distributed.get_world_size()))

'''optimize all parameters or only lm head.'''
if hasattr(model, 'lm_head') and args.opt_head:
    for n, p in model.named_parameters():
        p.requires_grad = False
    param_optimizer = model.lm_head.named_parameters()
    for n, p in param_optimizer:
        p.requires_grad = True
    logger.info("Just fine tune the lm_head parameters.")
else:
    param_optimizer = model.named_parameters()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

# param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']  # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_optim_steps
)

#########################################################################
# prepare the style language Model and discriminator.
#########################################################################
style_config = GPT2Config.from_pretrained(args.sty_lm_model_name_or_path)
style_lm_model = GPT2LMHeadModel.from_pretrained(args.sty_lm_model_name_or_path, config=style_config)
style_dis_model = Discriminator.from_pretrained(args.sty_dic_model_fi, encoder=style_lm_model.transformer,
                                                device=device)
style_lm_model.to(device)
style_dis_model.to(device)
style_lm_model.eval()
style_dis_model.eval()
for p in list(style_lm_model.parameters()) + list(style_dis_model.parameters()):
    p.requires_grad = False

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    [model, style_lm_model, style_dis_model], optimizer = \
        amp.initialize([model, style_lm_model, style_dis_model], optimizer)

#########################################################################
# Training !
##########################################################################

if args.local_rank == -1 or get_rank() == 0:
    tb_writer = SummaryWriter(log_dir)

global_step = 0
step = 0
epoch = 0

disable_tqdm = 'PT_DATA_DIR' in os.environ.keys() or args.local_rank not in [-1, 0]

if args.continue_from:
    global_step = args.continue_from
    step = global_step * 2 - 1

if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar and not disable_tqdm:
        pbar = tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None

while True:
    model.train()
    tr_loss, tr_ppl, tr_kl, tr_dis = 0, 0, 0, 0
    tr_dis_loss, dis_loss = 0, 0
    gum_rep_lengths = []
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        context_ids, response_ids = batch
        context_len = context_ids.size(1)  # <|endoftext|> contained.

        loss = 0
        if args.dis_scale > 0:
            rep_logits, gumbel_weights, rep_mask = gumbel_sequence_sample(model, context_ids, args.gumbel_temp, 10, enc)

            gum_rep_len = rep_mask.sum(dim=1).float().mean().item()
            gum_rep_lengths.append(gum_rep_len)

            # get the rep embs for style_lm and style_dis
            rep_embs = torch.matmul(gumbel_weights, style_lm_model.transformer.wte.weight)
            rep_embs = rep_embs * rep_mask.unsqueeze(-1) \
                       + style_lm_model.transformer.wte(rep_mask.new_zeros(rep_mask.size())) \
                       * (1 - rep_mask).unsqueeze(-1)

            log_probs = style_dis_model(inputs_embeds=rep_embs)
            dis_scores = log_probs.exp()[:, 1].mean()
            dis_loss = F.nll_loss(log_probs, torch.ones(log_probs.size(0), dtype=torch.long, device=log_probs.device))

            dis_loss = dis_loss * args.dis_scale
            tr_dis += dis_scores.item()
            tr_dis_loss += dis_loss
            loss += dis_loss

        if args.kl_scale > 0 or args.ce_scale > 0:
            seq_ids = torch.cat([context_ids, response_ids], dim=1)
            seq_ids = seq_ids * (seq_ids != -1).long()
            labels = torch.cat([context_ids, response_ids], dim=1)
            labels[:, :context_len] = -1
            ce_loss, ppl, sample_conv_logits = model(seq_ids, labels=labels)[:3]
            tr_ppl += ppl
            loss += ce_loss * args.ce_scale

        if args.kl_scale > 0:
            rep_mask = (response_ids != -1).long()
            sty_rep_logtis = style_lm_model(response_ids * rep_mask)[0].detach()
            src_rep_logits = sample_conv_logits[:, context_len:, :]
            _, kl_loss = compute_kl(src_rep_logits, sty_rep_logtis, rep_mask, args.kl_scale)
            tr_kl += kl_loss.item()
            loss += kl_loss

        tr_loss += loss.item()

        if n_gpu > 1:
            loss = loss.mean()

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # gradient update
        step += 1
        if step % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0]:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar.set_postfix_str(f"dis: {float(dis_loss):.2f} epoch: {epoch}")
                    pbar.update(1)

            if args.local_rank in [-1, 0] and args.logging_step > 0 and global_step % args.logging_step == 0:
                tb_writer.add_scalar('train/loss', tr_loss / args.logging_step, global_step)
                tb_writer.add_scalar('train/ppl', tr_ppl / args.logging_step, global_step)
                tb_writer.add_scalar('train/kl', tr_kl / args.logging_step, global_step)
                tb_writer.add_scalar('train/dis', tr_dis / args.logging_step, global_step)
                tb_writer.add_scalar('train/dis_loss', tr_dis_loss / args.logging_step, global_step)
                if args.dis_scale > 0:
                    tb_writer.add_scalar('train/gumbel_rep_len', np.mean(gum_rep_lengths), global_step)
                tb_writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
                tr_loss, tr_ppl, tr_kl, tr_dis = 0, 0, 0, 0
                tr_dis_loss = 0
                gum_rep_lengths = []

            if global_step % args.valid_step == 0:
                if args.local_rank == -1 or get_rank() == 0:
                    # only rank 0 process evaluate
                    torch.save(
                        {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                         for k, v in model.state_dict().items()},
                        join(output_dir, f'GP2-pretrain-step-{global_step}.pkl'))
                    logger.info(f"Save model to {output_dir} at step {global_step}.")

                    if args.do_eval:
                        metrics, gen_reps = evaluate(args.eval_input_file, args.eval_batch_size, model, style_dis_model,
                                                     enc, device, args.temperature, args.top_k, args.top_p,
                                                     args.beam_num,
                                                     args.return_num)

                        for m, s in metrics.items():
                            tb_writer.add_scalar(f"eval/{m}", s, global_step)

                        for i in range(4):
                            item = gen_reps[i]
                            logger.info(f"context: {item['context']}")
                            for j in range(2):
                                logger.info(f"response: {item['hyps'][j]}")

                    model.train()
            if global_step >= args.num_optim_steps:
                break

        if (step + 1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()

    if global_step >= args.num_optim_steps:
        break
    epoch += 1

if args.local_rank in [-1, 0]:
    tb_writer.close()
