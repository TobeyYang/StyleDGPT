import json
from collections import OrderedDict

import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

sys.path.append(os.path.dirname(__file__))
from data_loader import GPT2Dataset, BucketBatchSampler
from utils import load_model, boolean_string
from models import Discriminator

logger = logging.getLogger(__name__)

EOS_ID = 50256

def generate_responses(model, input_ids, temperature=1, top_k=None, top_p=None, max_length=30, num_beam=None,
                       num_return_sequence=1, do_sample=True):
    context_len = input_ids.size(1)
    output_sequence = model.generate(
        input_ids=input_ids,
        max_length=max_length+context_len,
        do_sample=do_sample,
        num_beams=num_beam,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=0,
        eos_token_ids=EOS_ID,
        length_penalty=None,
        num_return_sequences=num_return_sequence,
    )

    output_sequence = output_sequence.cpu()

    assert input_ids.size(0) * num_return_sequence == output_sequence.size(0)

    rep_ids = output_sequence[:,context_len:].view(input_ids.size(0), num_return_sequence, -1)

    #build dict
    rep_tensor_list = [_ for _ in rep_ids] # tensor list
    rep_ids_list = [[cut_seq_to_eos(_) for _ in l.tolist()] for l in rep_tensor_list]

    return rep_tensor_list, rep_ids_list


def evaluate(eval_input_file, batch_size, model, disc_model, tokenizer, device=torch.device('cuda:0'),
             temperature=None, top_k=None, top_p=None, beam_num=None, num_return=None, no_emb=True):
    eval_data_set = GPT2Dataset(eval_input_file, tokenizer=tokenizer, is_test=True)
    sampler = BucketBatchSampler(eval_data_set.context_lens, batch_size, False, False)
    eval_data_loader = DataLoader(eval_data_set, batch_sampler=sampler, collate_fn=eval_data_set.collate_fn)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data_set))
    logger.info("  Batch size = %d", batch_size)

    model = model.module if hasattr(model, 'module') else model
    model = model.to(device)

    model.eval()
    disc_model.eval()
    eval_reps = []
    with torch.no_grad():
        for batch in tqdm(eval_data_loader, desc='eval', disable='PT_DATA_DIR' in os.environ.keys()):
            con_ids = batch[0].to(device)
            reps_1, reps_2 = generate_responses(model, con_ids, temperature, top_k, top_p, 25,
                                                beam_num, num_return)
            eval_reps += list(zip(reps_1, reps_2))

    from functools import reduce
    order_ids = reduce((lambda x, y: x + y), sampler)
    assert len(order_ids) == len(eval_reps)
    # re-order
    zip_ = list(zip(eval_reps, order_ids))
    zip_ = sorted(zip_, key=lambda x: x[1])
    eval_reps = [_[0] for _ in zip_]

    # order the responses by disc_model score. As the generation probs are close, we sort topk hyps according to intensity.
    logger.info("Begin to score the hypothesis...")
    top_scores, top_reps = [], []
    gen_reps = []
    with torch.no_grad():
        for i, rp in enumerate(eval_reps):
            # ss = torch.softmax(disc_model(rp[0]), dim=1)[:, 1]
            ss = disc_model(rp[0]).exp()[:, 1]
            index = torch.argmax(ss)
            top_scores.append(ss[index].item())
            top_reps.append(rp[1][index])

            # order the reps by score
            ss = ss.tolist()
            zip_ = list(zip(ss, [tokenizer.decode(_) for _ in rp[1]]))
            zip_ = sorted(zip_, key=lambda x: x[0], reverse=True)
            gen_reps.append({"context": eval_data_set[i].context,
                             "reps": eval_data_set[i].responses,
                             "hyps": zip_})

    assert len(top_reps) == len(eval_data_set)

    logger.info("Begin to calculate metrics...")
    try:
        from metrics.compute_metrics import compute_metrics
    except:
        from src.style_rl.metrics.compute_metrics import compute_metrics

    refs = [_.responses for _ in eval_data_set]
    hyps = [tokenizer.decode(_) for _ in top_reps]
    metrics = compute_metrics(hyps, refs, no_emb=no_emb)
    metrics['intensity'] = np.mean(top_scores)
    metrics['avg_length'] = np.mean([len(_.split()) for _ in hyps])

    for m, s in metrics.items():
        logger.info(f'{m}:  {s:.4f}')

    return metrics, gen_reps



def evaluate_from_json(hyps_fi, disc_model=None, tokenizer=None, device=torch.device('cuda:0'), no_emb=False):
    pass


if __name__ == '__main__':
    from argparse import ArgumentParser
    from transformers import GPT2Config, GPT2Tokenizer

    from models.modeling_gpt2 import GPT2LMHeadModel

    parser = ArgumentParser("evaluate")
    parser.add_argument('--model_name_or_path', type=str, default='',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sty_lm_model_name_or_path", type=str)
    parser.add_argument('--sty_dic_model_fi', type=str)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument("--load_checkpoint", '-c', type=str, default='')
    parser.add_argument('--eval_input_file', type=str, help='the test tsv file.')
    parser.add_argument('--output_fi', type=str, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--beam_num", type=int, default=None)
    parser.add_argument("--return_num", type=int, default=50)

    parser.add_argument("--no_emb", type=boolean_string, default=True)
    parser.add_argument('--json_fi', type=str, help='the hypothesis json file.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    device = torch.device(f"cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    args.device = device

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config = GPT2Config.from_pretrained(args.model_name_or_path)
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

    style_config = GPT2Config.from_pretrained(args.sty_lm_model_name_or_path)
    style_lm_model = GPT2LMHeadModel.from_pretrained(args.sty_lm_model_name_or_path, config=style_config)
    style_dis_model = Discriminator.from_pretrained(args.sty_dic_model_fi, encoder=style_lm_model.transformer,
                                                    device=device)
    style_lm_model.to(device)
    style_dis_model.to(device)
    style_lm_model.eval()
    style_dis_model.eval()

    if args.eval_input_file:
        logger.info(f"Begin to evaluate file {args.eval_input_file}")

        model = load_model(GPT2LMHeadModel(config), args.load_checkpoint, args, verbose=True)
        model.to(device)
        model.eval()

        metrics, gen_reps = evaluate(args.eval_input_file, args.eval_batch_size, model, style_dis_model, enc, device,
                                     args.temperature, args.top_k, args.top_p, args.beam_num, args.return_num, args.no_emb)

        fi = args.output_fi if args.output_fi else args.load_checkpoint + ".hyps.json"
        with open(fi, 'w', encoding='utf8')as p:
            json.dump(gen_reps, p, indent=2)
        logger.info(f"Dump the generated hypothesis to {fi}")

    elif args.json_fi:
        logger.info(f"Begin calculate the metrics from {args.json_fi}")
        evaluate_from_json(args.json_fi, disc_model=style_dis_model, tokenizer=enc, device=device, no_emb=args.no_emb)


