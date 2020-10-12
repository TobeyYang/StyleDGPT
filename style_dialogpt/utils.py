

from __future__ import print_function

import math
import pickle
import torch.distributed

import os
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

SEQ_LENGTH_SHRINK_PROP = 0.9

from transformers import GPT2Tokenizer

from models.modeling_gpt2 import GPT2LMHeadModel


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def set_lr(optimizer, step, schedule, lr, warmup_steps, warmup_proportion, n_embd, tot_steps):
    from models.optim import warmup_linear, noam_decay, noamwd_decay
    if schedule == 'None':
        lr_this_step = lr
    elif schedule == 'noam':  # transformer like
        lr_this_step = lr * 1e4 * noam_decay(step + 1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  # transformer like
        lr_this_step = lr * 1e4 * noamwd_decay(step + 1, warmup_steps, n_embd)
    else:
        lr_this_step = lr * warmup_linear(step / tot_steps,
                                          warmup_proportion)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step


def load_model(model, checkpoint, args, verbose=False):
    device = args.device
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
                and all(not s.startswith('transformer.')
                        for s in model_state_dict.keys())):
            logger.info('loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

    # if args.fp16:
    #     logger.info('in fp16, model.half() activated')
    #     model.half()
    model.to(device)
    # if n_gpu > 1:
    #     logging.info('data parallel because more than one gpu')
    #     model = torch.nn.DataParallel(model)
    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')

        # fix the divergence between ``pytorch_pretrained_bert'' and ``transformers''
        if new_key == 'lm_head.decoder.weight':
            new_key = 'lm_head.weight'

        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict


def is_master(opt, device_id):
    return opt.gpu_ranks[device_id] == 0


def multi_init(opt, device_id, logger=None):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=opt.master_ip,
        master_port=opt.master_port)
    dist_world_size = opt.world_size
    torch.distributed.init_process_group(
        backend=opt.gpu_backend, init_method=dist_init_method,
        world_size=dist_world_size, rank=opt.gpu_ranks[device_id])
    gpu_rank = torch.distributed.get_rank()
    if not is_master(opt, device_id) and logger is not None:
        logger.disabled = True

    return gpu_rank


def all_reduce_and_rescale_tensors(tensors, rescale_denom, buffer_size=10485760):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset + numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset])
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset + numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            torch.distributed.all_reduce(t)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size)
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size + 2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    return y

    # if not hard:
    #     return y
    #     # return y.view(-1, latent_dim * categorical_dim)

    # shape = y.size()
    # _, ind = y.max(dim=-1)
    # y_hard = torch.zeros_like(y).view(-1, shape[-1])
    # y_hard.scatter_(1, ind.view(-1, 1), 1)
    # y_hard = y_hard.view(*shape)
    # # Set gradients w.r.t. y_hard gradients w.r.t. y
    # y_hard = (y_hard - y).detach() + y
    # return y_hard.view(-1, latent_dim * categorical_dim)


def _gumbel_sequence_sample(model: GPT2LMHeadModel, input_ids, temperature, rep_length):
    '''

    Args:
        model:
        tenperature:
        rep_length:

    Returns:
        logits: the source logits of each token [B x seq_len x vsize]
        embeds: the representations of each token [B x  seq_len x hidden_dim]
    '''
    cur_len = 0
    past = None
    logits, embeds = [], []
    input_emb = None

    while cur_len <= rep_length:
        if input_emb is not None:
            input_emb = input_emb.unsqueeze(1)
            outputs = model(inputs_embeds=input_emb, past=past)
        else:
            outputs = model(input_ids, past=past)
        next_token_logits = outputs[0][:, -1, :]
        past = outputs[1]

        gumbel_weights = gumbel_softmax(next_token_logits, temperature)
        input_emb = torch.matmul(gumbel_weights, model.transformer.wte.weight)

        logits.append(next_token_logits)
        embeds.append(input_emb)
        cur_len += 1

    logits = logits[1:]     # remove the fist logits for <|endoftext|>
    embeds = embeds[:-1]    # remove the last input_emb

    logits = torch.stack(logits, 1)
    embeds = torch.stack(embeds, 1)
    assert logits.size(1) == rep_length

    return (logits, embeds)



def gumbel_sequence_sample(model: GPT2LMHeadModel, input_ids, temperature, rep_length, enc:GPT2Tokenizer):
    '''

    Args:
        model:
        tenperature:
        rep_length:

    Returns:
        logits: the source logits of each token [B x seq_len x vsize]
        embeds: the representations of each token [B x  seq_len x hidden_dim]
    '''
    eos_id = enc.eos_token_id
    cur_len = 0
    past = None
    input_emb = None

    sample_mask = torch.ones(input_ids.size(0), rep_length, device=input_ids.device).type_as(input_ids)
    gumbel_weights = []
    logits = []
    # argmax_id = []
    # gumbel_id = []

    while cur_len <= rep_length:
        if input_emb is not None:
            input_emb = input_emb.unsqueeze(1)
            outputs = model(inputs_embeds=input_emb, past=past)
        else:
            outputs = model(input_ids, past=past)
        next_token_logits = outputs[0][:, -1, :]
        past = outputs[1]
        g_weights = gumbel_softmax(next_token_logits, temperature)

        # argmax_id.append(next_token_logits[0].argmax().item())
        # gumbel_id.append(g_weights[0].argmax().item())

        input_emb = torch.matmul(g_weights, model.transformer.wte.weight)

        # if the input_emb is <|endoftext|>
        eos_probs = g_weights[:,eos_id].detach()
        not_eos = (eos_probs < 0.5).type_as(sample_mask)
        sample_mask[:,cur_len+1:] = sample_mask[:, cur_len+1:] * not_eos.unsqueeze(-1)

        gumbel_weights.append(g_weights)
        logits.append(next_token_logits)
        cur_len += 1

    logits = logits[1:]     # remove the fist logits for <|endoftext|>
    gumbel_weights = gumbel_weights[:-1]

    logits = torch.stack(logits, 1)
    gumbel_weights = torch.stack(gumbel_weights, 1)

    assert logits.size(1) == rep_length

    return (logits, gumbel_weights, sample_mask)


def tsv2binary(data_file, model_name_or_path):
    from transformers import GPT2Tokenizer
    import pickle
    import sys, pathlib
    # sys.path.append(pathlib.Path(__file__).absolute().parent.parent)
    # [print(_) for _ in sys.path]
    from data_loader import RedditExample
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

    res = []
    with open(data_file, encoding='utf8')as p:
        for i, line in tqdm(enumerate(p)):
            con, rep = line.strip().split('\t')
            con = ' '.join(con.split())
            rep = ' '.join(rep.split())

            con_id = tokenizer.encode(con)
            rep_id = tokenizer.encode(rep)
            res.append(RedditExample(
                conv_id=i,
                context=con,
                response=rep,
                context_id=con_id,
                response_id=rep_id
            ))
    with open(data_file + '.pkl', 'wb')as p:
        pickle.dump(res, p)


if __name__ == '__main__':
    tsv2binary('/mnt/tobey/DialoGPT/data/reddit/valid.tsv', '/mnt/tobey/StyleGPT/models/medium')
    tsv2binary('/mnt/tobey/DialoGPT/data/reddit/train.tsv', '/mnt/tobey/StyleGPT/models/medium')
