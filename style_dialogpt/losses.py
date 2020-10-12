import torch
import torch.functional as F


def compute_kl(logits, sty_logits, pad_mask, kl_scale=1):
    '''
    Compute the KL
    Args:
        logits: the logits of sampled sequences. shape: B x seq_len x vocab_size
        sty_logits: the logits of sampled sequences from style language model.
        pad_mask: ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens. shape: B x seq_len

    Returns:
        kl_1: B x seq_len
        kl_2: B
    '''
    SMALL_CONST = torch.finfo(logits.dtype).tiny
    probs = torch.softmax(logits, dim=-1)
    sty_probs = torch.softmax(sty_logits, dim=-1)

    sty_probs = sty_probs + SMALL_CONST * (sty_probs < SMALL_CONST).float().detach()
    probs = probs + SMALL_CONST * (probs < SMALL_CONST).float().detach()

    kl_1 = kl_scale * ((probs * (probs / sty_probs).log()).sum(-1) * pad_mask)
    kl_2 = kl_1.sum()

    return kl_1, kl_2
