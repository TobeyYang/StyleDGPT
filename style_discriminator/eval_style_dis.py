import argparse, os, sys
from transformers import GPT2Config, GPT2Tokenizer
import torch

try:
    from modeling_discriminator import Discriminator
    from modeling_gpt2 import GPT2LMHeadModel
except:
    from src.style_discriminator.modeling_discriminator import Discriminator
    from src.style_discriminator.modeling_gpt2 import GPT2LMHeadModel

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def clf_eval(style_dis_model, tokenizer, sentences):
    scores = []
    bsz = 600
    for i in tqdm(range(0, len(sentences), bsz)):
        with torch.no_grad():
            ids = [tokenizer.encode(_.strip())[:50] + [tokenizer.eos_token_id] for _ in sentences[i:i + bsz]]
            ids = [torch.tensor(_, dtype=torch.long) for _ in ids]
            ids = pad_sequence(ids, batch_first=True, padding_value=0)
            ids = ids.cuda()

            logits = style_dis_model(ids)
            scores += logits.exp()[:, 1].tolist()

    score = sum(scores) / len(scores)
    print(f'Test {len(scores)} sentences, score: {score:.2f}')
    return scores, score


def filter(style_dis_model, tokenizer, input_fi, output_fi, threshold):
    with open(input_fi, encoding='utf8')as p:
        lines = p.readlines()#[:10_000]

    contexts, responses = [], []
    for line in lines:
        con, rep = line.strip().split('\t')
        contexts.append(con)
        responses.append(rep)

    scores, _ = clf_eval(style_dis_model, tokenizer, responses)

    res = []
    for s, con, rep in zip(scores, contexts, responses):
        if s > threshold:
            res.append(f"{con}\t{rep}\n")

    with open(output_fi, 'w', encoding='utf8')as p:
        p.writelines(res)

    print(f"threshold:{threshold} before: {len(lines)} after:{len(res)} ")


def interact(style_dis_model, tokenizer):
    while True:
        print('---please input---')
        text = input()
        if text:
            ids = torch.tensor([tokenizer.encode(text.strip()) + [tokenizer.eos_token_id]], dtype=torch.long)
            ids = ids.cuda()
            prob = style_dis_model(ids).exp()[:, 1].squeeze().item()
            print('The score: {:.4f}'.format(prob))
            print("\n" * 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='interact or eval or filter')
    parser.add_argument('--hyp', type=str, help='the hypothesis file')
    parser.add_argument('--src', type=str, help='the source file to filter')
    parser.add_argument('--res', type=str, help='the result file')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--sty_dic_model_fi', type=str)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    style_config = GPT2Config.from_pretrained(args.model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    gpt2_model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=style_config)
    gpt2_model = gpt2_model.cuda()
    style_dis_model = Discriminator.from_pretrained(args.sty_dic_model_fi, encoder=gpt2_model.transformer,
                                                    device='cuda:0')
    style_dis_model = style_dis_model.cuda()

    if args.mode == 'interact':
        interact(style_dis_model, tokenizer)
    elif args.mode == 'eval':
        with open(args.hyp, encoding='utf8')as p:
            sentences = [_.strip() for _ in p.readlines()]
        clf_eval(style_dis_model, tokenizer, sentences)
    elif args.mode == 'filter':
        filter(style_dis_model, tokenizer, args.src, args.res, args.threshold)
    else:
        print('Please check the mode')
