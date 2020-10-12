'''
build arxiv data for discriminator training.
data source : arxiv, reddit, holmes
'''
import os, sys
from random import shuffle
from tqdm import tqdm
import argparse

# reddit_data = '/mnt/tobey/StyleFusion/src_data/Reddit/conv(d2-10,l30w,s0,t1)/ref_10_2011/merge.tsv'
# #style_data = '../arxiv/merge.tsv'
# #res_fi = 'arxiv/style.tsv'
# style_data = '../holmes/bias_nonc_all.txt'
# res_fi = 'holmes/style.tsv'

# ratio = 5
EOS='<|endoftext|>'

res = []
def main(style_data, reddit_data, ratio, res_fi)
    with open(style_data, encoding='utf8')as p:
        lines = p.readlines()
        iterator = tqdm(lines)
        if EOS in lines[0]:
            res += ['style' + '\t' + _ for _ in iterator if _]
        else:
            res += ['style' + '\t' + _.strip() + f' {EOS}\n' for _ in iterator if _]
        sty_num = len(res)

    with open(reddit_data, encoding='utf8')as p:
        for i, line in tqdm(enumerate(p), desc='non_style'):
            if not line.strip():
                continue

            res.append('non_style'+'\t'+ line.strip().split('\t')[1] + f' {EOS}\n')
            if len(res)>= (ratio+1) * sty_num:
                break

    shuffle(res)
    with open(res_fi, 'w', encoding='utf8')as p:
        p.writelines(res)

    print('Bingo')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Build the data for the discriminator training.')
    parser.add_argument("--style_data", 
                        type=str, 
                        default="data/dis_data/arxiv/merge.tsv"
                        help='the path to the stylized data file.')
    parser.add_argument("--reddit_data", 
                        type=str, 
                        default="data/reddit/conv(d2-10,l30w,s0,t1)/ref_10_2011/merge.tsv"
                        help='the path to the reddit data file.')
    parser.add_argument("--ratio", 
                        type=float, 
                        defalult=5, 
                        help='the sampling ratio (negative:positive)')
    parser.add_argument("--res_fi", 
                        type=str, 
                        help="the result file.")
    args = parser.parse_args()

    main(args.style_data, args.reddit_data, args.ratio, args.res_fi)

