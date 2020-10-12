from collections import Counter
import argparse
from nltk import ngrams

def distinct(hypothesis):
    unigram_counter, bigram_counter = Counter(), Counter()
    for hypo in hypothesis:
        tokens = hypo.split()
        unigram_counter.update(tokens)
        bigram_counter.update(ngrams(tokens, 2))

    distinct_1 = len(unigram_counter) / sum(unigram_counter.values())
    distinct_2 = len(bigram_counter) / sum(bigram_counter.values())
    return distinct_1, distinct_2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', default="results.json",help="predicted text file, one example per line")
    args = parser.parse_args()

    with open(args.hypothesis, 'r', encoding='utf8')as p:
        hypothesis = [_.strip() for _ in p.readlines()]
        d1, d2 = distinct(hypothesis)
        print(f'Distinct1: {d1:.6f}')
        print(f'Distinct2: {d2:.6f}')