from pathlib import Path
import sys

sys.path.append(str(Path(__file__).absolute().parent))



from .distinct import distinct
import numpy as np
from collections import OrderedDict, defaultdict


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score



# gts is references, res is generated items
def compute_metrics(hypothesis, references, no_bleu=False, no_emb=True, no_rouge=False, lang='english'):
    assert len(hypothesis) == len(references)
    hypothesis = [_.lower().strip() for _ in hypothesis]
    references = [[_.lower().strip() for _ in refs] for refs in references]

    res, gts = {}, {}
    for i, (h, r) in enumerate(zip(hypothesis, references)):
        res[str(i)] = [h]
        gts[str(i)] = r

    rval = OrderedDict()

    if not no_bleu:
        from .bleu.bleu import Bleu
        _, ss = Bleu(4).compute_score(gts, res)
        for k,v in enumerate(_):
            rval[f'bleu-{k+1}'] = v

    # for i in range(4):
    #     print(f'bleu-{i+1}: {np.sum([_ for _ in ss[i]]) / len(ss[i]):.6f} \n')

    if not no_rouge:
        from .rouge.rouge import Rouge
        _, ss = Rouge().compute_score(gts, res)
        rval['rouge'] = _

    if not no_emb:
        from .emb_evaluate import eval_emb_metrics
        # average, extrama, greedy = [], [], []
        # from tqdm import tqdm
        # for hyp, refs in tqdm(zip(hypothesis, references)):
        #     hyp_list = [hyp if hyp else 'unk']
        #     ref_list = [refs]
        #     ref_list_T = np.array(ref_list).T.tolist()
        #     scores = eval_emb_metrics(hyp_list, ref_list_T)
        #     average.append(scores['EmbeddingAverageCosineSimilairty'])
        #     extrama.append(scores['VectorExtremaCosineSimilarity'])
        #     greedy.append(scores['GreedyMatchingScore'])
        #
        # rval['EmbeddingAverageCosineSimilairty'] = np.mean(average)
        # rval['VectorExtremaCosineSimilarity'] = np.mean(extrama)
        # rval['GreedyMatchingScore'] = np.mean(greedy)

        hyps = hypothesis
        references = [_ if len(_)>=4 else _*4 for _ in references]
        min_num = min([len(_) for _ in references])
        references = [_[:min_num] for _ in references]
        ref_list_T = np.array(references).T.tolist()
        rval.update(eval_emb_metrics(hyps, ref_list_T, lang=lang))

    d1, d2 = distinct(hypothesis)
    rval['distinct_1'] = d1
    rval['distinct_2'] = d2

    etps,_ = cal_entropy(hypothesis)
    for i, s in enumerate(etps):
        rval[f'entropy_{i+1}'] = s


    return rval
