
import math
import pickle
import random
from collections import OrderedDict

import torch
import copy
import logging

from torch.utils.data import DataLoader, Dataset, SequentialSampler

from env import END_OF_TEXT_TOKEN


from torch.utils.data.sampler import BatchSampler

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditExample(object):
    def __init__(self, conv_id, context, response, context_id, response_id):
        self.conv_id = conv_id
        self.context = context
        self.response = response
        self.context_id = context_id
        self.response_id = response_id

    def __repr__(self):
        return 'conv_id = {}\ncontext = {}\nresponse = {}'.format(
            self.conv_id, self.context, self.response)

    def __str__(self):
        return self.__repr__()


class RedditMultiRefExample(object):
    def __init__(self, conv_id, context, responses, context_id, responses_ids, hypothesis=None):
        self.conv_id = conv_id
        self.context = context
        self.responses = responses
        self.context_id = context_id
        self.response_ids = responses_ids
        self.hypothesis = None

    def __repr__(self):
        return 'conv_id = {}\ncontext = {}\nresponse = {}'.format(
            self.conv_id, self.context, '\n'.join(self.responses))

    def __str__(self):
        return self.__repr__()


class BucketBatchSampler(BatchSampler):
    '''
    The samples within each batch have same length.
    '''

    def __init__(self, lens, batch_size, drop_last=False, shuffle=False):
        logger.info('Build BucketSampler...')
        self.lens = lens
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._build_order()
        logger.info('Done!')

    def _build_order(self):
        self.ids = list(range(len(self.lens)))
        self.length_dict = {}
        for id in self.ids:
            k = str(self.lens[id])
            if k not in self.length_dict:
                self.length_dict[k] = [id]
            else:
                self.length_dict[k].append(id)

        if self.shuffle:
            for k, v in self.length_dict.items():
                random.shuffle(v)

        self.buckets = []
        for k, v in self.length_dict.items():
            b_num = int(len(v) / self.batch_size) if self.drop_last else math.ceil(len(v) / self.batch_size)
            for i in range(b_num):
                self.buckets.append(v[i * self.batch_size: (i + 1) * self.batch_size])

        if self.shuffle:
            random.shuffle(self.buckets)

    def __iter__(self):
        return iter(self.buckets)

    def __len__(self):
        return len(self.buckets)


class GPT2Dataset(Dataset):
    def __init__(self, data_file, max_len=None, tokenizer=None, portions=None, is_test=False):
        '''
        Args:
            data_file: The pkl file stored all reddit examples.
            max_len: max length of response.
        '''

        assert portions is None or max(portions) == 1

        logger.info(f"Build dataset from {data_file}")
        self.data_file = data_file
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.portions = portions
        self.is_test = is_test

        self.data_set = self._read_data_set()
        self.context_lens = [len(_.context_id) for _ in self.data_set]

        logger.info("Done!")

    def _read_data_set(self):
        if self.data_file.endswith('pkl'):
            with open(self.data_file, 'rb')as p:
                data_set = pickle.load(p)
        elif self.data_file.endswith('tsv') and not self.is_test:
            data_set = []
            with open(self.data_file, encoding='utf8')as p:
                for id, line in enumerate(p):
                    # if id > 100:  # debug
                    #     break
                    line = line.strip()
                    if not line:
                        continue
                    if len(line.split('\t')) == 2:
                        con, rep = line.split('\t')
                    else:
                        con, rep = line, line
                    con = ' '.join(con.split())
                    rep = ' '.join(rep.split())
                    con_id, rep_id = self.tokenizer.encode(con), self.tokenizer.encode(rep)
                    data_set.append(RedditExample(id, con, rep, con_id, rep_id))

        elif self.data_file.endswith('tsv') and self.is_test:
            data_set = []
            conv_dict = OrderedDict()
            with open(self.data_file, encoding='utf8')as p:
                for line in p:
                    line = line.strip()
                    if not line:
                        continue
                    con, rep = line.split('\t')
                    if con not in conv_dict and len(conv_dict)>=2000:
                        break

                    if con not in conv_dict:
                        conv_dict[con] = [rep]
                    else:
                        conv_dict[con].append(rep)

                for i, (con, reps) in enumerate(conv_dict.items()):
                    con_id = self.tokenizer.encode(con)
                    reps_ids = [self.tokenizer.encode(rep) for rep in reps]
                    data_set.append(RedditMultiRefExample(i, con, reps, con_id, reps_ids))
        else:
            raise Exception(f"Data file {self.data_file} is invalid (is_test: {self.is_test}).")

        if self.portions is None:
            return data_set

        else:
            temp = []
            portion_size = int(len(data_set) / len(self.portions))
            for i, _ in enumerate(self.portions):
                if _:
                    temp += data_set[i * portion_size: (i + 1) * portion_size]
            return temp

    def __getitem__(self, i):
        # add the <|endoftext|> token at end of context and response
        end_of_text_id = self.tokenizer.eos_token_id
        item = copy.deepcopy(self.data_set[i])
        item.context_id.append(end_of_text_id)
        if isinstance(item, RedditExample):
            item.response_id.append(end_of_text_id)
            if not self.is_test and self.max_len and len(item.response_id) > self.max_len:
                item.response_id = item.response_id[:self.max_len]

        return item

    def __len__(self):
        return len(self.data_set)

    @staticmethod
    def collate_fn(examples):
        try:
            contexts = [e.context_id for e in examples]
            batch_contexts = torch.tensor(contexts, dtype=torch.long, device='cpu')
            res = (batch_contexts,)

            if isinstance(examples[0], RedditExample):
                responses = [e.response_id for e in examples]
                # pad or cut response to max_length
                max_rep = max([len(_) for _ in responses])
                max_rep = min(max_rep, 50)
                responses = [_[:max_rep] + [-1] * (max_rep - len(_)) for _ in responses]
                batch_responses = torch.tensor(responses, dtype=torch.long, device='cpu')
                res += (batch_responses,)

            return res
        except:
            print(vars(examples))


class BucketDataLoader(object):

    def __init__(self, data_file, batch_size, max_seq_len, tokenizer=None,
                 drop_last=False, shuffle=False, rank=-1, world_size=-1):
        '''
        Bucket data loader.
        Args:
            data_file:
            batch_size:
            max_seq_len:
            tokenizer:
            drop_last:
            shuffle:
            rank:
            world_size:
        '''
        self.data_file = data_file
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.portions = self._get_portion()

        self.data_set = GPT2Dataset(self.data_file, self.max_seq_len, self.tokenizer, self.portions)
        self.sampler = BucketBatchSampler(self.data_set.context_lens, self.batch_size, self.drop_last, self.shuffle)

    def _get_portion(self):
        '''
        portions to load.
        Args:
            portions: 0,1 list.  ``[0,0,1,0]'' presents the data is divided into 4 portions and just load the third portion.
        '''
        if self.rank != -1 and self.world_size != -1:
            portions = [1 if i == self.rank else 0 for i in range(self.world_size)]
        else:
            portions = [1]
        return portions

    def __iter__(self):
        self.loader = DataLoader(self.data_set, batch_sampler=self.sampler, num_workers=5,
                                 collate_fn=GPT2Dataset.collate_fn)
        yield from self.loader

    def __len__(self):
        return len(self.sampler)


class SequentialDataLoader(object):
    def __init__(self, data_file, max_seq_len=None, tokenizer=None):
        self.data_set = GPT2Dataset(data_file, max_seq_len, tokenizer)
        self.sampler = SequentialSampler(self.data_set)

    def __iter__(self):
        yield from DataLoader(self.data_set, 1, False, self.sampler,
                              num_workers=1, collate_fn=GPT2Dataset.collate_fn)

    def __len__(self):
        return len(self.data_set)


if __name__ == '__main__':
    pass
