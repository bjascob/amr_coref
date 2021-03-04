#!/usr/bin/python3
import re
import os
import logging
from   collections import Counter
from   amr_coref.utils.logging import setup_logging
from   amr_coref.coref.coref_mention_data import CorefMentionData
from   amr_coref.utils.data_utils import load_json


# Get the token counts for in cluster and out of cluster.
def get_token_counts(coref_fpath):
    print('Loading', coref_fpath)
    t_in_ctr = Counter()
    t_no_ctr = Counter()
    cr_data = load_json(coref_fpath)
    mdata = CorefMentionData(cr_data, None)
    for doc_name, mentions in mdata.mentions.items():
        for midx, mention in enumerate(mentions):
            token = mention.token
            if mention.cluster_id != None:
                t_in_ctr[token] += 1
            else:
                t_no_ctr[token] += 1
    return t_in_ctr, t_no_ctr


if __name__ == '__main__':
    setup_logging(logfname='logs/create_mention_tokens.log', level=logging.WARN)
    train_fpath = 'data/tdata/train.json.gz'
    test_fpath  = 'data/tdata/test.json.gz'
    out_fpath   = 'data/tdata/mention_tokens.txt'
    num_print   = 20

    # Get unique token names for in and out of cluster tokens
    train_in_ctr, train_no_ctr = get_token_counts(train_fpath)
    test_in_ctr,  test_no_ctr  = get_token_counts(test_fpath)
    train_in_set = set(train_in_ctr.keys())
    train_no_set = set(train_no_ctr.keys())
    test_in_set  = set(test_in_ctr.keys())
    test_no_set  = set(test_no_ctr.keys())

    print('Saving mention token list to', out_fpath)
    with open(out_fpath, 'w') as f:
        for token in sorted(test_in_set|train_in_set):
            f.write('%s\n' % token)
    print()

    num_in_train  = sum(train_in_ctr.values())
    num_no_train  = sum(train_no_ctr.values())
    num_train_tot = num_in_train + num_no_train
    num_in_test   = sum(test_in_ctr.values())
    num_no_test   = sum(test_no_ctr.values())
    num_test_tot  = num_in_test + num_no_test
    num_in_total  = num_in_train + num_in_test
    num_no_total  = num_no_train + num_no_test
    num_total     = num_in_total + num_no_total

    print('Number of mentions in CorefMentionData(mention_set_fn=None)')
    print('Stats:   InCluster   NoCluster       Total')
    print('-'*42)
    print('Train:  {:10,}  {:10,}  {:10,}'.format(num_in_train, num_no_train, num_train_tot))
    print('Test:   {:10,}  {:10,}  {:10,}'.format(num_in_test,  num_no_test,  num_test_tot))
    print('-'*42)
    print('Total:  {:10,}  {:10,}  {:10,}'.format(num_in_total, num_no_total, num_total))
    print()
    print()

    print('Number of unique mention tokens')
    print('Stats:   InCluster   NoCluster')
    print('-'*30)
    print('Train:  {:10,}  {:10,}'.format(len(train_in_set), len(train_no_set)))
    print('Test:   {:10,}  {:10,}'.format(len(test_in_set),  len(test_no_set)))
    print('-'*30)
    print('Diff:   {:10,}  {:10,}'.format(len(test_in_set-train_in_set),  len(test_no_set-train_no_set)))
    print('Sum:    {:10,}  {:10,}'.format(len(test_in_set|train_in_set),  len(test_no_set|train_no_set)))
    print()

    # Graph tokens in the test set are included in the final mention set but may be problematic
    # since there are no training examples where they are in clusters. They still might be
    # clustered correcly if the pre-trained embeddings or their other features are enough.
    # print(' '.join(sorted(test_in_set-train_in_set)))
