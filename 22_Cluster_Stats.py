#!/usr/bin/python3
import os
import logging
from   collections import Counter
from   amr_coref.utils.logging import setup_logging
from   amr_coref.coref.coref_mention_data import CorefMentionData
from   amr_coref.utils.data_utils import load_json
from   amr_coref.coref.vocab_embeddings import load_word_set


if __name__ == '__main__':
    setup_logging(level=logging.ERROR)
    coref_fpath = 'data/tdata/train.json.gz'
    #coref_fpath = 'data/tdata/test.json.gz'
    men_set_fn  = 'data/tdata/mention_tokens.txt'
    num_print   = 10

    # Load data and get the token counts
    print('Loading', coref_fpath)
    token_in_ctr = Counter()
    token_no_ctr = Counter()
    mention_set  = load_word_set(men_set_fn)
    cr_data      = load_json(coref_fpath)
    mdata        = CorefMentionData(cr_data, mention_set)
    for doc_name, mentions in mdata.mentions.items():
        for midx, mention in enumerate(mentions):
            if mention.cluster_id != None:
                token_in_ctr[mention.token] += 1
            else:
                token_no_ctr[mention.token] += 1
    print()

    # Stats for mention data
    num_men_in   = sum(token_in_ctr.values())
    num_mentions = num_men_in + sum(token_no_ctr.values())
    tok_in_set   = set(token_in_ctr.keys())
    tok_no_set   = set(token_no_ctr.keys())
    num_tokens   = len(tok_in_set|tok_no_set)
    num_tok_in   = len(tok_in_set)
    print('Stats for CorefMentionData(men_set_fn=%s)' % str(men_set_fn))
    print('There are {:,} mentions and {:,} are in a cluster = {:.1f}%'.format(\
            num_mentions, num_men_in, 100*num_men_in/num_mentions))
    print('There are {:,} unique tokens and {:,} appear in clusters = {:.1f}%'.format(\
        num_tokens, num_tok_in, 100*num_tok_in/num_tokens))
    print()

    # Print counts / percentages of tokens in clusters
    pct_list = []
    for token, in_count in token_in_ctr.most_common():
        no_count = token_no_ctr[token]
        pct = 100*in_count/(in_count+no_count)
        pct_list.append((token, in_count, no_count, pct))
    #pct_list = sorted(pct_list, key=lambda x:x[3], reverse=True)   # sort by pct
    pct_list = sorted(pct_list, key=lambda x:x[1], reverse=True)    # sort by in_count
    print('%-24s %12s %12s' % ('Token', '#Cluster', '#No-Cluster'))
    print('-'*60)
    for token, in_count, no_count, pct in pct_list[:num_print]:
        print('%-24s %12d %12d %6.1f%%' % (token, in_count, no_count, pct))
    print()
