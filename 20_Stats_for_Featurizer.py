#!/usr/bin/python3
import numpy
from   tqdm import tqdm
from   collections import defaultdict
from   amr_coref.utils.logging import setup_logging, ERROR
from   amr_coref.coref.coref_mention_data import CorefMentionData
from   amr_coref.utils.data_utils import load_json
from   amr_coref.coref.vocab_embeddings import load_word_set


if __name__ == '__main__':
    setup_logging(level=ERROR)
    coref_fpath = 'data/tdata/train.json.gz'
    #coref_fpath = 'data/tdata/test.json.gz'
    men_set_fn  = 'data/tdata/mention_tokens.txt'
    max_dist = 999999

    print('Loading and testing', coref_fpath)
    mention_set = load_word_set(men_set_fn)
    cr_data     = load_json(coref_fpath)
    mdata       = CorefMentionData(cr_data, mention_set)
    print('There are {:,} documents'.format(len(mdata.mentions.keys())))
    print()

    # Stats for max anaphor to antecedent distances
    distances  = []
    mlist_lens = []
    pair_count = 0
    for doc_name, mlist in mdata.mentions.items():
        mlist_lens.append(len(mlist))
        for midx in range(len(mlist)):
            mention = mlist[midx]
            antecedents = mlist[:midx]
            pair_count += len(antecedents)
            if mention.cluster_id:
                for antecedent in antecedents:
                    if mention.cluster_id == antecedent.cluster_id:
                        dist = mention.mdata_idx-antecedent.mdata_idx
                        distances.append( dist )
    assert numpy.min(distances) > 0
    print('Mention list lengths go from %d to %d' % (min(mlist_lens), max(mlist_lens)))
    print('There are {:,} total (potential) anaphor to antecedent pairs'.format(pair_count))
    print('with {:,} pairs in clusters {:.1f}%'.format(len(distances), 100.*len(distances)/pair_count))
    print('Max distance is {:,} and the average is {:.1f} with a stdev of {:.1f}'.format(\
        numpy.max(distances), numpy.mean(distances), numpy.std(distances)))
    print()

    print('Stats for features')
    stats = defaultdict(list)
    for doc_name, mlist in tqdm(mdata.mentions.items(), ncols=100, leave=False):
        for midx in range(len(mlist)):
            mention     = mlist[midx]
            antecedents = mlist[:midx]
            antecedents = antecedents[-max_dist:]
            # Accumulate so data for statistics
            stats['sent_idx'].append(mention.sent_idx)
            stats['tok_idx'].append(mention.tok_idx)
            stats['sidx_diff'] = [mention.sent_idx - a.sent_idx for a in antecedents]
            doc_idx = mdata.get_doc_tok_idx(mention)
            stats['doc_idx_diff'] += [doc_idx - mdata.get_doc_tok_idx(a) for a in antecedents]
            stats['men_idx_diff'] += [mention.mdata_idx - a.mdata_idx for a in antecedents]
    for key, values in stats.items():
        mean = numpy.mean(values)
        std  = numpy.std(values)
        print('%-12s  mean=%5d  max=%5d  stdev=%7.1f  95%%CI=%7.1f' % (key, mean, numpy.max(values),
            std, mean+2*std))
