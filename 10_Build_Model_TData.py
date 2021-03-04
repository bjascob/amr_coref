#!/usr/bin/python3
import os
import logging
import numpy
from   amr_coref.utils.logging import setup_logging, silence_penman
from   amr_coref.coref.build_coref_tdata import build_coref_tdata
from   amr_coref.utils.data_utils import dump_json


def print_stats(data):
    # Doc / cluster stats
    ndocs     = len(data['clusters'])
    nclusters = sum([len(doc)      for doc in data['clusters'].values()])
    nmentions = sum([len(mentions) for doc in data['clusters'].values() for mentions in doc])
    print('There are {:,} docs with a total of {:,} clusters and {:,} mentions'.format(\
        ndocs, nclusters, nmentions))
    # Graph stats
    ngraphs   = len(data['gdata'])
    num_sents = [len(doc_ids)  for doc_ids in data['doc_gids'].values()]
    gd_list   = data['gdata'].values()
    num_vars  = [len(gdata['var2concept']) for gdata in gd_list]
    n_g_w_sg  = sum([1 if 'sg_vars' in gdata else 0 for gdata in gd_list])
    num_sg    = sum([len(gdata.get('sg_vars', [])) for gdata in gd_list])
    print('There are {:,} graphs and {:,} have a total of {:,} sub-graphs'.format(
        ngraphs, n_g_w_sg, num_sg))
    print('The average number of graphs per document is %.1f and the max is %.1f' % \
        (numpy.mean(num_sents), numpy.max(num_sents)))
    print('The average number of variables per graph is %.1f and the max is %.1f' % \
        (numpy.mean(num_vars), numpy.max(num_vars)))



if __name__ == '__main__':
    setup_logging(logfname='logs/build_model_tdata.log', level=logging.WARN)
    silence_penman()
    amr3_dir = '/home/bjascob/DataRepoTemp/AMR-Data/amr_annotation_3.0'
    test_fn  = 'data/tdata/test.json.gz'
    train_fn = 'data/tdata/train.json.gz'

    os.makedirs(os.path.dirname(train_fn), exist_ok=True)

    print('Processing data for testing')
    data = build_coref_tdata(amr3_dir, is_train=False)
    print('Writing data to', test_fn)
    dump_json(data, test_fn)
    print_stats(data)
    print()

    print('Processing data for training')
    data = build_coref_tdata(amr3_dir, is_train=True)
    print('Writing data to', train_fn)
    dump_json(data, train_fn)
    print_stats(data)
