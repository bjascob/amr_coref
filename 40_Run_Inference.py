#!/usr/bin/python3
import os
import penman
from   penman.models.noop import NoOpModel
from   amr_coref.coref.inference import Inference


# Gather up a set of graphs that compose a document for coreferencing
# Use ones from AMR "multisentence" with gold data as an inference example
# Note that ordering is important and graphs must be loaded model=NoOpModel()
def gather_test_graphs():
    # These are for amr_annotation_3.0/data/multisentence/ms-amr-split/test/msamr_dfa_007.xml
    fn = 'data/amr_annotation_3.0/data/amrs/unsplit/amr-release-3.0-amrs-dfa.txt'
    gids = ["DF-200-192400-625_7046.1",  "DF-200-192400-625_7046.2",  "DF-200-192400-625_7046.3",
            "DF-200-192400-625_7046.4",  "DF-200-192400-625_7046.5",  "DF-200-192400-625_7046.6",
            "DF-200-192400-625_7046.7",  "DF-200-192400-625_7046.8",  "DF-200-192400-625_7046.9",
            "DF-200-192400-625_7046.10", "DF-200-192400-625_7046.11", "DF-200-192400-625_7046.12",
            "DF-200-192400-625_7046.13", "DF-200-192400-625_7046.14", "DF-200-192400-625_7046.15",
            "DF-200-192400-625_7046.16", "DF-200-192400-625_7046.17", "DF-200-192400-625_7046.18"]
    # Load the AMR file with penman and then extract the specific ids and put them in order
    pgraphs = penman.load(fn, model=NoOpModel())
    ordered_pgraphs = [None]*len(gids)
    for pgraph in pgraphs:
        gid = pgraph.metadata['id']
        doc_idx = gids.index(gid) if gid in gids else None
        if doc_idx is not None:
            ordered_pgraphs[doc_idx] = pgraph
    assert None not in ordered_pgraphs
    return ordered_pgraphs


# Simple function to print a list a strings in columns
def print_list_of_strings(items, col_w, max_w):
    cur_len = 0
    fmt = '%%-%ds' % col_w  # ie.. fmt = '%-8s'
    print('  ', end='')
    for item in items:
        print(fmt % item, end='')
        cur_len += 8
        if cur_len > max_w:
            print('\n  ', end='')
            cur_len = 0
    if cur_len != 0:
        print()


if __name__ == '__main__':
    device    = 'cpu'   # 'cuda:0'
    model_dir = 'data/model_coref-v0.1.0/'

    # Load the model and test data
    print('Loading model from %s' % model_dir)
    inference = Inference(model_dir, device=device)

    # Get test data
    print('Loading test data')
    ordered_pgraphs = gather_test_graphs()

    # Cluster the data
    # This returns cluster_dict[relation_string] = [(graph_idx, variable), ...]
    print('Clustering')
    cluster_dict = inference.coreference(ordered_pgraphs)
    print()

    # Print out the clusters
    print('Clusters')
    for relation, clusters in cluster_dict.items():
        print(relation)
        cid_strings = ['%d.%s' % (graph_idx, var) for (graph_idx, var) in clusters]
        print_list_of_strings(cid_strings, col_w=8, max_w=120)
