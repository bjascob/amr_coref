#!/usr/bin/python3
import os
import logging
from   amr_coref.coref.build_embeddings import build_embeddings
from   amr_coref.utils.logging import setup_logging, silence_penman
from   amr_coref.coref.word_vectors import load_embeddings, save_embeddings
from   amr_coref.utils.data_utils import load_json


# Note: right now this builds the vocabulary / embeddings on the training and test data
# but for more general use, this should probably use the entire amr corpus
if __name__ == '__main__':
    setup_logging(logfname='logs/build_embeddings.log', level=logging.WARN)
    silence_penman()
    coref_fpath_train = 'data/tdata/train.json.gz'
    coref_fpath_test  = 'data/tdata/test.json.gz'
    embed_in_fpath    = 'data/GloVe/glove.6B.50d.txt'
    embed_out_fpath   = 'data/tdata/embeddings.txt'

    os.makedirs(os.path.dirname(embed_out_fpath), exist_ok=True)

    # Load all the graph data from the coref files
    gdata_dict = {}
    for coref_fpath in [coref_fpath_test, coref_fpath_train]:
        print('Loading mention data from', coref_fpath)
        data = load_json(coref_fpath)
        for sent_id, gdata in data['gdata'].items():
            assert sent_id not in gdata_dict
            gdata_dict[sent_id] = gdata

    # Load the raw word vectors
    print('Loading embeddings from', embed_in_fpath)
    embed_in_dict, _ = load_embeddings(embed_in_fpath)
    print()

    # Using the pretrained embeddings new set off of the vocabular from the graphs
    embed_out_dict = build_embeddings(embed_in_dict, gdata_dict)

    # Save the data
    print('Saving embeddings to', embed_out_fpath)
    save_embeddings(embed_out_dict, embed_out_fpath)
