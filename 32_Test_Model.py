#!/usr/bin/python3
import os
import pickle
from   amr_coref.coref.tester import Tester
from   amr_coref.evaluate import scorch
from   amr_coref.evaluate.pr_scorer import PRScorerForSets
from   amr_coref.coref.clustering import cluster_and_save_sdata


# Convert the clusters into a flat list of unique keys for scoring
def prscore_cluster_dicts(cluster_dicts):
    scorer   = PRScorerForSets()
    gold_set = set()
    pred_set = set()
    for cd in cluster_dicts:
        doc_name = cd['doc_name']
        for rel_str, mlist in cd['gold'].items():
            for mention in mlist:
                key = '%s.%d.%d' % (mention.doc_name, mention.sent_idx, mention.tok_idx)
                scorer.add_gold(key)
        for rel_str, mlist in cd['pred'].items():
            for mention in mlist:
                key = '%s.%d.%d' % (mention.doc_name, mention.sent_idx, mention.tok_idx)
                scorer.add_pred(key)
    return scorer


if __name__ == '__main__':
    model_dir      = 'data/model'
    results_dir    = 'data/test'
    #results_dir    = os.path.join(model_dir, 'coref_test')
    #test_fn        = 'data/tdata/train.json.gz'
    test_fn        = 'data/tdata/test.json.gz'
    scores_fn      = os.path.join(results_dir, 'scores.txt')
    dev_probs_fn   = None #os.path.join(results_dir, 'predict_probs.pkl')
    max_dist       = None  # None => use value in model.config
    greedyness     = 0.0   # +/- 0.0 to 1.0 (higher positive ==> more antecedents)

    # Load the model and test data
    print('Loading model from %s and data from %s' % (model_dir, test_fn))
    tester = Tester.from_file(model_dir, test_fn, max_dist=max_dist)

    # Test and save of debug / development
    print('Running test')
    results = tester.run_test()
    if dev_probs_fn is not None:
        os.makedirs(os.path.dirname(dev_probs_fn), exist_ok=True)
        print('Development data written to', dev_probs_fn)
        with open(dev_probs_fn, 'wb') as f:
            pickle.dump(results, f)
    print()

    # Precision / recall scores on label data
    single_scores, pair_scores = tester.get_precision_recall_scores(results)
    print('Precision/Recall on dataset labels')
    print('Single: ', single_scores)
    print('Pair:   ', pair_scores)
    print('Counts: ', ' '.join('{:>8s}'.format(s) for s in ('tp', 'tn', 'fp', 'fn')))
    print('Single: ', ' '.join('{:8,}'.format(v) for v in single_scores.get_counts()))
    print('Pair:   ', ' '.join('{:8,}'.format(v) for v in pair_scores.get_counts()))
    print()

    # Clustering scores
    # CoNLL-2012 average score is the average of the F1 for MUC, B-cubed and CEAF_e
    print('Clustering Scores. written to:', scores_fn)
    gold_dir, pred_dir, cluster_dicts = cluster_and_save_sdata(tester.mdata, results['s_probs'],
                                        results['p_probs'], results_dir, greedyness=greedyness)
    scores = scorch.get_scores(gold_dir, pred_dir)
    scores_string = scorch.scores_to_string(scores)
    print(scores_string)
    with open(scores_fn, 'w') as f:
        f.write(scores_string)

    # Precision / Recall stats on clusters
    scorer = prscore_cluster_dicts(cluster_dicts)
    len_gold, len_pred, num_intersect, num_missing = scorer.get_counts()
    print('Precision/Recall on clusters')
    print(scorer)
    print('gold:%d   pred:%d  intersecting:%d  missing:%d' % \
        (len_gold, len_pred, num_intersect, num_missing))
    print()

    # Because I can never remember these definitions
    print('precision: number correct out of the number predicted = tp/(tp + fp)')
    print('recall:    number correct out of the number of gold   = tp/(tp + fn)')
    print()
