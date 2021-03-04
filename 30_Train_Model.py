#!/usr/bin/python3
import os
import logging
import shutil
from   amr_coref.utils.logging import setup_logging
from   amr_coref.coref.trainer import Trainer


# Helper function to decode config.all_pair_x, config.top_pair_x)
def decode_config(config, loss_type):
    nepochs = getattr(config, loss_type + '_epochs')
    lr      = getattr(config, loss_type + '_lr')
    wd      = getattr(config, loss_type + '_wd')
    return nepochs, lr, wd

# Run the training with specific params
def run_training(loss_type, new_model, load_optim=False, debug=False, **kwargs):
    setup_logging(logfname='logs/train_'+loss_type+'.log', level=logging.WARN)
    model_dir      = kwargs.get('model_dir', 'data/model')
    trn_logfn      = kwargs.get('trn_logfn', 'train_'+loss_type+'.log')
    config_fn      = kwargs.get('config_fn', 'configs/config_01.json')
    graph_embed_fn = 'data/tdata/embeddings.txt'
    mention_set_fn = 'data/tdata/mention_tokens.txt'
    train_fn       = 'data/tdata/train.json.gz'
    test_fn        = 'data/tdata/test.json.gz'
    # Create a new model or load an existing one
    if new_model:
        trainer = Trainer.from_scratch(model_dir, config_fn, graph_embed_fn, mention_set_fn, trn_logfn)
        nepochs, lr, wd = decode_config(trainer.config, loss_type)
        trainer.set_optimizer(lr, wd)
    else:
        trainer = Trainer.from_pretrained(model_dir, trn_logfn)
        nepochs, lr, wd = decode_config(trainer.config, loss_type)
        trainer.set_optimizer(lr, wd)
        if load_optim:
            trainer.optimizer.load_state_dict(trainer.model.optimizer_state_dict)
    # Run the model or just run a quick test for debug
    if not debug:
        trainer.setup_test_data(test_fn)
        trainer.setup_train_data(train_fn)
    else:
        # Use short dataset for training, no test setup, 1 epoch and no saving
        trainer.setup_train_data(test_fn)
        trainer.config.all_pair_epochs = 1
        trainer.config.top_pair_epochs = 1
        trainer.config.ranking_epochs  = 1
        trainer.config.save_interval   = 999999
    # Do the actual training
    trainer.train(loss_type, nepochs)


if __name__ == '__main__':
    #run_training(Trainer.all_pair, new_model=True, debug=True)
    run_training(Trainer.all_pair, new_model=True)
    # shutil.copytree('data/model', 'data/model_p1')
    #run_training(Trainer.top_pair, new_model=False, load_optim=False)
    # shutil.copytree('data/model', 'data/model_p2')
    #run_training(Trainer.ranking,  new_model=False, load_optim=False)
    # shutil.copytree('data/model', 'data/model_p3')
