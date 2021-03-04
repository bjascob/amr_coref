# amr_coref

**A python library / model for creating co-references between AMR graph nodes.**

## About
amr_coref is a python library and trained model designed to do co-referencing
between [Abstract Meaning Representation](https://amr.isi.edu/) graphs.

The code was initially based on the [neuralcoref project](https://github.com/huggingface/neuralcoref)
and it's excellent
[blog on the co-referencing](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe).

The trained model achieves the following scores
```
MUC   :  R=0.647  P=0.779  F₁=0.706
B³    :  R=0.633  P=0.638  F₁=0.630
CEAF_m:  R=0.515  P=0.744  F₁=0.609
CEAF_e:  R=0.200  P=0.734  F₁=0.306
BLANC :  R=0.524  P=0.799  F₁=0.542
CoNLL-2012 average score: 0.548
```

## Installation and usage
There is currently no pip installation. To use the library, simply clone the code and use it in place.

The pre-trained model can be downloaded from [releases](https://github.com/bjascob/amrlib/releases)
and downloaded from the assets section.

To use the model create a `data` directory and un-tar the model in it.

The script `40_Run_Inference.py`, is an example of how to use the model.


## Training
If you'd like to train the model from scratch, you'll need a copy of the AMR corpus.
To complete training run the scripts in order from `10_Build_Model_TData.py` through
`30_Train_model.py`.  Training takes approximately 4 hours.
